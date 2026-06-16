import casadi as cs
from pathlib import Path
import numpy as np
from acados_template import (
    AcadosModel,
    AcadosOcp,
    AcadosOcpSolver,
    AcadosSim,
    AcadosSimSolver,
)
from pyquaternion import Quaternion

from .math_utils import skew_symmetric
from .quadrotor_model import QuadrotorParams
import math

class Controller:
    FRAME_TRANSFORM = np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=float
    )

    @staticmethod
    def ned_to_xyz_vector(vec: np.ndarray) -> np.ndarray:
        return Controller.FRAME_TRANSFORM @ np.asarray(vec, dtype=float)

    @staticmethod
    def xyz_to_ned_vector(vec: np.ndarray) -> np.ndarray:
        return Controller.FRAME_TRANSFORM @ np.asarray(vec, dtype=float)

    @staticmethod
    def quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
        quat_arr = np.asarray(quat, dtype=float)
        norm = np.linalg.norm(quat_arr)
        if norm == 0.0:
            return np.eye(3)
        quat_obj = Quaternion(quat_arr / norm)
        return np.array(quat_obj.rotation_matrix, dtype=float)

    @staticmethod
    def rotmat_to_quat(rotmat: np.ndarray) -> np.ndarray:
        quat_obj = Quaternion(matrix=np.asarray(rotmat, dtype=float)).normalised
        quat = np.array([quat_obj.w, quat_obj.x, quat_obj.y, quat_obj.z], dtype=float)
        if quat[0] < 0.0:
            quat *= -1.0
        return quat

    @staticmethod
    def ned_to_xyz_quat(quat: np.ndarray) -> np.ndarray:
        rotmat_ned = Controller.quat_to_rotmat(quat)
        rotmat_xyz = (
            Controller.FRAME_TRANSFORM @ rotmat_ned @ Controller.FRAME_TRANSFORM.T
        )
        return Controller.rotmat_to_quat(rotmat_xyz)

    @staticmethod
    def yaw_to_quat(yaw: np.ndarray) -> np.ndarray:
        yaw = np.asarray(yaw, dtype=float)
        return np.column_stack(
            (
                np.cos(0.5 * yaw),
                np.zeros_like(yaw),
                np.zeros_like(yaw),
                np.sin(0.5 * yaw),
            )
        )

    @staticmethod
    def normalize_quat(quat: np.ndarray) -> np.ndarray:
        quat_arr = np.asarray(quat, dtype=float)
        norm = np.linalg.norm(quat_arr)
        if norm == 0.0:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        return quat_arr / norm

    @staticmethod
    def make_quat_sequence_continuous(quats: np.ndarray) -> np.ndarray:
        continuous = np.array(quats, dtype=float, copy=True)
        if len(continuous) == 0:
            return continuous

        continuous[0] = Controller.normalize_quat(continuous[0])
        for idx in range(1, len(continuous)):
            continuous[idx] = Controller.normalize_quat(continuous[idx])
            if np.dot(continuous[idx - 1], continuous[idx]) < 0.0:
                continuous[idx] *= -1.0
        return continuous

    @staticmethod
    def slerp_quat(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
        q0 = Controller.normalize_quat(q0)
        q1 = Controller.normalize_quat(q1)
        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        dot = np.clip(dot, -1.0, 1.0)
        if dot > 0.9995:
            return Controller.normalize_quat(q0 + t * (q1 - q0))

        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * t
        return (
            np.sin(theta_0 - theta) / sin_theta_0 * q0
            + np.sin(theta) / sin_theta_0 * q1
        )

    def __init__(
        self,
        quad: QuadrotorParams,
        t_horizon = 1.0,
        n_nodes = 10,
        expected_frequency = None,
        max_body_rate=None,
        q_cost=None,
        r_cost=None,
        q_mask=None,
        rdrv_d_mat=None,
        model_name: str = "quad_3d_acados_mpc",
        acados_codegen_root: str | Path = "/tmp/ros2_quadcopter_acados_codegen",
        solver_options=None,
        enable_integrator: bool = False,
        logger=None,
    ):
        if q_cost is None or r_cost is None:
            raise ValueError("q_cost and r_cost must be provided")
        q_cost = np.asarray(q_cost, dtype=float)
        r_cost = np.asarray(r_cost, dtype=float)
        if q_cost.shape != (6,):
            raise ValueError(f"q_cost must contain 6 weights, got shape {q_cost.shape}")
        if r_cost.shape != (4,):
            raise ValueError(f"r_cost must contain 4 weights, got shape {r_cost.shape}")
        max_body_rate = self._validate_speed_limits(
            "max_body_rate", max_body_rate
        )

        self.T = t_horizon
        self.N = n_nodes
        self.quad = quad
        self.max_u = quad.max_input_value
        self.min_u = quad.min_input_value
        self.hover_u = (
            (quad.mass * 9.81 / 4.0 - self.quad.min_thrust)
            / (self.quad.max_thrust - self.quad.min_thrust)
        )
        self.hover_u = float(np.clip(self.hover_u, self.min_u, self.max_u))
        self.max_body_rate = self._ned_limits_to_xyz(max_body_rate)
        self.dt = self.T / self.N
        self.control_dt = self.dt
        self.substeps = 1
        self.acados_codegen_root = Path(acados_codegen_root).expanduser().resolve()
        if expected_frequency is not None:
            self.substeps = math.ceil(expected_frequency * self.dt)
            self.control_dt = 1.0 / expected_frequency

        self.q = cs.MX.sym("a", 4)
        self.r = cs.MX.sym("r", 3)
        self.target_attitude = None
        self.target_common_thrust = self.hover_u

        self.x = cs.vertcat(self.q, self.r)
        self.state_dim = 7

        u1 = cs.MX.sym("u1")
        u2 = cs.MX.sym("u2")
        u3 = cs.MX.sym("u3")
        u4 = cs.MX.sym("u4")
        self.u = cs.vertcat(u1, u2, u3, u4)

        self.quad_xdot_nominal = self.quad_dynamics(rdrv_d_mat)
        acados_models, nominal_with_gp = self.acados_setup_model(
            self.quad_xdot_nominal(x=self.x, u=self.u)["x_dot"], model_name
        )

        self.quad_xdot = {}
        for dyn_model_idx in nominal_with_gp.keys():
            dyn = nominal_with_gp[dyn_model_idx]
            self.quad_xdot[dyn_model_idx] = cs.Function(
                "x_dot", [self.x, self.u], [dyn], ["x", "u"], ["x_dot"]
            )

        q_diagonal = np.concatenate(([0.0], q_cost[:3], q_cost[3:]))
        if q_mask is not None:
            q_mask = np.asarray(q_mask, dtype=float)
            if q_mask.shape != (6,):
                raise ValueError(
                    f"q_mask must contain 6 values, got shape {q_mask.shape}"
                )
            q_mask = np.concatenate(([0.0], q_mask))
            q_diagonal *= q_mask

        self.model = None
        for key_model in acados_models.values():
            self.model = key_model
            nx = key_model.x.size()[0]
            nu = key_model.u.size()[0]
            ny = nx + nu
            n_param = key_model.p.size()[0] if isinstance(key_model.p, cs.MX) else 0

            ocp = AcadosOcp()
            ocp.model = key_model
            ocp.dims.N = self.N
            ocp.solver_options.tf = t_horizon

            ocp.dims.np = n_param
            ocp.parameter_values = np.zeros(n_param)

            ocp.cost.cost_type = "LINEAR_LS"
            ocp.cost.cost_type_e = "LINEAR_LS"

            ocp.cost.W = np.diag(np.concatenate((q_diagonal, r_cost)))
            ocp.cost.W_e = np.diag(q_diagonal)

            ocp.cost.Vx = np.zeros((ny, nx))
            ocp.cost.Vx[:nx, :nx] = np.eye(nx)
            ocp.cost.Vu = np.zeros((ny, nu))
            ocp.cost.Vu[-4:, -4:] = np.eye(nu)
            ocp.cost.Vx_e = np.eye(nx)

            x_ref = np.zeros(nx)
            ocp.cost.yref = np.concatenate((x_ref, np.array([0.0, 0.0, 0.0, 0.0])))
            ocp.cost.yref_e = x_ref
            ocp.constraints.x0 = x_ref

            ocp.constraints.lbu = np.array([self.min_u] * 4)
            ocp.constraints.ubu = np.array([self.max_u] * 4)
            ocp.constraints.idxbu = np.array([0, 1, 2, 3])

            rate_state_indices = np.arange(4, 7, dtype=int)
            ocp.constraints.idxbx = rate_state_indices
            ocp.constraints.lbx = -self.max_body_rate
            ocp.constraints.ubx = self.max_body_rate
            ocp.constraints.idxbx_e = rate_state_indices
            ocp.constraints.lbx_e = -self.max_body_rate
            ocp.constraints.ubx_e = self.max_body_rate

            ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
            ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
            ocp.solver_options.integrator_type = "ERK"
            ocp.solver_options.print_level = 0
            ocp.solver_options.nlp_solver_type = (
                "SQP_RTI"
                if solver_options is None
                else solver_options.get("solver_type", "SQP_RTI")
            )

            code_export_dir, json_file = self._acados_codegen_paths(
                key_model.name, "ocp"
            )
            ocp.code_gen_opts.code_export_directory = str(code_export_dir)
            self.acados_ocp_solver = AcadosOcpSolver(ocp, json_file=json_file)

        self.acados_sim_solver = (
            self.create_integrator(self.model) if enable_integrator else None
        )
        self.logger = logger
        if self.logger:
            self.logger.info(f"Controller time horizon = {t_horizon}")
            self.logger.info(f"Controller steps = {n_nodes}")
            self.logger.info(f"Controller dt = {self.dt}")
            self.logger.info(f"Controller integration dt = {self.control_dt}")
            self.logger.info(
                f"Controller body-rate limits xyz = {self.max_body_rate}"
            )
            self.logger.info(f"acados codegen root = {self.acados_codegen_root}")

    @staticmethod
    def _validate_speed_limits(name: str, limits) -> np.ndarray:
        values = np.asarray(limits, dtype=float)
        if values.shape != (3,):
            raise ValueError(f"{name} must contain 3 values, got shape {values.shape}")
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError(f"{name} values must be finite and positive")
        return values

    @staticmethod
    def _ned_limits_to_xyz(limits: np.ndarray) -> np.ndarray:
        return np.abs(Controller.FRAME_TRANSFORM) @ limits

    def _acados_codegen_paths(
        self, model_name: str, solver_kind: str
    ) -> tuple[Path, str]:
        solver_dir = self.acados_codegen_root / model_name / solver_kind
        code_export_dir = solver_dir / f"{model_name}_{solver_kind}_generated_code"
        code_export_dir.mkdir(parents=True, exist_ok=True)
        json_file = solver_dir / f"{model_name}_acados_{solver_kind}.json"
        return code_export_dir, str(json_file)

    def create_integrator(self, model: AcadosModel) -> AcadosSimSolver:
        sim = AcadosSim()
        sim.model = model
        sim.solver_options.T = self.control_dt
        sim.solver_options.integrator_type = "ERK"
        sim.solver_options.num_stages = 4
        sim.solver_options.num_steps = 1
        code_export_dir, json_file = self._acados_codegen_paths(model.name, "sim")
        sim.code_gen_opts.code_export_directory = str(code_export_dir)
        return AcadosSimSolver(sim, json_file=json_file)

    def acados_setup_model(self, nominal, model_name):
        def fill_in_acados_model(x, u, p, dynamics, name):
            x_dot = cs.MX.sym("x_dot", dynamics.shape)
            f_impl = x_dot - dynamics

            model = AcadosModel()
            model.f_expl_expr = dynamics
            model.f_impl_expr = f_impl
            model.x = x
            model.xdot = x_dot
            model.u = u
            model.p = p
            model.name = name
            return model

        acados_models = {}
        dynamics_equations = {0: nominal}
        acados_models[0] = fill_in_acados_model(
            x=self.x, u=self.u, p=[], dynamics=nominal, name=model_name
        )
        return acados_models, dynamics_equations

    def quad_dynamics(self, rdrv_d):
        x_dot = cs.vertcat(
            self.q_dynamics(),
            self.w_dynamics(),
        )
        return cs.Function(
            "x_dot", [self.x, self.u], [x_dot], ["x", "u"], ["x_dot"]
        )

    def q_dynamics(self):
        return 0.5 * cs.mtimes(skew_symmetric(self.r), self.q)

    def w_dynamics(self):
        f_thrust = self.motor_command_to_thrust(self.u)

        x_f = cs.MX(self.quad.x_f)
        y_f = cs.MX(self.quad.y_f)
        c_f = cs.MX(self.quad.z_l_tau)

        return cs.vertcat(
            (
                cs.mtimes(f_thrust.T, x_f)
                - (self.quad.J[2] - self.quad.J[1]) * self.r[1] * self.r[2]
            )
            / self.quad.J[0],
            (
                cs.mtimes(f_thrust.T, y_f)
                - (self.quad.J[2] - self.quad.J[0]) * self.r[0] * self.r[2]
            )
            / self.quad.J[1],
            (
                cs.mtimes(f_thrust.T, c_f)
                - (self.quad.J[1] - self.quad.J[0]) * self.r[0] * self.r[1]
            )
            / self.quad.J[2],
        )

    def motor_command_to_thrust(self, u):
        return self.quad.min_thrust + u * (
            self.quad.max_thrust - self.quad.min_thrust
        )

    @property
    def has_target(self) -> bool:
        return self.target_attitude is not None

    def update_target(
        self, attitude_ned: np.ndarray, common_thrust: float
    ) -> None:
        attitude_ned = np.asarray(attitude_ned, dtype=float)
        if attitude_ned.shape != (4,):
            raise ValueError(
                f"target attitude must have shape (4,), got {attitude_ned.shape}"
            )
        if not np.all(np.isfinite(attitude_ned)):
            raise ValueError("target attitude must be finite")
        if not math.isfinite(common_thrust):
            raise ValueError("common_thrust must be finite")

        self.target_attitude = self.ned_to_xyz_quat(
            self.normalize_quat(attitude_ned)
        )
        self.target_common_thrust = float(
            np.clip(common_thrust, self.min_u, self.max_u)
        )

    def state_ned_to_xyz(self, state: np.ndarray) -> np.ndarray:
        xyz_state = np.asarray(state, dtype=float).copy()
        if xyz_state.shape == (13,):
            xyz_state = np.concatenate((xyz_state[3:7], xyz_state[10:13]))
        if xyz_state.shape != (7,):
            raise ValueError(
                f"angular state must have shape (7,), got {xyz_state.shape}"
            )
        xyz_state[0:4] = self.ned_to_xyz_quat(xyz_state[0:4])
        xyz_state[4:7] = self.ned_to_xyz_vector(xyz_state[4:7])
        return xyz_state

    def integrate_control_step(
        self, initial_state: np.ndarray, cmd: np.ndarray
    ) -> np.ndarray:
        if self.acados_sim_solver is None:
            self.acados_sim_solver = self.create_integrator(self.model)
        x_init = self.state_ned_to_xyz(initial_state)
        u = np.array(cmd[:4], dtype=float, copy=True)
        return np.asarray(self.acados_sim_solver.simulate(x=x_init, u=u), dtype=float)

    def run_optimization(self, initial_state=None):
        if initial_state is None:
            initial_state = [1, 0, 0, 0, 0, 0, 0]

        x_init = self.state_ned_to_xyz(initial_state)
        if not self.has_target:
            return np.zeros(4), x_init

        target_attitude = self.target_attitude.copy()
        if np.dot(x_init[:4], target_attitude) < 0.0:
            target_attitude *= -1.0
        reference = np.concatenate((target_attitude, np.zeros(3)))

        self.acados_ocp_solver.set(0, "lbx", x_init)
        self.acados_ocp_solver.set(0, "ubx", x_init)

        for j in range(self.N):
            y_ref = np.concatenate(
                (
                    reference,
                    np.full(4, self.target_common_thrust, dtype=float),
                )
            )
            self.acados_ocp_solver.set(j, "yref", y_ref)

        self.acados_ocp_solver.set(self.N, "yref", reference)

        self.acados_ocp_solver.solve()

        w_opt_acados = np.ndarray((self.N, 4))
        x_opt_acados = np.ndarray((self.N + 1, len(x_init)))
        x_opt_acados[0, :] = self.acados_ocp_solver.get(0, "x")
        for i in range(self.N):
            w_opt_acados[i, :] = self.acados_ocp_solver.get(i, "u")
            x_opt_acados[i + 1, :] = self.acados_ocp_solver.get(i + 1, "x")
        if self.logger:
            self.logger.info(
                f"Planned state trajectory: {x_opt_acados.tolist()}"
            )
        w_opt_acados = np.reshape(w_opt_acados, (-1))

        return w_opt_acados[:4], x_opt_acados[1]
