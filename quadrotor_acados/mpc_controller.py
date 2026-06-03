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

from .math_utils import (
    quaternion_inverse,
    skew_symmetric,
    v_dot_q,
)
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
        quat_arr = quat_arr / norm
        if quat_arr[0] < 0.0:
            quat_arr *= -1.0
        return quat_arr

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
        if q_cost.shape != (12,):
            raise ValueError(f"q_cost must contain 12 weights, got shape {q_cost.shape}")
        if r_cost.shape != (4,):
            raise ValueError(f"r_cost must contain 4 weights, got shape {r_cost.shape}")

        self.T = t_horizon
        self.N = n_nodes
        self.quad = quad
        self.max_u = quad.max_input_value
        self.min_u = quad.min_input_value
        self.hover_u = quad.mass * 9.81 / 4 * self.quad.max_thrust
        self.dt = self.T / self.N
        self.control_dt = self.dt
        self.substeps = 1
        self.acados_codegen_root = Path(acados_codegen_root).expanduser().resolve()
        if expected_frequency is not None:
            self.substeps = math.ceil(expected_frequency * self.dt)
            self.control_dt = 1.0 / expected_frequency

        self.p = cs.MX.sym("p", 3)
        self.q = cs.MX.sym("a", 4)
        self.v = cs.MX.sym("v", 3)
        self.r = cs.MX.sym("r", 3)
        self.preferred_step = None

        self.x = cs.vertcat(self.p, self.q, self.v, self.r)
        self.state_dim = 13

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

        q_diagonal = np.concatenate(
            (q_cost[:3], np.mean(q_cost[3:6])[np.newaxis], q_cost[3:])
        )
        if q_mask is not None:
            q_mask = np.concatenate((q_mask[:3], np.zeros(1), q_mask[3:]))
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
            terminal_cost = (
                0
                if solver_options is None
                or not solver_options.get("terminal_cost", False)
                else 1
            )
            ocp.cost.W_e *= terminal_cost

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
        self.time_traj = None
        self.last_closest_index = 0

        self.logger = logger
        if self.logger:
            self.logger.info(f"Controller time horizon = {t_horizon}")
            self.logger.info(f"Controller steps = {n_nodes}")
            self.logger.info(f"Controller dt = {self.dt}")
            self.logger.info(f"Controller integration dt = {self.control_dt}")
            self.logger.info(f"acados codegen root = {self.acados_codegen_root}")

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
            self.p_dynamics(),
            self.q_dynamics(),
            self.v_dynamics(rdrv_d),
            self.w_dynamics(),
        )
        return cs.Function(
            "x_dot", [self.x[:13], self.u], [x_dot], ["x", "u"], ["x_dot"]
        )

    def p_dynamics(self):
        return self.v

    def q_dynamics(self):
        return 0.5 * cs.mtimes(skew_symmetric(self.r), self.q)

    def v_dynamics(self, rdrv_d):
        f_thrust = self.u * self.quad.max_thrust
        g = cs.vertcat(0.0, 0.0, -9.81)
        a_thrust = (
            cs.vertcat(
                0.0, 0.0, (f_thrust[0] + f_thrust[1] + f_thrust[2] + f_thrust[3])
            )
            / self.quad.mass
        )

        v_dyn = v_dot_q(a_thrust, self.q) + g

        if rdrv_d is not None:
            v_b = v_dot_q(self.v, quaternion_inverse(self.q))
            rdrv_drag = v_dot_q(cs.mtimes(rdrv_d, v_b), self.q)
            v_dyn += rdrv_drag

        return v_dyn

    def w_dynamics(self):
        f_thrust = self.u * self.quad.max_thrust

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

    def update_trajectory(
        self, trajectory: np.ndarray, preferred_speed: float | None = None
    ):
        traj = self._normalize_reference_trajectory(trajectory)
        if preferred_speed is None:
            self.preferred_step = None
            self.time_traj = traj
        else:
            self.preferred_step = preferred_speed * self.T / self.N / self.substeps
            self.time_traj = self._resample_reference_trajectory(
                traj, self.preferred_step
            )

        self.time_traj[:, :3] = np.array(
            [self.ned_to_xyz_vector(point) for point in self.time_traj[:, :3]]
        )
        if self.time_traj.shape[1] >= 7:
            self.time_traj[:, 3:7] = np.array(
                [self.ned_to_xyz_quat(quat) for quat in self.time_traj[:, 3:7]]
            )
        if self.time_traj.shape[1] >= 10:
            self.time_traj[:, 7:10] = np.array(
                [self.ned_to_xyz_vector(vel) for vel in self.time_traj[:, 7:10]]
            )
        if self.time_traj.shape[1] >= 13:
            self.time_traj[:, 10:13] = np.array(
                [self.ned_to_xyz_vector(rate) for rate in self.time_traj[:, 10:13]]
            )

        # self.logger.info(f"Got new trajectory = {self.time_traj}")
        self.last_closest_index = 0

    def _normalize_reference_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        traj = np.array(trajectory, dtype=float, copy=True)
        if traj.ndim == 1:
            traj = traj.reshape(1, -1)
        if traj.ndim != 2 or traj.shape[1] < 3:
            raise ValueError(
                f"Trajectory must be Nx3 or wider, got shape {traj.shape}."
            )
        if traj.shape[1] == 4:
            traj = np.column_stack((traj[:, :3], self.yaw_to_quat(traj[:, 3])))
        elif 4 < traj.shape[1] < 7:
            raise ValueError(
                "Trajectory with attitude must use Nx4 xyz+yaw or Nx7 xyz+quaternion."
            )
        return traj

    def _resample_reference_trajectory(
        self, traj: np.ndarray, preferred_step: float
    ) -> np.ndarray:
        if len(traj) < 2 or preferred_step <= 0.0:
            return np.array(traj, dtype=float, copy=True)

        segment_lengths = np.linalg.norm(np.diff(traj[:, :3], axis=0), axis=1)
        total_length = float(np.sum(segment_lengths))
        if total_length <= 1e-9:
            return np.array(traj, dtype=float, copy=True)

        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        sample_distances = np.arange(0.0, total_length, preferred_step)
        if len(sample_distances) == 0 or not np.isclose(
            sample_distances[-1], total_length
        ):
            sample_distances = np.append(sample_distances, total_length)

        rows = [
            self._interpolate_reference_row(traj, cumulative, segment_lengths, distance)
            for distance in sample_distances
        ]
        return np.vstack(rows)

    def _interpolate_reference_row(
        self,
        traj: np.ndarray,
        cumulative: np.ndarray,
        segment_lengths: np.ndarray,
        distance: float,
    ) -> np.ndarray:
        if distance >= cumulative[-1]:
            return traj[-1].copy()

        segment_idx = max(0, np.searchsorted(cumulative, distance, side="right") - 1)
        while (
            segment_idx < len(segment_lengths) - 1
            and segment_lengths[segment_idx] <= 1e-9
        ):
            segment_idx += 1

        segment_length = segment_lengths[segment_idx]
        if segment_length <= 1e-9:
            return traj[segment_idx].copy()

        t = (distance - cumulative[segment_idx]) / segment_length
        start = traj[segment_idx]
        end = traj[segment_idx + 1]
        row = start + t * (end - start)
        row[:3] = start[:3] + t * (end[:3] - start[:3])
        if traj.shape[1] >= 7:
            row[3:7] = self.slerp_quat(start[3:7], end[3:7], float(t))
        return row

    def _reference_state_from_point(self, point: np.ndarray) -> np.ndarray:
        reference = np.zeros(13, dtype=float)
        reference[:3] = point[:3]
        if len(point) >= 7:
            reference[3:7] = self.normalize_quat(point[3:7])
        else:
            reference[3] = 1.0
        if len(point) >= 10:
            reference[7:10] = point[7:10]
        if len(point) >= 13:
            reference[10:13] = point[10:13]
        return reference

    def state_ned_to_xyz(self, state: np.ndarray) -> np.ndarray:
        xyz_state = np.array(state, dtype=float, copy=True)
        xyz_state[0:3] = self.ned_to_xyz_vector(xyz_state[0:3])
        xyz_state[3:7] = self.ned_to_xyz_quat(xyz_state[3:7])
        xyz_state[7:10] = self.ned_to_xyz_vector(xyz_state[7:10])
        xyz_state[10:13] = self.ned_to_xyz_vector(xyz_state[10:13])
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
        if self.time_traj is None or len(self.time_traj) == 0:
            return np.zeros(4)

        if initial_state is None:
            initial_state = [0, 0, 0] + [1, 0, 0, 0] + [0, 0, 0] + [0, 0, 0]

        x_init = self.state_ned_to_xyz(initial_state)

        self.acados_ocp_solver.set(0, "lbx", x_init)
        self.acados_ocp_solver.set(0, "ubx", x_init)

        starting_index = (
            np.argmin(
                np.sum(
                    (self.time_traj[self.last_closest_index :, :3] - x_init[:3]) ** 2,
                    axis=1,
                )
            )
            + self.last_closest_index
        )
        starting_row = self.time_traj[starting_index].copy()
        starting_row[:3] = x_init[:3]
        if len(starting_row) >= 7:
            starting_row[3:7] = x_init[3:7]
        if len(starting_row) >= 10:
            starting_row[7:10] = x_init[7:10]
        if len(starting_row) >= 13:
            starting_row[10:13] = x_init[10:13]
        starting_trajectory = np.vstack(
            [starting_row, self.time_traj[starting_index]]
        )

        if self.preferred_step is not None:
            starting_trajectory = self._resample_reference_trajectory(
                starting_trajectory, self.preferred_step
            )

        full_trajectory = np.vstack(
            (
                starting_trajectory,
                self.time_traj[
                    starting_index + 1 : starting_index + self.N * self.substeps
                ],
            )
        )
        used_substeps = self.substeps
        local_trajectory = []
        while (len(local_trajectory) < self.N + 1) and (
            used_substeps * 2 > self.substeps
        ):
            local_trajectory = full_trajectory[
                : self.N * used_substeps + 1 : used_substeps
            ]
            used_substeps -= 1
        used_substeps += 1

        if len(local_trajectory) < self.N + 1:
            pad_len = self.N + 1 - len(local_trajectory)
            pad_value = self.time_traj[-1]
            pad_rows = np.repeat(
                pad_value[None, :],
                pad_len,
                axis=0
            )
            local_trajectory = np.vstack([local_trajectory, pad_rows])
        self.last_closest_index = starting_index

        for j in range(self.N):
            y_ref = np.concatenate(
                (
                    self._reference_state_from_point(local_trajectory[j]),
                    np.array(
                        [self.hover_u, self.hover_u, self.hover_u, self.hover_u],
                        dtype=float,
                    ),
                )
            )
            self.acados_ocp_solver.set(j, "yref", y_ref)

        y_refN = self._reference_state_from_point(local_trajectory[self.N])
        self.acados_ocp_solver.set(self.N, "yref", y_refN)

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
