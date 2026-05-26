import casadi as cs
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
    transform_trajectory,
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
        solver_options=None,
        enable_integrator: bool = False,
        logger=None,
    ):
        if q_cost is None:
            q_cost = np.array(
                [10.0, 10.0, 30.0, 0.05, 0.05, 10.0, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05]
            )
        if r_cost is None:
            r_cost = np.array([0.01, 0.01, 0.01, 0.01])

        self.T = t_horizon
        self.N = n_nodes
        self.quad = quad
        self.max_u = quad.max_input_value
        self.min_u = quad.min_input_value
        self.dt = self.T / self.N
        self.control_dt = self.dt
        self.substeps = 1
        if expected_frequency is not None:
            self.substeps = math.ceil(expected_frequency * self.dt)
            self.control_dt = 1.0 / expected_frequency

        self.p = cs.MX.sym("p", 3)
        self.q = cs.MX.sym("a", 4)
        self.v = cs.MX.sym("v", 3)
        self.r = cs.MX.sym("r", 3)

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

            json_file = str(f"{key_model.name}_acados_ocp.json")
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

    def create_integrator(self, model: AcadosModel) -> AcadosSimSolver:
        sim = AcadosSim()
        sim.model = model
        sim.solver_options.T = self.control_dt
        sim.solver_options.integrator_type = "ERK"
        sim.solver_options.num_stages = 4
        sim.solver_options.num_steps = 1
        json_file = str(f"{model.name}_acados_sim.json")
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
        traj = np.array(trajectory, dtype=float, copy=True)
        if preferred_speed is None:
            self.time_traj = traj
        else:
            self.time_traj = transform_trajectory(
                traj, preferred_speed * self.T / self.N / self.substeps
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
                    (self.time_traj[self.last_closest_index :] - x_init[:3]) ** 2,
                    axis=1,
                )
            )
            + self.last_closest_index
        )

        local_trajectory = []
        used_substeps = self.substeps
        while (len(local_trajectory) < self.N + 1) and (used_substeps * 2 > self.substeps):
            local_trajectory = self.time_traj[starting_index : starting_index + self.N * used_substeps + 1 : used_substeps]
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
            y_ref = np.array(
                [
                    local_trajectory[j, 0],
                    local_trajectory[j, 1],
                    local_trajectory[j, 2],
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )
            self.acados_ocp_solver.set(j, "yref", y_ref)

        y_refN = np.array(
            [
                local_trajectory[self.N, 0],
                local_trajectory[self.N, 1],
                local_trajectory[self.N, 2],
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        )
        self.acados_ocp_solver.set(self.N, "yref", y_refN)

        self.acados_ocp_solver.solve()

        w_opt_acados = np.ndarray((self.N, 4))
        x_opt_acados = np.ndarray((self.N + 1, len(x_init)))
        x_opt_acados[0, :] = self.acados_ocp_solver.get(0, "x")
        for i in range(self.N):
            w_opt_acados[i, :] = self.acados_ocp_solver.get(i, "u")
            x_opt_acados[i + 1, :] = self.acados_ocp_solver.get(i + 1, "x")
        w_opt_acados = np.reshape(w_opt_acados, (-1))

        return w_opt_acados[:4], x_opt_acados[1]
