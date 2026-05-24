from threading import Lock

import numpy as np
import rclpy
from nav_msgs.msg import Path as PathMsg
from px4_msgs.msg import (
    ActuatorMotors,
    OffboardControlMode,
    VehicleCommand,
    VehicleOdometry,
    VehicleStatus,
)

try:
    from px4_msgs.msg import VehicleRatesSetpoint
except ImportError:
    VehicleRatesSetpoint = None
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from .mpc_controller import Controller
from .quadrotor_model import load_params


class Px4MpcNode(Node):
    def __init__(self):
        super().__init__(
            "px4_mpc_node", automatically_declare_parameters_from_overrides=True
        )

        self.path_topic = self.get_parameter("path_topic").value
        self.odom_topic = self.get_parameter("odometry_topic").value
        self.vehicle_status_topic = self.get_parameter("vehicle_status_topic").value
        self.actuator_topic = self.get_parameter("actuator_topic").value
        self.vehicle_rates_setpoint_topic = self._get_param(
            "vehicle_rates_setpoint_topic", "/fmu/in/vehicle_rates_setpoint"
        )
        self.command_output_mode = str(
            self._get_param("command_output_mode", "actuator_motors")
        ).lower()
        self.offboard_control_mode_topic = self.get_parameter(
            "offboard_control_mode_topic"
        ).value
        self.vehicle_command_topic = self.get_parameter("vehicle_command_topic").value
        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.offboard_control_rate_hz = float(
            self.get_parameter("offboard_control_rate_hz").value
        )
        self.preferred_speed = float(self.get_parameter("preferred_speed").value)
        horizon_sec = float(self.get_parameter("horizon_sec").value)
        horizon_nodes = int(self.get_parameter("horizon_nodes").value)
        self.min_thrust = 0
        self.max_thrust = 1
        if self.command_output_mode not in (
            "actuator_motors",
            "vehicle_rates_setpoint",
        ):
            raise ValueError(
                "command_output_mode must be 'actuator_motors' or "
                "'vehicle_rates_setpoint'"
            )
        if (
            self.command_output_mode == "vehicle_rates_setpoint"
            and VehicleRatesSetpoint is None
        ):
            raise ImportError(
                "px4_msgs.msg.VehicleRatesSetpoint is not available. "
                "Install a px4_msgs version that includes VehicleRatesSetpoint "
                "or use command_output_mode:=actuator_motors."
            )
        self.lock = Lock()
        self.current_state = None
        self.final_reference_point = None
        self.final_point_reached_radius = 0.25
        self.final_point_reached_logged = False
        self.is_armed = False
        self.offboard_setpoint_counter = 0
        self.arm_sequence_sent = False
        self.last_state_log_time_sec = -1.0
        self.last_body_rates = np.zeros(3, dtype=float)
        self.last_thrust = 0.0

        try:
            self.quad = load_params(self)
            self.controller = Controller(
                quad=self.quad,
                t_horizon=horizon_sec,
                n_nodes=horizon_nodes,
                logger=self.get_logger(),
                expected_frequency=self.control_rate_hz,
                enable_integrator=(
                    self.command_output_mode == "vehicle_rates_setpoint"
                ),
            )
        except Exception as exc:
            self.get_logger().error(f"Failed to initialize controller: {exc}")
            raise

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.path_sub = self.create_subscription(
            PathMsg, self.path_topic, self.path_callback, 10
        )
        self.odom_sub = self.create_subscription(
            VehicleOdometry, self.odom_topic, self.odom_callback, qos_sensor
        )
        self.vehicle_status_sub = self.create_subscription(
            VehicleStatus,
            self.vehicle_status_topic,
            self.vehicle_status_callback,
            qos_sensor,
        )
        self.motor_pub = self.create_publisher(
            ActuatorMotors, self.actuator_topic, qos_sensor
        )
        self.vehicle_rates_setpoint_pub = None
        if self.command_output_mode == "vehicle_rates_setpoint":
            self.vehicle_rates_setpoint_pub = self.create_publisher(
                VehicleRatesSetpoint, self.vehicle_rates_setpoint_topic, qos_sensor
            )
        self.offboard_control_mode_pub = self.create_publisher(
            OffboardControlMode, self.offboard_control_mode_topic, qos_sensor
        )
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand, self.vehicle_command_topic, qos_sensor
        )

        self.control_timer = self.create_timer(
            1.0 / self.control_rate_hz, self.control_loop
        )
        self.offboard_mode_timer = self.create_timer(
            1.0 / self.offboard_control_rate_hz, self.publish_offboard_control_mode
        )

        self.get_logger().info(
            "px4_mpc_node started. "
            f"path={self.path_topic}, odom={self.odom_topic}, vehicle_status={self.vehicle_status_topic}, "
            f"output_mode={self.command_output_mode}, actuator={self.actuator_topic}, "
            f"vehicle_rates_setpoint={self.vehicle_rates_setpoint_topic}, "
            f"offboard_control_mode={self.offboard_control_mode_topic}, "
            f"vehicle_command={self.vehicle_command_topic}"
        )

    def _get_param(self, name: str, default):
        if not self.has_parameter(name):
            self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def path_callback(self, msg: PathMsg) -> None:
        if not msg.poses:
            return

        trajectory = np.array(
            [
                [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]
                for pose in msg.poses
            ],
            dtype=float,
        )

        with self.lock:
            self.final_reference_point = trajectory[-1].copy()
            self.final_point_reached_logged = False
            self.controller.update_trajectory(
                trajectory, preferred_speed=self.preferred_speed
            )

        self.get_logger().info(f"Received trajectory with {len(trajectory)} waypoints")

    def odom_callback(self, msg: VehicleOdometry) -> None:
        position = np.array(
            [msg.position[0], msg.position[1], msg.position[2]], dtype=float
        )
        quat = np.array([msg.q[0], msg.q[1], msg.q[2], msg.q[3]], dtype=float)
        velocity = np.array(
            [msg.velocity[0], msg.velocity[1], msg.velocity[2]], dtype=float
        )
        angular_velocity = np.array(
            [msg.angular_velocity[0], msg.angular_velocity[1], msg.angular_velocity[2]],
            dtype=float,
        )

        state = np.concatenate([position, quat, velocity, angular_velocity])

        with self.lock:
            self.current_state = state
        # if self.arm_sequence_sent:
        #     self.get_logger().info(f"Updated state = {state}")

    def vehicle_status_callback(self, msg: VehicleStatus) -> None:
        if not hasattr(msg, "arming_state"):
            return

        armed_state = getattr(VehicleStatus, "ARMING_STATE_ARMED", 2)
        is_armed_now = msg.arming_state == armed_state
        with self.lock:
            if is_armed_now == self.is_armed:
                return
            self.is_armed = is_armed_now

        self.get_logger().info(f"Vehicle armed state changed: armed={self.is_armed}")

    def publish_offboard_control_mode(self) -> None:
        if self.current_state is None or self.controller.time_traj is None:
            return

        msg = OffboardControlMode()
        if hasattr(msg, "timestamp"):
            msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        for field in [
            "position",
            "velocity",
            "acceleration",
            "attitude",
            "body_rate",
            "thrust_and_torque",
        ]:
            if hasattr(msg, field):
                setattr(msg, field, False)

        if self.command_output_mode == "vehicle_rates_setpoint":
            if hasattr(msg, "body_rate"):
                msg.body_rate = True
            if hasattr(msg, "direct_actuator"):
                msg.direct_actuator = False
            elif hasattr(msg, "actuator"):
                msg.actuator = False
        else:
            if hasattr(msg, "direct_actuator"):
                msg.direct_actuator = True
            elif hasattr(msg, "actuator"):
                msg.actuator = True

        self.offboard_control_mode_pub.publish(msg)

        if self.arm_sequence_sent:
            return

        if self.offboard_setpoint_counter == 10:
            self.set_offboard_mode()
            self.arm()
            self.arm_sequence_sent = True
            self.get_logger().info("Offboard mode set, vehicle armed")

        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1

    def publish_vehicle_command(
        self, command: int, param1: float = 0.0, param2: float = 0.0
    ) -> None:
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        if hasattr(msg, "timestamp"):
            msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_pub.publish(msg)

    def arm(self) -> None:
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0
        )

    def set_offboard_mode(self) -> None:
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0
        )

    def control_loop(self) -> None:
        with self.lock:
            if self.current_state is None or self.controller.time_traj is None:
                return
            self.publish_offboard_control_mode()
            current_state = self.current_state.copy()
            current_position = current_state[:3].copy()
            final_reference_point = (
                None
                if self.final_reference_point is None
                else self.final_reference_point.copy()
            )
            should_log_final_point = False
            final_distance = None
            if final_reference_point is not None and not self.final_point_reached_logged:
                final_distance = np.linalg.norm(current_position - final_reference_point)
                if final_distance <= self.final_point_reached_radius:
                    self.final_point_reached_logged = True
                    should_log_final_point = True
            cmd = self.controller.run_optimization(initial_state=current_state)

        if should_log_final_point:
            self.get_logger().info(
                "Reached final reference point: "
                f"position={current_position}, target={final_reference_point}, "
                f"distance={final_distance:.3f} m"
            )

        cmd = np.clip(np.array(cmd[:4], dtype=float), 0.0, 1.0)
        if self.command_output_mode == "vehicle_rates_setpoint":
            next_state_xyz = self.controller.integrate_control_step(current_state, cmd)
            body_rates_px4 = Controller.xyz_to_ned_vector(next_state_xyz[10:13])
            normalized_thrust = self.motor_command_to_thrust(cmd)
            with self.lock:
                self.last_body_rates = body_rates_px4.copy()
                self.last_thrust = normalized_thrust
            msg = VehicleRatesSetpoint()
            if hasattr(msg, "timestamp"):
                msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

            msg.roll = float(body_rates_px4[0])
            msg.pitch = float(body_rates_px4[1])
            msg.yaw = float(body_rates_px4[2])
            msg.thrust_body = [0.0, 0.0, -float(normalized_thrust)]
            self.get_logger().info(f"Current state: {self.controller.state_ned_to_xyz(current_state)}")
            self.get_logger().info(f"Predicted next state: {next_state_xyz}")
            self.vehicle_rates_setpoint_pub.publish(msg)

        if  self.command_output_mode == "actuator_motors":
            msg = ActuatorMotors()
            timestamp_us = int(self.get_clock().now().nanoseconds / 1000)
            if hasattr(msg, "timestamp"):
                msg.timestamp = timestamp_us
            if hasattr(msg, "timestamp_sample"):
                msg.timestamp_sample = timestamp_us
            if hasattr(msg, "reversible_flags"):
                msg.reversible_flags = 0

            control = [float("nan")] * 12
            control[0:4] = [float(cmd[0]), float(cmd[1]), float(cmd[2]), float(cmd[3])]
            msg.control = control
            self.motor_pub.publish(msg)

    def publish_vehicle_rates_setpoint(
        self, body_rates_px4: np.ndarray, normalized_thrust: float
    ) -> None:
        if self.vehicle_rates_setpoint_pub is None:
            return

        msg = VehicleRatesSetpoint()
        if hasattr(msg, "timestamp"):
            msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        msg.roll = float(body_rates_px4[0])
        msg.pitch = float(body_rates_px4[1])
        msg.yaw = float(body_rates_px4[2])
        msg.thrust_body = [0.0, 0.0, -float(normalized_thrust)]
        self.vehicle_rates_setpoint_pub.publish(msg)

    def motor_command_to_thrust(self, cmd: np.ndarray) -> float:
        thrust = float(np.sum(cmd) / 4.0)
        return float(np.clip(thrust, self.min_thrust, self.max_thrust))


def main(args=None):
    rclpy.init(args=args)
    node = Px4MpcNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
