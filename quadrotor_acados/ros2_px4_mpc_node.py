from threading import Lock

import numpy as np
import rclpy
from px4_msgs.msg import (
    ActuatorMotors,
    OffboardControlMode,
    VehicleAttitudeSetpoint,
    VehicleCommand,
    VehicleOdometry,
    VehicleStatus,
)

try:
    from px4_msgs.msg import VehicleRatesSetpoint
except ImportError:
    VehicleRatesSetpoint = None
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from .mpc_controller import Controller
from .quadrotor_model import load_params


class Px4MpcNode(Node):
    def __init__(self):
        super().__init__(
            "px4_mpc_node", automatically_declare_parameters_from_overrides=True
        )

        self.target_attitude_setpoint_topic = self._get_param(
            "target_attitude_setpoint_topic", "/target/attitude_setpoint"
        )
        self.odom_topic = self.get_parameter("odometry_topic").value
        self.vehicle_status_topic = self.get_parameter("vehicle_status_topic").value
        self.actuator_topic = self.get_parameter("actuator_topic").value
        self.vehicle_rates_setpoint_topic = self._get_param(
            "vehicle_rates_setpoint_topic", "/fmu/in/vehicle_rates_setpoint"
        )
        self.command_output_mode = str(
            self._get_param("command_output_mode", "actuator_motors")
        ).lower()
        self.use_normalized_rotor_speed = bool(
            self._get_param("use_normalized_rotor_speed", False)
        )
        self.enabled = bool(self._get_param("enabled", True))
        self.offboard_control_mode_topic = self.get_parameter(
            "offboard_control_mode_topic"
        ).value
        self.vehicle_command_topic = self.get_parameter("vehicle_command_topic").value
        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.offboard_control_rate_hz = float(
            self.get_parameter("offboard_control_rate_hz").value
        )
        horizon_sec = float(self.get_parameter("horizon_sec").value)
        horizon_nodes = int(self.get_parameter("horizon_nodes").value)
        max_body_rate = np.asarray(
            self.get_parameter("max_body_rate").value, dtype=float
        )
        q_cost = np.asarray(self.get_parameter("q_cost").value, dtype=float)
        r_cost = np.asarray(self.get_parameter("r_cost").value, dtype=float)
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
        self.state_update_seq = 0
        self.last_optimized_state_update_seq = 0
        self.target_attitude = None
        self.target_common_thrust = None
        self.is_armed = False
        self.offboard_setpoint_counter = 0
        self.arm_sequence_sent = False
        self.last_state_log_time_sec = -1.0
        self.last_body_rates = np.zeros(3, dtype=float)
        self.last_thrust = 0.0

        try:
            self.quad = load_params(self)
            if self.quad.max_thrust <= self.quad.min_thrust:
                raise ValueError("max_rotor_speed must be greater than min_rotor_speed")
            self.controller = Controller(
                quad=self.quad,
                t_horizon=horizon_sec,
                n_nodes=horizon_nodes,
                max_body_rate=max_body_rate,
                q_cost=q_cost,
                r_cost=r_cost,
                logger=self.get_logger(),
                expected_frequency=self.control_rate_hz,
                enable_integrator=True,
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

        qos_target = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.target_sub = self.create_subscription(
            VehicleAttitudeSetpoint,
            self.target_attitude_setpoint_topic,
            self.target_callback,
            qos_target,
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
        self.add_on_set_parameters_callback(self.parameter_callback)

        self.get_logger().info(
            "px4_mpc_node started. "
            f"target_attitude_setpoint={self.target_attitude_setpoint_topic}, "
            f"odom={self.odom_topic}, vehicle_status={self.vehicle_status_topic}, "
            f"output_mode={self.command_output_mode}, actuator={self.actuator_topic}, "
            f"use_normalized_rotor_speed={self.use_normalized_rotor_speed}, "
            f"enabled={self.enabled}, "
            f"vehicle_rates_setpoint={self.vehicle_rates_setpoint_topic}, "
            f"offboard_control_mode={self.offboard_control_mode_topic}, "
            f"vehicle_command={self.vehicle_command_topic}"
        )

    def _get_param(self, name: str, default):
        if not self.has_parameter(name):
            self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def parameter_callback(self, parameters):
        for parameter in parameters:
            if parameter.name != "enabled":
                continue
            if not isinstance(parameter.value, bool):
                return SetParametersResult(
                    successful=False, reason="enabled must be a boolean"
                )
            with self.lock:
                self.enabled = parameter.value
                self.offboard_setpoint_counter = 0
                self.arm_sequence_sent = False
            state = "enabled" if parameter.value else "disabled"
            self.get_logger().info(f"MPC output {state}")
        return SetParametersResult(successful=True)

    def target_callback(self, msg: VehicleAttitudeSetpoint) -> None:
        attitude = np.asarray(msg.q_d, dtype=float)
        attitude_norm = np.linalg.norm(attitude)
        thrust_fraction = -float(msg.thrust_body[2])
        if (
            attitude.shape != (4,)
            or not np.all(np.isfinite(attitude))
            or attitude_norm < 1e-9
            or not np.isfinite(thrust_fraction)
        ):
            self.get_logger().warning("Ignoring invalid target attitude setpoint")
            return
        attitude /= attitude_norm
        thrust_fraction = float(np.clip(thrust_fraction, 0.0, 1.0))
        common_thrust = self.thrust_fraction_to_motor_command(thrust_fraction)

        with self.lock:
            self.target_attitude = attitude
            self.target_common_thrust = common_thrust

        self.get_logger().info(
            f"Received target attitude setpoint: attitude={attitude}, "
            f"thrust_fraction={thrust_fraction:.3f}"
        )

    def odom_callback(self, msg: VehicleOdometry) -> None:
        quat = np.array([msg.q[0], msg.q[1], msg.q[2], msg.q[3]], dtype=float)
        angular_velocity = np.array(
            [msg.angular_velocity[0], msg.angular_velocity[1], msg.angular_velocity[2]],
            dtype=float,
        )

        state = np.concatenate([quat, angular_velocity])
        if self.arm_sequence_sent:
            self.get_logger().info(f"Updated angular state = {state}")

        with self.lock:
            self.current_state = state
            self.state_update_seq += 1

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
        if not self.enabled or self.current_state is None or self.target_attitude is None:
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

    def thrust_fraction_to_motor_command(self, thrust_fraction: float) -> float:
        thrust = thrust_fraction * self.quad.max_thrust
        command = (thrust - self.quad.min_thrust) / (
            self.quad.max_thrust - self.quad.min_thrust
        )
        return float(np.clip(command, 0.0, 1.0))

    def control_loop(self) -> None:
        with self.lock:
            if not self.enabled or self.current_state is None or self.target_attitude is None:
                return
            self.publish_offboard_control_mode()
            if self.state_update_seq == self.last_optimized_state_update_seq:
                return
            current_state = self.current_state.copy()
            self.last_optimized_state_update_seq = self.state_update_seq
            target_attitude = self.target_attitude.copy()
            common_thrust = self.target_common_thrust
            self.controller.update_target(target_attitude, common_thrust)
            self.get_logger().info(
                "Angular control target: "
                f"attitude_ned={target_attitude}, "
                f"common_thrust={common_thrust:.3f}"
            )
            cmd, next_x = self.controller.run_optimization(initial_state=current_state)

        cmd = np.clip(np.array(cmd[:4], dtype=float), 0.0, 1.0)
        next_state_xyz = self.controller.integrate_control_step(current_state, cmd)
        self.get_logger().info(f"Got control = {list(cmd)}")
        self.get_logger().info(f"Predicted next angular state: {next_state_xyz}")
        if self.command_output_mode == "vehicle_rates_setpoint":
            body_rates_px4 = Controller.xyz_to_ned_vector(next_x[4:7])
            normalized_thrust = self.motor_commands_to_output_value(cmd)
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
            self.get_logger().info(
                "Send control: "
                f"rates={list(body_rates_px4)}, thrust={float(normalized_thrust)}"
            )
            self.vehicle_rates_setpoint_pub.publish(msg)

        if  self.command_output_mode == "actuator_motors":
            actuator_cmd = self.motor_command_to_actuator_control(cmd)
            self.get_logger().info(f"Send control = {list(actuator_cmd)}")
            msg = ActuatorMotors()
            timestamp_us = int(self.get_clock().now().nanoseconds / 1000)
            if hasattr(msg, "timestamp"):
                msg.timestamp = timestamp_us
            if hasattr(msg, "timestamp_sample"):
                msg.timestamp_sample = timestamp_us
            if hasattr(msg, "reversible_flags"):
                msg.reversible_flags = 0

            control = [float("nan")] * 12
            control[0:4] = [
                float(actuator_cmd[0]),
                float(actuator_cmd[1]),
                float(actuator_cmd[2]),
                float(actuator_cmd[3]),
            ]
            msg.control = control
            self.motor_pub.publish(msg)

    def motor_commands_to_output_value(self, cmd: np.ndarray) -> float:
        return float(np.mean(self.motor_command_to_output_value(cmd)))

    def motor_command_to_actuator_control(self, cmd: np.ndarray) -> np.ndarray:
        return self.motor_command_to_output_value(cmd)

    def motor_command_to_output_value(self, cmd: np.ndarray) -> np.ndarray:
        cmd = np.clip(np.asarray(cmd, dtype=float), 0.0, 1.0)
        if not self.use_normalized_rotor_speed:
            return cmd

        rotor_speed = np.sqrt(
            self.quad.min_rotor_speed**2
            + cmd
            * (self.quad.max_rotor_speed**2 - self.quad.min_rotor_speed**2)
        )
        return (rotor_speed - self.quad.min_rotor_speed) / (
            self.quad.max_rotor_speed - self.quad.min_rotor_speed
        )

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
