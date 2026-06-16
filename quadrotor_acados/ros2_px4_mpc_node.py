from threading import Lock

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
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

        self.target_topic = self.get_parameter("target_topic").value
        self.target_radius = float(self._get_param("target_radius", 1.0))
        if self.target_radius < 0.0:
            raise ValueError("target_radius must be non-negative")
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
        self.target_direction = None
        self.target_distance = None
        self.target_reached_logged = False
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
            hover_thrust = self.quad.mass * 9.81 / (
                4.0 * self.quad.max_thrust
            )
            configured_common_thrust = float(
                self._get_param("preferred_common_thrust", -1.0)
            )
            self.preferred_common_thrust = (
                hover_thrust
                if configured_common_thrust < 0.0
                else configured_common_thrust
            )
            self.distance_thrust_gain = float(
                self._get_param("distance_thrust_gain", 0.0)
            )
            self.min_common_thrust = float(
                self._get_param("min_common_thrust", 0.0)
            )
            self.max_common_thrust = float(
                self._get_param("max_common_thrust", 1.0)
            )
            self.max_tilt_rad = np.deg2rad(
                float(self._get_param("max_tilt_deg", 35.0))
            )
            thrust_parameters = (
                self.preferred_common_thrust,
                self.distance_thrust_gain,
                self.min_common_thrust,
                self.max_common_thrust,
                self.max_tilt_rad,
            )
            if not np.all(np.isfinite(thrust_parameters)):
                raise ValueError("thrust and tilt parameters must be finite")
            if not (
                0.0 <= self.min_common_thrust <= self.max_common_thrust <= 1.0
            ):
                raise ValueError(
                    "common thrust limits must satisfy "
                    "0 <= min_common_thrust <= max_common_thrust <= 1"
                )
            if self.max_tilt_rad < 0.0 or self.max_tilt_rad >= np.pi / 2.0:
                raise ValueError("max_tilt_deg must be in [0, 90)")
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
            Odometry, self.target_topic, self.target_callback, qos_target
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
            f"target={self.target_topic}, target_radius={self.target_radius}, "
            f"odom={self.odom_topic}, vehicle_status={self.vehicle_status_topic}, "
            f"output_mode={self.command_output_mode}, actuator={self.actuator_topic}, "
            f"use_normalized_rotor_speed={self.use_normalized_rotor_speed}, "
            f"vehicle_rates_setpoint={self.vehicle_rates_setpoint_topic}, "
            f"offboard_control_mode={self.offboard_control_mode_topic}, "
            f"vehicle_command={self.vehicle_command_topic}, "
            f"common_thrust={self.preferred_common_thrust:.3f}, "
            f"distance_thrust_gain={self.distance_thrust_gain:.3f}, "
            f"max_tilt_deg={np.rad2deg(self.max_tilt_rad):.1f}"
        )

    def _get_param(self, name: str, default):
        if not self.has_parameter(name):
            self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def target_callback(self, msg: Odometry) -> None:
        direction = np.array(
            [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ],
            dtype=float,
        )
        direction_norm = np.linalg.norm(direction)
        if not np.all(np.isfinite(direction)) or direction_norm < 1e-9:
            self.get_logger().warning("Ignoring invalid target direction observation")
            return
        direction /= direction_norm

        distance = float(msg.twist.twist.linear.x)
        if not np.isfinite(distance) or distance < 0.0:
            distance = None

        with self.lock:
            self.target_direction = direction
            self.target_distance = distance

        self.get_logger().info(
            f"Received target observation: direction={direction}, "
            f"distance={distance}"
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
        if self.current_state is None or self.target_direction is None:
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

    def preferred_common_motor_thrust(
        self, distance, direction_ned: np.ndarray
    ) -> float:
        thrust = self.preferred_common_thrust
        if distance is not None:
            direction = np.asarray(direction_ned, dtype=float)
            thrust_direction_scale = np.linalg.norm(direction[:2]) - direction[2]
            thrust += (
                self.distance_thrust_gain
                * distance
                * thrust_direction_scale
            )
        return float(
            np.clip(thrust, self.min_common_thrust, self.max_common_thrust)
        )

    def preferred_attitude(
        self,
        direction_ned: np.ndarray,
        common_motor_thrust: float,
        current_attitude_ned: np.ndarray,
    ) -> np.ndarray:
        direction = np.asarray(direction_ned, dtype=float)
        horizontal = direction[:2]
        horizontal_norm = np.linalg.norm(horizontal)

        current_rotation = Controller.quat_to_rotmat(current_attitude_ned)
        current_heading = current_rotation[:2, 0]
        current_heading_norm = np.linalg.norm(current_heading)
        if horizontal_norm > 1e-9:
            heading = horizontal / horizontal_norm
        elif current_heading_norm > 1e-9:
            heading = current_heading / current_heading_norm
        else:
            heading = np.array([1.0, 0.0])

        total_thrust = 4.0 * common_motor_thrust * self.quad.max_thrust
        if total_thrust <= 1e-9:
            tilt = 0.0
        else:
            hover_ratio = np.clip(
                self.quad.mass * 9.81 / total_thrust, 0.0, 1.0
            )
            available_tilt = np.arccos(hover_ratio)
            direction_scale = horizontal_norm / max(
                horizontal_norm + abs(direction[2]), 1e-9
            )
            tilt = min(self.max_tilt_rad, available_tilt) * direction_scale

        # PX4 NED/FRD: body +Z points down, while rotor thrust points along -Z.
        body_z = np.array(
            [
                -heading[0] * np.sin(tilt),
                -heading[1] * np.sin(tilt),
                np.cos(tilt),
            ],
            dtype=float,
        )
        heading_axis = np.array([heading[0], heading[1], 0.0], dtype=float)
        body_y = np.cross(body_z, heading_axis)
        body_y_norm = np.linalg.norm(body_y)
        if body_y_norm < 1e-9:
            body_y = np.array([-heading[1], heading[0], 0.0], dtype=float)
        else:
            body_y /= body_y_norm
        body_x = np.cross(body_y, body_z)
        body_x /= np.linalg.norm(body_x)

        rotation_ned = np.column_stack((body_x, body_y, body_z))
        return Controller.rotmat_to_quat(rotation_ned)

    def control_loop(self) -> None:
        with self.lock:
            if self.current_state is None or self.target_direction is None:
                return
            self.publish_offboard_control_mode()
            if self.state_update_seq == self.last_optimized_state_update_seq:
                return
            current_state = self.current_state.copy()
            self.last_optimized_state_update_seq = self.state_update_seq
            target_direction = self.target_direction.copy()
            target_distance = self.target_distance
            common_thrust = self.preferred_common_motor_thrust(
                target_distance, target_direction
            )
            target_attitude = self.preferred_attitude(
                target_direction, common_thrust, current_state[:4]
            )
            self.controller.update_target(target_attitude, common_thrust)
            should_log_target_reached = (
                target_distance is not None
                and target_distance <= self.target_radius
                and not self.target_reached_logged
            )
            if should_log_target_reached:
                self.target_reached_logged = True
            elif target_distance is None or target_distance > self.target_radius:
                self.target_reached_logged = False
            self.get_logger().info(
                "Angular control target: "
                f"direction_ned={target_direction}, distance={target_distance}, "
                f"attitude_ned={target_attitude}, "
                f"common_thrust={common_thrust:.3f}"
            )
            cmd, next_x = self.controller.run_optimization(initial_state=current_state)

        if should_log_target_reached:
            self.get_logger().info(
                "Reached observed target range: "
                f"distance={target_distance:.3f} m, radius={self.target_radius:.3f} m"
            )

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
