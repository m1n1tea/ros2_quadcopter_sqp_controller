import json
import math

import numpy as np
import rclpy
from px4_msgs.msg import VehicleAttitudeSetpoint, VehicleOdometry
from pyquaternion import Quaternion
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


class MovingTargetPublisher(Node):
    """Publish attitude/thrust setpoints derived from a moving target observation."""

    def __init__(self) -> None:
        super().__init__("moving_target_publisher")

        self.declare_parameter("attitude_setpoint_topic", "/target/attitude_setpoint")
        self.declare_parameter("odometry_topic", "/fmu/out/vehicle_odometry")
        self.declare_parameter("x", 5.0)
        self.declare_parameter("y", 0.0)
        self.declare_parameter("z", -2.0)
        self.declare_parameter("vx", 0.0)
        self.declare_parameter("vy", 0.0)
        self.declare_parameter("vz", 0.0)
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("log_interval_sec", 1.0)
        self.declare_parameter("telemetry_log_interval_sec", 0.1)
        self.declare_parameter("collision_radius_m", 0.5)
        self.declare_parameter("direction_noise_std", 0.0)
        self.declare_parameter("publish_distance", True)
        self.declare_parameter("distance_noise_std", 0.0)
        self.declare_parameter("random_seed", 0)
        self.declare_parameter("preferred_common_thrust", -1.0)
        self.declare_parameter("distance_thrust_gain", 0.01)
        self.declare_parameter("min_common_thrust", 0.0)
        self.declare_parameter("max_common_thrust", 1.0)
        self.declare_parameter("max_tilt_deg", 35.0)
        self.declare_parameter("uav_mass", 2.0)
        self.declare_parameter("thrust_constant", 8.54858e-06)
        self.declare_parameter("min_rotor_speed", 150.0)
        self.declare_parameter("max_rotor_speed", 1000.0)

        self.attitude_setpoint_topic = str(
            self.get_parameter("attitude_setpoint_topic").value
        )
        self.odometry_topic = str(self.get_parameter("odometry_topic").value)
        self.initial_position = np.array(
            [float(self.get_parameter(name).value) for name in ("x", "y", "z")]
        )
        self.velocity = np.array(
            [float(self.get_parameter(name).value) for name in ("vx", "vy", "vz")]
        )
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.log_interval_sec = float(self.get_parameter("log_interval_sec").value)
        self.telemetry_log_interval_sec = float(
            self.get_parameter("telemetry_log_interval_sec").value
        )
        self.collision_radius_m = float(
            self.get_parameter("collision_radius_m").value
        )
        self.direction_noise_std = float(
            self.get_parameter("direction_noise_std").value
        )
        self.publish_distance = bool(self.get_parameter("publish_distance").value)
        self.distance_noise_std = float(
            self.get_parameter("distance_noise_std").value
        )
        self.distance_thrust_gain = float(
            self.get_parameter("distance_thrust_gain").value
        )
        self.min_common_thrust = float(
            self.get_parameter("min_common_thrust").value
        )
        self.max_common_thrust = float(
            self.get_parameter("max_common_thrust").value
        )
        self.max_tilt_rad = np.deg2rad(
            float(self.get_parameter("max_tilt_deg").value)
        )
        self.mass = float(self.get_parameter("uav_mass").value)
        thrust_constant = float(self.get_parameter("thrust_constant").value)
        self.min_rotor_speed = float(self.get_parameter("min_rotor_speed").value)
        self.max_rotor_speed = float(self.get_parameter("max_rotor_speed").value)
        self.min_thrust = thrust_constant * self.min_rotor_speed**2
        self.max_thrust = thrust_constant * self.max_rotor_speed**2
        configured_common_thrust = float(
            self.get_parameter("preferred_common_thrust").value
        )
        hover_command = self.thrust_to_motor_command(self.mass * 9.81 / 4.0)
        self.preferred_common_thrust = (
            hover_command
            if configured_common_thrust < 0.0
            else configured_common_thrust
        )
        self.rng = np.random.default_rng(
            int(self.get_parameter("random_seed").value)
        )

        values = (
            *self.initial_position,
            *self.velocity,
            self.publish_rate_hz,
            self.log_interval_sec,
            self.telemetry_log_interval_sec,
            self.collision_radius_m,
            self.direction_noise_std,
            self.distance_noise_std,
            self.preferred_common_thrust,
            self.distance_thrust_gain,
            self.min_common_thrust,
            self.max_common_thrust,
            self.max_tilt_rad,
            self.mass,
            self.min_rotor_speed,
            self.max_rotor_speed,
            thrust_constant,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("all target, thrust, and noise parameters must be finite")
        if self.publish_rate_hz <= 0.0 or self.mass <= 0.0:
            raise ValueError("publish_rate_hz and uav_mass must be positive")
        if self.log_interval_sec < 0.0 or self.telemetry_log_interval_sec < 0.0:
            raise ValueError("log intervals must be non-negative")
        if self.collision_radius_m < 0.0:
            raise ValueError("collision_radius_m must be non-negative")
        if self.direction_noise_std < 0.0 or self.distance_noise_std < 0.0:
            raise ValueError("noise standard deviations must be non-negative")
        if self.max_thrust <= self.min_thrust:
            raise ValueError("max_rotor_speed must be greater than min_rotor_speed")
        if not 0.0 <= self.min_common_thrust <= self.max_common_thrust <= 1.0:
            raise ValueError("common thrust limits must satisfy 0 <= min <= max <= 1")
        if not 0.0 <= self.max_tilt_rad < np.pi / 2.0:
            raise ValueError("max_tilt_deg must be in [0, 90)")

        qos_target = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.publisher = self.create_publisher(
            VehicleAttitudeSetpoint, self.attitude_setpoint_topic, qos_target
        )
        self.odometry_sub = self.create_subscription(
            VehicleOdometry, self.odometry_topic, self.odometry_callback, qos_sensor
        )
        self.vehicle_position = None
        self.vehicle_attitude = None
        self.start_time = self.get_clock().now()
        self.next_log_time_sec = 0.0
        self.next_telemetry_log_time_sec = 0.0
        self.was_in_collision = False
        self.timer = self.create_timer(
            1.0 / self.publish_rate_hz, self.publish_target_setpoint
        )

        self.get_logger().info(
            f"Publishing target attitude setpoints on {self.attitude_setpoint_topic} "
            f"from odometry={self.odometry_topic}, hover_command="
            f"{self.preferred_common_thrust:.3f}, telemetry_interval="
            f"{self.telemetry_log_interval_sec:.3f}s, collision_radius="
            f"{self.collision_radius_m:.3f}m"
        )

    def thrust_to_motor_command(self, thrust: float) -> float:
        command = (thrust - self.min_thrust) / (self.max_thrust - self.min_thrust)
        return float(np.clip(command, 0.0, 1.0))

    def motor_command_to_thrust(self, command: float) -> float:
        return float(
            self.min_thrust + command * (self.max_thrust - self.min_thrust)
        )

    def preferred_common_motor_command(self, distance, direction_ned: np.ndarray) -> float:
        command = self.preferred_common_thrust
        if distance is not None:
            direction = np.asarray(direction_ned, dtype=float)
            command += self.distance_thrust_gain * distance * (
                np.linalg.norm(direction[:2]) - direction[2]
            )
        return float(np.clip(command, self.min_common_thrust, self.max_common_thrust))

    def preferred_attitude(
        self,
        direction_ned: np.ndarray,
        common_motor_command: float,
        current_attitude_ned: np.ndarray,
    ) -> np.ndarray:
        direction = np.asarray(direction_ned, dtype=float)
        horizontal = direction[:2]
        horizontal_norm = np.linalg.norm(horizontal)
        current_rotation = Quaternion(current_attitude_ned).normalised.rotation_matrix
        current_heading = current_rotation[:2, 0]
        if horizontal_norm > 1e-9:
            heading = horizontal / horizontal_norm
        elif np.linalg.norm(current_heading) > 1e-9:
            heading = current_heading / np.linalg.norm(current_heading)
        else:
            heading = np.array([1.0, 0.0])

        total_thrust = 4.0 * self.motor_command_to_thrust(common_motor_command)
        hover_ratio = np.clip(self.mass * 9.81 / total_thrust, 0.0, 1.0)
        available_tilt = np.arccos(hover_ratio)
        direction_scale = horizontal_norm / max(horizontal_norm + abs(direction[2]), 1e-9)
        tilt = min(self.max_tilt_rad, available_tilt) * direction_scale

        body_z = np.array(
            [-heading[0] * np.sin(tilt), -heading[1] * np.sin(tilt), np.cos(tilt)]
        )
        heading_axis = np.array([heading[0], heading[1], 0.0])
        body_y = np.cross(body_z, heading_axis)
        body_y /= np.linalg.norm(body_y)
        body_x = np.cross(body_y, body_z)
        rotation_ned = np.column_stack((body_x, body_y, body_z))
        quaternion = Quaternion(matrix=rotation_ned).normalised
        return np.array([quaternion.w, quaternion.x, quaternion.y, quaternion.z])

    def odometry_callback(self, msg: VehicleOdometry) -> None:
        position = np.asarray(msg.position[:3], dtype=float)
        attitude = np.asarray(msg.q[:4], dtype=float)
        if np.all(np.isfinite(position)):
            self.vehicle_position = position
        if np.all(np.isfinite(attitude)) and np.linalg.norm(attitude) > 1e-9:
            self.vehicle_attitude = attitude / np.linalg.norm(attitude)

    @staticmethod
    def quaternion_to_euler_deg(quaternion: np.ndarray) -> np.ndarray:
        """Return roll, pitch, yaw in degrees for a w-x-y-z quaternion."""
        w, x, y, z = np.asarray(quaternion, dtype=float)
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = math.asin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return np.rad2deg([roll, pitch, yaw])

    def log_tracking_telemetry(
        self,
        elapsed_sec: float,
        target_position: np.ndarray,
        distance: float,
        desired_attitude: np.ndarray,
        thrust_fraction: float,
    ) -> None:
        payload = {
            "schema_version": 1,
            "time_sec": round(float(elapsed_sec), 6),
            "vehicle_position_ned_m": self.vehicle_position.tolist(),
            "target_position_ned_m": target_position.tolist(),
            "distance_m": round(float(distance), 6),
            "desired_attitude_quat_wxyz": desired_attitude.tolist(),
            "observed_attitude_quat_wxyz": self.vehicle_attitude.tolist(),
            "desired_euler_deg": self.quaternion_to_euler_deg(desired_attitude).tolist(),
            "observed_euler_deg": self.quaternion_to_euler_deg(
                self.vehicle_attitude
            ).tolist(),
            "thrust_fraction": round(float(thrust_fraction), 6),
        }
        self.get_logger().info(
            "TRACKING_TELEMETRY " + json.dumps(payload, separators=(",", ":"))
        )

    def log_collision(
        self, elapsed_sec: float, target_position: np.ndarray, distance: float
    ) -> None:
        payload = {
            "schema_version": 1,
            "time_sec": round(float(elapsed_sec), 6),
            "vehicle_position_ned_m": self.vehicle_position.tolist(),
            "target_position_ned_m": target_position.tolist(),
            "distance_m": round(float(distance), 6),
            "collision_radius_m": self.collision_radius_m,
        }
        self.get_logger().warning(
            "TRACKING_COLLISION " + json.dumps(payload, separators=(",", ":"))
        )

    def publish_target_setpoint(self) -> None:
        if self.vehicle_position is None or self.vehicle_attitude is None:
            return

        now = self.get_clock().now()
        elapsed_sec = max(0.0, (now - self.start_time).nanoseconds * 1e-9)
        target_position = self.initial_position + self.velocity * elapsed_sec
        relative_position = target_position - self.vehicle_position
        true_distance = float(np.linalg.norm(relative_position))
        if true_distance < 1e-9:
            if not self.was_in_collision:
                self.log_collision(elapsed_sec, target_position, true_distance)
            self.was_in_collision = True
            return

        direction = relative_position / true_distance
        direction += self.rng.normal(0.0, self.direction_noise_std, size=3)
        direction /= np.linalg.norm(direction)
        distance = None
        if self.publish_distance:
            distance = max(
                0.0, true_distance + float(self.rng.normal(0.0, self.distance_noise_std))
            )
        common_command = self.preferred_common_motor_command(distance, direction)
        attitude = self.preferred_attitude(
            direction, common_command, self.vehicle_attitude
        )
        thrust_fraction = self.motor_command_to_thrust(common_command) / self.max_thrust

        msg = VehicleAttitudeSetpoint()
        msg.timestamp = int(now.nanoseconds / 1000)
        msg.q_d = [float(value) for value in attitude]
        msg.thrust_body = [0.0, 0.0, -float(thrust_fraction)]
        self.publisher.publish(msg)

        in_collision = true_distance <= self.collision_radius_m
        if in_collision and not self.was_in_collision:
            self.log_collision(elapsed_sec, target_position, true_distance)
        self.was_in_collision = in_collision

        if (
            self.telemetry_log_interval_sec > 0.0
            and elapsed_sec >= self.next_telemetry_log_time_sec
        ):
            self.log_tracking_telemetry(
                elapsed_sec,
                target_position,
                true_distance,
                attitude,
                thrust_fraction,
            )
            self.next_telemetry_log_time_sec = (
                elapsed_sec + self.telemetry_log_interval_sec
            )

        if self.log_interval_sec > 0.0 and elapsed_sec >= self.next_log_time_sec:
            self.get_logger().info(
                f"Target setpoint: t={elapsed_sec:.2f} s, direction={direction.tolist()}, "
                f"distance={distance}, attitude={attitude.tolist()}, "
                f"thrust_fraction={thrust_fraction:.3f}"
            )
            self.next_log_time_sec = elapsed_sec + self.log_interval_sec


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MovingTargetPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
