import math

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleOdometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


class MovingTargetPublisher(Node):
    def __init__(self) -> None:
        super().__init__("moving_target_publisher")

        self.declare_parameter("target_topic", "/target/odometry")
        self.declare_parameter("odometry_topic", "/fmu/out/vehicle_odometry")
        self.declare_parameter("frame_id", "ned")
        self.declare_parameter("x", 5.0)
        self.declare_parameter("y", 0.0)
        self.declare_parameter("z", -2.0)
        self.declare_parameter("vx", 0.0)
        self.declare_parameter("vy", 0.0)
        self.declare_parameter("vz", 0.0)
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("log_interval_sec", 1.0)
        self.declare_parameter("direction_noise_std", 0.0)
        self.declare_parameter("publish_distance", True)
        self.declare_parameter("distance_noise_std", 0.0)
        self.declare_parameter("random_seed", 0)

        self.target_topic = str(self.get_parameter("target_topic").value)
        self.odometry_topic = str(self.get_parameter("odometry_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.initial_position = np.array(
            [float(self.get_parameter(name).value) for name in ("x", "y", "z")]
        )
        self.velocity = np.array(
            [float(self.get_parameter(name).value) for name in ("vx", "vy", "vz")]
        )
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.log_interval_sec = float(self.get_parameter("log_interval_sec").value)
        self.direction_noise_std = float(
            self.get_parameter("direction_noise_std").value
        )
        self.publish_distance = bool(self.get_parameter("publish_distance").value)
        self.distance_noise_std = float(
            self.get_parameter("distance_noise_std").value
        )
        self.rng = np.random.default_rng(
            int(self.get_parameter("random_seed").value)
        )

        values = (
            *self.initial_position,
            *self.velocity,
            self.publish_rate_hz,
            self.log_interval_sec,
            self.direction_noise_std,
            self.distance_noise_std,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("all target and noise parameters must be finite")
        if self.publish_rate_hz <= 0.0:
            raise ValueError("publish_rate_hz must be positive")
        if self.log_interval_sec < 0.0:
            raise ValueError("log_interval_sec must be non-negative")
        if self.direction_noise_std < 0.0 or self.distance_noise_std < 0.0:
            raise ValueError("noise standard deviations must be non-negative")

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
            Odometry, self.target_topic, qos_target
        )
        self.odometry_sub = self.create_subscription(
            VehicleOdometry,
            self.odometry_topic,
            self.odometry_callback,
            qos_sensor,
        )
        self.vehicle_position = None
        self.start_time = self.get_clock().now()
        self.next_log_time_sec = 0.0
        self.timer = self.create_timer(
            1.0 / self.publish_rate_hz, self.publish_target_observation
        )

        self.get_logger().info(
            f"Publishing target observations on {self.target_topic} from "
            f"odometry={self.odometry_topic}: target={self.initial_position}, "
            f"velocity={self.velocity}, direction_noise_std="
            f"{self.direction_noise_std}, publish_distance="
            f"{self.publish_distance}, distance_noise_std="
            f"{self.distance_noise_std}"
        )

    def odometry_callback(self, msg: VehicleOdometry) -> None:
        position = np.asarray(msg.position[:3], dtype=float)
        if np.all(np.isfinite(position)):
            self.vehicle_position = position

    def publish_target_observation(self) -> None:
        if self.vehicle_position is None:
            return

        now = self.get_clock().now()
        elapsed_sec = max(0.0, (now - self.start_time).nanoseconds * 1e-9)
        target_position = self.initial_position + self.velocity * elapsed_sec
        relative_position = target_position - self.vehicle_position
        true_distance = float(np.linalg.norm(relative_position))
        if true_distance < 1e-9:
            return

        true_direction = relative_position / true_distance
        noisy_direction = true_direction + self.rng.normal(
            0.0, self.direction_noise_std, size=3
        )
        noisy_direction_norm = np.linalg.norm(noisy_direction)
        if noisy_direction_norm < 1e-9:
            noisy_direction = true_direction
        else:
            noisy_direction /= noisy_direction_norm

        observed_distance = float("nan")
        if self.publish_distance:
            observed_distance = max(
                0.0,
                true_distance
                + float(self.rng.normal(0.0, self.distance_noise_std)),
            )

        msg = Odometry()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = self.frame_id
        msg.child_frame_id = "target_direction_ned"
        msg.pose.pose.position.x = float(noisy_direction[0])
        msg.pose.pose.position.y = float(noisy_direction[1])
        msg.pose.pose.position.z = float(noisy_direction[2])
        msg.pose.pose.orientation.w = 1.0
        msg.twist.twist.linear.x = observed_distance
        self.publisher.publish(msg)

        if self.log_interval_sec > 0.0 and elapsed_sec >= self.next_log_time_sec:
            self.get_logger().info(
                f"Target observation: t={elapsed_sec:.2f} s, "
                f"target_position={target_position.tolist()}, "
                f"vehicle_position={self.vehicle_position.tolist()}, "
                f"direction={noisy_direction.tolist()}, "
                f"distance={observed_distance}, true_distance={true_distance:.3f}, "
                f"subscribers={self.publisher.get_subscription_count()}"
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
