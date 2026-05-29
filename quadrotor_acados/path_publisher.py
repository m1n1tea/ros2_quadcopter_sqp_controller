from pathlib import Path as FilePath

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


class PathPublisher(Node):
    def __init__(self) -> None:
        super().__init__("path_publisher")

        self.declare_parameter("path_topic", "/reference_path")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("points_file", "")
        self.declare_parameter("wait_for_subscribers_sec", 0.0)
        self.declare_parameter("keep_alive_sec", 0.0)

        self.path_topic = str(self.get_parameter("path_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.points_file = str(self.get_parameter("points_file").value)
        self.wait_for_subscribers_sec = float(
            self.get_parameter("wait_for_subscribers_sec").value
        )
        self.keep_alive_sec = float(self.get_parameter("keep_alive_sec").value)

        qos_reference_path = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.publisher = self.create_publisher(
            Path, self.path_topic, qos_reference_path
        )
        self.path_msg = self._build_path_message()

        self.get_logger().info(
            f"Prepared trajectory from {self.points_file} for {self.path_topic}: "
            f"{len(self.path_msg.poses)} points"
        )

    def _build_path_message(self) -> Path:
        points = self._load_points_file()
        points = self._normalize_points(points)

        msg = Path()
        msg.header.frame_id = self.frame_id
        self.get_logger().info("Building path:")
        for x, y, z in points:
            pose = PoseStamped()
            pose.header.frame_id = self.frame_id
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = float(z)
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
            self.get_logger().info(f"{(x, y, z)}")
        return msg

    def _load_points_file(self) -> np.ndarray:
        if not self.points_file:
            raise ValueError("points_file parameter is required and must point to a .npy file.")

        path = FilePath(self.points_file).expanduser()
        if path.suffix.lower() != ".npy":
            raise ValueError(f'points_file must be a .npy file, got "{path}".')
        if not path.exists():
            raise FileNotFoundError(f'points_file does not exist: "{path}".')

        return np.load(path, allow_pickle=False)

    def _normalize_points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Trajectory points must be Nx3, got shape {points.shape}.")
        if len(points) < 2:
            raise ValueError("Trajectory must contain at least two points.")
        return points

    def publish_path(self) -> None:
        now = self.get_clock().now().to_msg()
        self.path_msg.header.stamp = now
        for pose in self.path_msg.poses:
            pose.header.stamp = now
        self.publisher.publish(self.path_msg)
        self.get_logger().info("Reference path published.")

    def wait_for_subscribers(self) -> None:
        if self.wait_for_subscribers_sec <= 0.0:
            return

        deadline = (
            self.get_clock().now().nanoseconds
            + int(self.wait_for_subscribers_sec * 1e9)
        )
        while (
            rclpy.ok()
            and self.publisher.get_subscription_count() == 0
            and self.get_clock().now().nanoseconds < deadline
        ):
            rclpy.spin_once(self, timeout_sec=0.1)

        count = self.publisher.get_subscription_count()
        if count == 0:
            self.get_logger().warn(
                "Publishing reference path with no matched subscribers. "
                "The transient-local publisher will stay alive briefly for late joiners."
            )
        else:
            self.get_logger().info(f"Matched {count} reference path subscriber(s).")

    def keep_alive(self) -> None:
        if self.keep_alive_sec <= 0.0:
            return

        deadline = self.get_clock().now().nanoseconds + int(self.keep_alive_sec * 1e9)
        while rclpy.ok() and self.get_clock().now().nanoseconds < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PathPublisher()
    try:
        node.wait_for_subscribers()
        node.publish_path()
        node.keep_alive()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
