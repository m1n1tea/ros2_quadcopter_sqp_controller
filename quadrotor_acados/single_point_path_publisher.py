import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


class SinglePointPathPublisher(Node):
    def __init__(self) -> None:
        super().__init__("single_point_path_publisher")

        self.declare_parameter("path_topic", "/reference_path")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("x", 0.0)
        self.declare_parameter("y", 0.0)
        self.declare_parameter("z", -0.04)
        self.declare_parameter("wait_for_subscribers_sec", 0.0)
        self.declare_parameter("keep_alive_sec", 3.0)

        self.path_topic = str(self.get_parameter("path_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.point = (
            float(self.get_parameter("x").value),
            float(self.get_parameter("y").value),
            float(self.get_parameter("z").value),
        )
        self.wait_for_subscribers_sec = float(
            self.get_parameter("wait_for_subscribers_sec").value
        )
        self.keep_alive_sec = float(self.get_parameter("keep_alive_sec").value)

        qos_reference_path = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.publisher = self.create_publisher(
            Path, self.path_topic, qos_reference_path
        )
        self.path_msg = self._build_path_message()

        self.get_logger().info(
            f"Prepared single-point reference for {self.path_topic}: {self.point}"
        )

    def _build_path_message(self) -> Path:
        msg = Path()
        msg.header.frame_id = self.frame_id

        pose = PoseStamped()
        pose.header.frame_id = self.frame_id
        pose.pose.position.x = self.point[0]
        pose.pose.position.y = self.point[1]
        pose.pose.position.z = self.point[2]
        pose.pose.orientation.w = 1.0
        msg.poses.append(pose)

        return msg

    def publish_path(self) -> None:
        now = self.get_clock().now().to_msg()
        self.path_msg.header.stamp = now
        self.path_msg.poses[0].header.stamp = now
        self.publisher.publish(self.path_msg)
        self.get_logger().info(f"Single-point reference published: {self.point}")

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
                "Publishing single-point reference with no matched subscribers. "
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
    node = SinglePointPathPublisher()
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
