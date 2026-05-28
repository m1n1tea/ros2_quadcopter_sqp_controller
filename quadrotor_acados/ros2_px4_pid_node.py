from threading import Lock

import numpy as np
import rclpy
from nav_msgs.msg import Path as PathMsg
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleOdometry,
    VehicleStatus,
)
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


class Px4PidNode(Node):
    def __init__(self):
        super().__init__("px4_pid_node")

        self.path_topic = "/reference_path"
        self.odom_topic = "/fmu/out/vehicle_odometry"
        self.vehicle_status_topic = "/fmu/out/vehicle_status"
        self.trajectory_setpoint_topic = "/fmu/in/trajectory_setpoint"
        self.offboard_control_mode_topic = "/fmu/in/offboard_control_mode"
        self.vehicle_command_topic = "/fmu/in/vehicle_command"
        self.control_rate_hz = 50.0
        self.offboard_control_rate_hz = 3.0
        self.waypoint_reached_radius = 0.6
        self.final_point_reached_radius = 0.1

        self.lock = Lock()
        self.current_position = None
        self.reference_trajectory = None
        self.current_waypoint_idx = 0
        self.final_point_reached_logged = False
        self.is_armed = False
        self.offboard_setpoint_counter = 0
        self.arm_sequence_sent = False

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        qos_reference_path = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.path_sub = self.create_subscription(
            PathMsg, self.path_topic, self.path_callback, qos_reference_path
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
        self.trajectory_setpoint_pub = self.create_publisher(
            TrajectorySetpoint, self.trajectory_setpoint_topic, qos_sensor
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
            "px4_pid_node started. "
            f"path={self.path_topic}, odom={self.odom_topic}, vehicle_status={self.vehicle_status_topic}, "
            f"trajectory_setpoint={self.trajectory_setpoint_topic}, offboard_control_mode={self.offboard_control_mode_topic}, "
            f"vehicle_command={self.vehicle_command_topic}"
        )

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
            self.reference_trajectory = trajectory
            self.current_waypoint_idx = 0
            self.final_point_reached_logged = False

        self.get_logger().info(f"Received trajectory with {len(trajectory)} waypoints")

    def odom_callback(self, msg: VehicleOdometry) -> None:
        position = np.array(
            [msg.position[0], msg.position[1], msg.position[2]], dtype=float
        )
        with self.lock:
            self.current_position = position

        if self.arm_sequence_sent:
            self.get_logger().info(f"Updated state = {position}")

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
        if self.current_position is None or self.reference_trajectory is None:
            return
        msg = OffboardControlMode()
        if hasattr(msg, "timestamp"):
            msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        # Enable PX4 position control in offboard mode.
        for field in [
            "position",
            "velocity",
            "acceleration",
            "attitude",
            "body_rate",
            "thrust_and_torque",
            "direct_actuator",
        ]:
            if hasattr(msg, field):
                setattr(msg, field, False)
        if hasattr(msg, "position"):
            msg.position = True

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
            if self.current_position is None or self.reference_trajectory is None:
                return
            self.publish_offboard_control_mode()
            last_idx = len(self.reference_trajectory) - 1
            while self.current_waypoint_idx < last_idx:
                target = self.reference_trajectory[self.current_waypoint_idx]
                distance = np.linalg.norm(self.current_position - target)
                if distance > self.waypoint_reached_radius:
                    break
                self.current_waypoint_idx += 1

            target = self.reference_trajectory[self.current_waypoint_idx].copy()
            final_distance = np.linalg.norm(
                self.current_position - self.reference_trajectory[last_idx]
            )
            if (
                not self.final_point_reached_logged
                and self.current_waypoint_idx == last_idx
                and final_distance <= self.final_point_reached_radius
            ):
                self.final_point_reached_logged = True
                self.get_logger().info(
                    "Reached final reference point: "
                    f"position={self.current_position}, target={self.reference_trajectory[last_idx]}, "
                    f"distance={final_distance:.3f} m"
                )
            self.get_logger().info(f"Current target = {target}")

            msg = TrajectorySetpoint()
            timestamp_us = int(self.get_clock().now().nanoseconds / 1000)
            if hasattr(msg, "timestamp"):
                msg.timestamp = timestamp_us

            msg.position = [float(target[0]), float(target[1]), float(target[2])]
            if hasattr(msg, "velocity"):
                msg.velocity = [float("nan"), float("nan"), float("nan")]
            if hasattr(msg, "acceleration"):
                msg.acceleration = [float("nan"), float("nan"), float("nan")]
            if hasattr(msg, "yaw"):
                msg.yaw = float("nan")
            if hasattr(msg, "yawspeed"):
                msg.yawspeed = float("nan")

            self.trajectory_setpoint_pub.publish(msg)
        if self.arm_sequence_sent:
            self.get_logger().info(f"Current target = {target}")
            self.get_logger().info(f"Current idx = {self.current_waypoint_idx}")


def main(args=None):
    rclpy.init(args=args)
    node = Px4PidNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
