#!/usr/bin/env python3

from threading import Lock

import numpy as np
import rospy
from geometry_msgs.msg import TwistStamped
from mavros_msgs.msg import State, Thrust
from mavros_msgs.srv import CommandBool, SetMode
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path as PathMsg

try:
    from .mpc_controller import Controller
    from .quadrotor_model import load_params
except ImportError:
    from mpc_controller import Controller
    from quadrotor_model import load_params


class Ros1Logger:
    def info(self, msg: str) -> None:
        rospy.loginfo(msg)

    def warning(self, msg: str) -> None:
        rospy.logwarn(msg)

    def error(self, msg: str) -> None:
        rospy.logerr(msg)


class Ros1ParameterAdapter:
    def get_parameter(self, name: str):
        path = name.replace(".", "/")
        value = get_private_param(path)
        return type("ParameterValue", (), {"value": value})()


def get_private_param(name: str, default=None):
    private_name = "~" + name
    if rospy.has_param(private_name):
        return rospy.get_param(private_name)

    ros2_style_name = "~ros__parameters/" + name
    if rospy.has_param(ros2_style_name):
        return rospy.get_param(ros2_style_name)

    if default is None:
        return rospy.get_param(private_name)
    return default


class MavrosMpcNode:
    def __init__(self):
        rospy.init_node("mavros_mpc_node")

        self.path_topic = get_private_param("path_topic", "/reference_path")
        self.odom_topic = get_private_param(
            "odometry_topic", "/mavros/local_position/odom"
        )
        self.state_topic = get_private_param("mavros_state_topic", "/mavros/state")
        self.attitude_cmd_vel_topic = get_private_param(
            "mavros_attitude_cmd_vel_topic", "/mavros/setpoint_attitude/cmd_vel"
        )
        self.attitude_thrust_topic = get_private_param(
            "mavros_attitude_thrust_topic", "/mavros/setpoint_attitude/thrust"
        )
        self.arming_service_name = get_private_param(
            "mavros_arming_service", "/mavros/cmd/arming"
        )
        self.set_mode_service_name = get_private_param(
            "mavros_set_mode_service", "/mavros/set_mode"
        )
        self.control_rate_hz = float(get_private_param("control_rate_hz", 200.0))
        self.offboard_control_rate_hz = float(
            get_private_param("offboard_control_rate_hz", 10.0)
        )
        self.preferred_speed = float(get_private_param("preferred_speed", 2.0))
        horizon_sec = float(get_private_param("horizon_sec", 2.0))
        horizon_nodes = int(get_private_param("horizon_nodes", 40))
        self.final_point_reached_radius = float(
            get_private_param("final_point_reached_radius", 0.25)
        )
        self.body_rate_limit = float(get_private_param("body_rate_limit", 4.0))
        self.thrust_scale = float(get_private_param("mavros_thrust_scale", 1.0))
        self.min_thrust = float(get_private_param("mavros_min_thrust", 0.0))
        self.max_thrust = float(get_private_param("mavros_max_thrust", 1.0))
        self.arm_after_setpoints = bool(get_private_param("arm_after_setpoints", True))
        self.set_offboard_after_setpoints = bool(
            get_private_param("set_offboard_after_setpoints", True)
        )

        self.lock = Lock()
        self.current_state = None
        self.final_reference_point = None
        self.final_point_reached_logged = False
        self.mavros_state = State()
        self.offboard_setpoint_counter = 0
        self.arm_sequence_sent = False
        self.last_body_rates = np.zeros(3, dtype=float)
        self.last_thrust = 0.0

        self.quad = load_params(Ros1ParameterAdapter())
        self.controller = Controller(
            quad=self.quad,
            t_horizon=horizon_sec,
            n_nodes=horizon_nodes,
            logger=Ros1Logger(),
            expected_frequency=self.control_rate_hz,
        )

        self.path_sub = rospy.Subscriber(
            self.path_topic, PathMsg, self.path_callback, queue_size=1
        )
        self.odom_sub = rospy.Subscriber(
            self.odom_topic, Odometry, self.odom_callback, queue_size=1
        )
        self.state_sub = rospy.Subscriber(
            self.state_topic, State, self.state_callback, queue_size=1
        )
        self.cmd_vel_pub = rospy.Publisher(
            self.attitude_cmd_vel_topic, TwistStamped, queue_size=1
        )
        self.thrust_pub = rospy.Publisher(
            self.attitude_thrust_topic, Thrust, queue_size=1
        )

        self.arming_srv = rospy.ServiceProxy(self.arming_service_name, CommandBool)
        self.set_mode_srv = rospy.ServiceProxy(self.set_mode_service_name, SetMode)

        self.control_timer = rospy.Timer(
            rospy.Duration(1.0 / self.control_rate_hz), self.control_loop
        )
        self.offboard_timer = rospy.Timer(
            rospy.Duration(1.0 / self.offboard_control_rate_hz),
            self.offboard_control_loop,
        )

        rospy.loginfo(
            "mavros_mpc_node started. "
            f"path={self.path_topic}, odom={self.odom_topic}, state={self.state_topic}, "
            f"cmd_vel={self.attitude_cmd_vel_topic}, thrust={self.attitude_thrust_topic}"
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
            self.final_reference_point = trajectory[-1].copy()
            self.final_point_reached_logged = False
            self.controller.update_trajectory(
                trajectory, preferred_speed=self.preferred_speed
            )

        rospy.loginfo(f"Received trajectory with {len(trajectory)} waypoints")

    def odom_callback(self, msg: Odometry) -> None:
        position = np.array(
            [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ],
            dtype=float,
        )
        quat = np.array(
            [
                msg.pose.pose.orientation.w,
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
            ],
            dtype=float,
        )
        velocity = np.array(
            [
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z,
            ],
            dtype=float,
        )
        angular_velocity = np.array(
            [
                msg.twist.twist.angular.x,
                msg.twist.twist.angular.y,
                msg.twist.twist.angular.z,
            ],
            dtype=float,
        )

        state = np.concatenate([position, quat, velocity, angular_velocity])
        with self.lock:
            self.current_state = state

    def state_callback(self, msg: State) -> None:
        with self.lock:
            self.mavros_state = msg

    def offboard_control_loop(self, _event) -> None:
        with self.lock:
            ready = (
                self.current_state is not None
                and self.controller.time_traj is not None
            )
            state = self.mavros_state

        if not ready:
            return

        self.publish_latest_setpoint()

        if self.arm_sequence_sent:
            return

        if self.offboard_setpoint_counter == 10:
            self.set_offboard_mode(state)
            self.arm(state)
            self.arm_sequence_sent = True
            rospy.loginfo("Offboard mode requested and vehicle arm requested")

        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1

    def publish_latest_setpoint(self) -> None:
        with self.lock:
            body_rates = self.last_body_rates.copy()
            normalized_thrust = self.last_thrust

        stamp = rospy.Time.now()
        cmd_vel = TwistStamped()
        cmd_vel.header.stamp = stamp
        cmd_vel.twist.angular.x = float(body_rates[0])
        cmd_vel.twist.angular.y = float(body_rates[1])
        cmd_vel.twist.angular.z = float(body_rates[2])

        thrust = Thrust()
        thrust.header.stamp = stamp
        thrust.thrust = float(normalized_thrust)

        self.cmd_vel_pub.publish(cmd_vel)
        self.thrust_pub.publish(thrust)

    def set_offboard_mode(self, state: State) -> None:
        if not self.set_offboard_after_setpoints or state.mode == "OFFBOARD":
            return

        try:
            response = self.set_mode_srv(0, "OFFBOARD")
            if not response.mode_sent:
                rospy.logwarn("MAVROS did not accept OFFBOARD mode request")
        except rospy.ServiceException as exc:
            rospy.logerr(f"Failed to request OFFBOARD mode: {exc}")

    def arm(self, state: State) -> None:
        if not self.arm_after_setpoints or state.armed:
            return

        try:
            response = self.arming_srv(True)
            if not response.success:
                rospy.logwarn("MAVROS did not accept arm request")
        except rospy.ServiceException as exc:
            rospy.logerr(f"Failed to request arming: {exc}")

    def control_loop(self, _event) -> None:
        with self.lock:
            if self.current_state is None or self.controller.time_traj is None:
                return

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
            rospy.loginfo(
                "Reached final reference point: "
                f"position={current_position}, target={final_reference_point}, "
                f"distance={final_distance:.3f} m"
            )

        cmd = np.clip(np.array(cmd[:4], dtype=float), 0.0, 1.0)
        body_rates = self.motor_command_to_body_rates(cmd, current_state[10:13])
        normalized_thrust = self.motor_command_to_thrust(cmd)
        with self.lock:
            self.last_body_rates = body_rates.copy()
            self.last_thrust = normalized_thrust

        stamp = rospy.Time.now()
        cmd_vel = TwistStamped()
        cmd_vel.header.stamp = stamp
        cmd_vel.twist.angular.x = float(body_rates[0])
        cmd_vel.twist.angular.y = float(body_rates[1])
        cmd_vel.twist.angular.z = float(body_rates[2])

        thrust = Thrust()
        thrust.header.stamp = stamp
        thrust.thrust = float(normalized_thrust)

        self.cmd_vel_pub.publish(cmd_vel)
        self.thrust_pub.publish(thrust)

    def motor_command_to_body_rates(
        self, cmd: np.ndarray, current_body_rates: np.ndarray
    ) -> np.ndarray:
        f_thrust = cmd * self.quad.max_thrust
        angular_accel = np.array(
            [
                (
                    np.dot(f_thrust, self.quad.x_f)
                    - (self.quad.J[2] - self.quad.J[1])
                    * current_body_rates[1]
                    * current_body_rates[2]
                )
                / self.quad.J[0],
                (
                    np.dot(f_thrust, self.quad.y_f)
                    - (self.quad.J[2] - self.quad.J[0])
                    * current_body_rates[0]
                    * current_body_rates[2]
                )
                / self.quad.J[1],
                (
                    np.dot(f_thrust, self.quad.z_l_tau)
                    - (self.quad.J[1] - self.quad.J[0])
                    * current_body_rates[0]
                    * current_body_rates[1]
                )
                / self.quad.J[2],
            ],
            dtype=float,
        )
        next_rates = current_body_rates + angular_accel / self.control_rate_hz
        return np.clip(next_rates, -self.body_rate_limit, self.body_rate_limit)

    def motor_command_to_thrust(self, cmd: np.ndarray) -> float:
        thrust = self.thrust_scale * float(np.sum(cmd) / 4.0)
        return float(np.clip(thrust, self.min_thrust, self.max_thrust))


def main():
    MavrosMpcNode()
    rospy.spin()


if __name__ == "__main__":
    main()
