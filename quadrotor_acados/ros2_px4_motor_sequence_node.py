import math

import rclpy
from px4_msgs.msg import (
    ActuatorMotors,
    OffboardControlMode,
    VehicleCommand,
    VehicleStatus,
)
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


class Px4MotorSequenceNode(Node):
    def __init__(self):
        super().__init__(
            "px4_motor_sequence_node",
            automatically_declare_parameters_from_overrides=True,
        )

        self.vehicle_status_topic = self._get_param(
            "vehicle_status_topic", "/fmu/out/vehicle_status"
        )
        self.actuator_topic = self._get_param(
            "actuator_topic", "/fmu/in/actuator_motors"
        )
        self.offboard_control_mode_topic = self._get_param(
            "offboard_control_mode_topic", "/fmu/in/offboard_control_mode"
        )
        self.vehicle_command_topic = self._get_param(
            "vehicle_command_topic", "/fmu/in/vehicle_command"
        )

        self.control_rate_hz = float(self._get_param("control_rate_hz", 50.0))
        self.offboard_control_rate_hz = float(
            self._get_param("offboard_control_rate_hz", 10.0)
        )
        self.motor_count = int(self._get_param("motor_count", 4))
        self.base_value = float(self._get_param("base_value", 0.0))
        self.pulse_value = float(self._get_param("pulse_value", 0.1))
        self.stop_value = float(self._get_param("stop_value", 0.0))
        self.pulse_duration_sec = float(self._get_param("pulse_duration_sec", 1.0))
        self.pause_duration_sec = float(self._get_param("pause_duration_sec", 0.5))
        self.start_delay_sec = float(self._get_param("start_delay_sec", 2.0))
        self.repeat_cycles = int(self._get_param("repeat_cycles", 1))
        self.auto_arm = bool(self._get_param("auto_arm", True))
        self.disarm_after_sequence = bool(
            self._get_param("disarm_after_sequence", True)
        )

        if self.motor_count < 1 or self.motor_count > 12:
            raise ValueError("motor_count must be in range [1, 12]")
        if self.control_rate_hz <= 0.0 or self.offboard_control_rate_hz <= 0.0:
            raise ValueError("control rates must be positive")
        if self.pulse_duration_sec <= 0.0 or self.pause_duration_sec < 0.0:
            raise ValueError("pulse_duration_sec must be positive, pause non-negative")

        self.base_value = self._clip_control(self.base_value)
        self.pulse_value = self._clip_control(self.pulse_value)
        self.stop_value = self._clip_control(self.stop_value)

        self.is_armed = False
        self.offboard_setpoint_counter = 0
        self.arm_sequence_sent = False
        self.sequence_complete = False
        self.sequence_start_time_sec = self._now_sec() + self.start_delay_sec
        self.last_phase = None

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
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

        auto_arm_text = "enabled" if self.auto_arm else "disabled"
        self.get_logger().info(
            "px4_motor_sequence_node started. "
            f"actuator={self.actuator_topic}, offboard_control_mode={self.offboard_control_mode_topic}, "
            f"vehicle_command={self.vehicle_command_topic}, motor_count={self.motor_count}, "
            f"base={self.base_value:.3f}, pulse={self.pulse_value:.3f}, "
            f"pulse_duration={self.pulse_duration_sec:.3f}s, pause={self.pause_duration_sec:.3f}s, "
            f"repeat_cycles={self.repeat_cycles}, auto_arm={auto_arm_text}"
        )

    def _get_param(self, name: str, default):
        if not self.has_parameter(name):
            self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def _clip_control(value: float) -> float:
        return min(max(float(value), -1.0), 1.0)

    def vehicle_status_callback(self, msg: VehicleStatus) -> None:
        if not hasattr(msg, "arming_state"):
            return

        armed_state = getattr(VehicleStatus, "ARMING_STATE_ARMED", 2)
        is_armed_now = msg.arming_state == armed_state
        if is_armed_now == self.is_armed:
            return

        self.is_armed = is_armed_now
        self.get_logger().info(f"Vehicle armed state changed: armed={self.is_armed}")

    def publish_offboard_control_mode(self) -> None:
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
        if hasattr(msg, "direct_actuator"):
            msg.direct_actuator = True
        elif hasattr(msg, "actuator"):
            msg.actuator = True

        self.offboard_control_mode_pub.publish(msg)

        if not self.auto_arm or self.arm_sequence_sent:
            return

        if self.offboard_setpoint_counter == 10:
            self.set_offboard_mode()
            self.arm()
            self.arm_sequence_sent = True
            self.get_logger().info("Offboard mode set, vehicle arm requested")

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

    def disarm(self) -> None:
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0
        )

    def set_offboard_mode(self) -> None:
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0
        )

    def control_loop(self) -> None:
        now_sec = self._now_sec()
        motor_index = None
        value = self.base_value
        phase = "waiting"

        if now_sec >= self.sequence_start_time_sec and not self.sequence_complete:
            elapsed_sec = now_sec - self.sequence_start_time_sec
            step_duration_sec = self.pulse_duration_sec + self.pause_duration_sec
            step_index = int(math.floor(elapsed_sec / step_duration_sec))
            phase_time_sec = elapsed_sec - step_index * step_duration_sec
            cycle_index = step_index // self.motor_count

            if self.repeat_cycles > 0 and cycle_index >= self.repeat_cycles:
                self.sequence_complete = True
                phase = "complete"
                value = self.stop_value
                if self.disarm_after_sequence:
                    self.disarm()
                    self.get_logger().info("Sequence complete, vehicle disarm requested")
                else:
                    self.get_logger().info("Sequence complete")
            elif phase_time_sec < self.pulse_duration_sec:
                motor_index = step_index % self.motor_count
                value = self.pulse_value
                phase = f"motor_{motor_index}"
            else:
                phase = "pause"

        if self.sequence_complete:
            phase = "complete"
            value = self.stop_value

        if phase != self.last_phase:
            self.last_phase = phase
            if motor_index is None:
                self.get_logger().info(f"Motor sequence phase: {phase}")
            else:
                self.get_logger().info(
                    f"Motor sequence phase: motor={motor_index}, value={value:.3f}"
                )

        msg = ActuatorMotors()
        timestamp_us = int(self.get_clock().now().nanoseconds / 1000)
        if hasattr(msg, "timestamp"):
            msg.timestamp = timestamp_us
        if hasattr(msg, "timestamp_sample"):
            msg.timestamp_sample = timestamp_us
        if hasattr(msg, "reversible_flags"):
            msg.reversible_flags = 0

        control = [float("nan")] * 12
        for idx in range(self.motor_count):
            control[idx] = value if idx == motor_index else self.base_value
        if self.sequence_complete:
            for idx in range(self.motor_count):
                control[idx] = self.stop_value
        msg.control = control

        self.motor_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = Px4MotorSequenceNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
