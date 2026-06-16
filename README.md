# Quadrotor Moving-Target Interception using Model Predictive Control

## ROS 2 PX4 MPC Node
This repository now includes a ROS 2 package `quadrotor_acados` that:
- subscribes to `nav_msgs/msg/Odometry` for target direction/range observations,
- uses only attitude quaternion and angular velocity from
  `px4_msgs/msg/VehicleOdometry`,
- publishes `px4_msgs/msg/ActuatorMotors` by default, or
  `px4_msgs/msg/VehicleRatesSetpoint` when configured.

The launch file uses `quadrotor_acados/config/x500.yaml` by default.

Build and run:
```bash
source /opt/ros/kilted/setup.bash
colcon build --packages-select quadrotor_acados
source install/setup.bash
ros2 launch quadrotor_acados launch_mpc.py
```

If the workspace was moved between the devcontainer and host, rebuild generated
dependencies in the current path before rebuilding the controller:
```bash
source /opt/ros/kilted/setup.bash
rm -rf build/px4_msgs install/px4_msgs build/quadrotor_acados
colcon build --packages-select px4_msgs quadrotor_acados
```

### Python dependencies
Install pip dependencies from `requirements.txt`:
```bash
python3 -m pip install --upgrade pip setuptools wheel Cython
python3 -m pip install -r requirements.txt
```

To repair an existing activated venv before building ROS message
packages:
```bash
python -m pip install catkin_pkg "empy==3.3.4" lark
```

If the virtual environment is located inside the ROS 2 workspace, keep a
`COLCON_IGNORE` marker at its root so colcon does not scan Python packages:
```bash
touch venv_acados/COLCON_IGNORE
```

Topic names and node runtime parameters are defined in `quadrotor_acados/config/x500.yaml`.
The MPC state is `[quaternion, body_rates]`; position and linear velocity do not
enter the optimizer. `max_body_rate` configures hard FRD roll, pitch, and yaw
rate limits in `rad/s`.

The target observation uses `nav_msgs/msg/Odometry` as a transport:
- `pose.pose.position`: unit target direction in the PX4 NED world frame.
- `twist.twist.linear.x`: range in meters, or `NaN` when range is unavailable.

The node converts target direction and collective thrust into a desired
attitude. It points yaw toward the target and tilts toward it while reserving
enough vertical thrust to compensate gravity. `max_tilt_deg` caps this tilt.
`preferred_common_thrust` is the baseline per-motor command; a negative value
selects calculated hover thrust. When range is available, the command is
adjusted by range and direction: horizontal or upward targets increase thrust,
while downward targets reduce it. The result is clipped to
`min_common_thrust`/`max_common_thrust`.

### Target topic examples

Publish a target direction north and slightly upward, without range:
```bash
ros2 topic pub --once --qos-reliability reliable --qos-durability transient_local \
  /target/odometry nav_msgs/msg/Odometry \
  "{pose: {pose: {position: {x: 0.9285, y: 0.0, z: -0.3714}}}, \
    twist: {twist: {linear: {x: .nan}}}}"
```

The debug publisher computes a moving world target, listens to vehicle
odometry, and publishes a noisy relative observation. Run it directly:
```bash
ros2 run quadrotor_acados moving_target_publisher --ros-args \
  -p x:=30.0 -p y:=0.0 -p z:=-5.0 \
  -p vx:=0.0 -p vy:=2.5 -p vz:=0.25 \
  -p direction_noise_std:=0.02 \
  -p publish_distance:=true -p distance_noise_std:=0.25 \
  -p random_seed:=7
```

`direction_noise_std` is Gaussian noise added independently to the three
direction components before renormalization. Set `publish_distance:=false` to
test direction-only control. Set `log_interval_sec:=0.0` to disable logs.

Plot the vehicle and target trajectories and mark collision events:
```bash
python3 utils/analyze_target_tracking_log.py \
  --log ros2_log_mpc \
  --target-log ros2_log_target \
  --save-plot target_tracking.png
```

Omit `--target-log` when the MPC log contains `Received target observation`
messages. The first collision is shown as a red star; later collisions are
shown as orange crosses.

Set `command_output_mode: vehicle_rates_setpoint` to publish body-rate/thrust
setpoints on `vehicle_rates_setpoint_topic` instead of direct motor commands.
The node transforms the internally optimized body rates back to PX4 NED/FRD
before publishing the rates setpoint.

If you run the node directly, load the same config explicitly:
```bash
ros2 run quadrotor_acados px4_mpc_node --ros-args --params-file $(ros2 pkg prefix quadrotor_acados)/share/quadrotor_acados/config/x500.yaml
```

## PX4 Motor Sequence Node
For low-level motor checks, `px4_motor_sequence_node` publishes small direct
`ActuatorMotors` pulses to motors one by one. By default it does not arm the
vehicle; set `auto_arm:=true` only when the vehicle is safe for direct actuator
testing.

```bash
ros2 run quadrotor_acados px4_motor_sequence_node --ros-args -p base_value:=0.1
```

## Legacy Trajectory Publishers
The path publisher utilities remain available for experiments with the separate
PID node, but `px4_mpc_node` no longer subscribes to `nav_msgs/msg/Path`.

Generate the built-in sample trajectories as `.npy` files:
```bash
python3 utils/path_generators.py --output-dir /absolute/path/to/trajectories
```

Run one directly:
```bash
ros2 run quadrotor_acados path_publisher --ros-args \
  -p points_file:=/absolute/path/to/trajectories/square_path.npy
```

Or through launch:
```bash
ros2 launch quadrotor_acados launch_path_publisher.py \
  points_file:=/absolute/path/to/trajectories/square_path.npy
```

The reference path topic uses reliable transient-local QoS so controllers can
receive the last path even if DDS discovery finishes slightly after publication.
The sample publisher waits up to 5 seconds for a subscriber and remains alive
for 10 seconds after publishing. If publishing paths manually, use matching QoS:
```bash
ros2 topic pub --once --qos-reliability reliable --qos-durability transient_local \
  /reference_path nav_msgs/msg/Path "{...}"
```

To send a single-point reference path, defaulting to `(0, 0, -0.1)`:
```bash
ros2 run quadrotor_acados single_point_path_publisher
```

Override the point with parameters:
```bash
ros2 run quadrotor_acados single_point_path_publisher --ros-args \
  -p x:=0.0 -p y:=0.0 -p z:=-1.0
```

## Install Acados
To build Acados from source, see instructions [here](https://docs.acados.org/python_interface/index.html) or as follows:

Clone acados and its submodules by running:
```
git clone https://github.com/acados/acados.git
cd acados
git submodule update --recursive --init
```

Install acados as follows:

```
mkdir -p build
cd build
cmake -DACADOS_WITH_QPOASES=ON ..
make install -j4
```

Install acados_template Python package:
```
cd acados
python3 -m pip install --upgrade setuptools wheel Cython
pip install -e interfaces/acados_template
```
***Note:*** The ```<acados_root>``` is the full path from ```/home/```.

Add two paths below to ```~/.bashrc``` in order to add the compiled shared libraries ```libacados.so```, ```libblasfeo.so```, ```libhpipm.so``` to ```LD_LIBRARY_PATH``` (default path is ```<acados_root/lib>```):

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"<acados_root>/lib"
export ACADOS_SOURCE_DIR="<acados_root>"
```
