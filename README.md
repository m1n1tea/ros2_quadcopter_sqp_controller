# Quadrotor Moving-Target Interception using Model Predictive Control

## ROS 2 PX4 MPC Node
This repository now includes a ROS 2 package `quadrotor_acados` that:
- subscribes to `px4_msgs/msg/VehicleAttitudeSetpoint` for desired attitude and
  collective-thrust references,
- uses only attitude quaternion and angular velocity from
  `px4_msgs/msg/VehicleOdometry`,
- publishes `px4_msgs/msg/ActuatorMotors` by default, or
  `px4_msgs/msg/VehicleRatesSetpoint` when configured.

The launch file uses `quadrotor_acados/config/x500.yaml` by default.

### Hover with PX4 position control, then hand over to MPC

`px4_pid_node` publishes PX4 `TrajectorySetpoint` messages; the actual
position controller is PX4.  A NED z value of `-2.0` is approximately 2 m
above the local origin.  First start the PID bridge, then publish its one-point
reference:
```bash
# Terminal 1: PX4 SITL (or the vehicle bridge) must already be running.
# Terminal 2
ros2 run quadrotor_acados px4_pid_node

# Terminal 3
ros2 run quadrotor_acados single_point_path_publisher --ros-args \
  -p x:=0.0 -p y:=0.0 -p z:=-2.0
```

Wait for the vehicle to settle near z = -2 m.  While it is still hovering,
start the target publisher and pre-initialize MPC without allowing it to
publish PX4 commands:
```bash
# Terminal 4
ros2 launch quadrotor_acados launch_moving_target.py z:=-2.0

# Terminal 5
ros2 run quadrotor_acados px4_mpc_node --ros-args \
  --params-file $(ros2 pkg prefix quadrotor_acados)/share/quadrotor_acados/config/x500.yaml \
  -p enabled:=false
```

Then stop `px4_pid_node` and immediately enable MPC:
```bash
ros2 param set /px4_mpc_node enabled true
```

Do not run enabled PID and MPC nodes together: both publish PX4 Offboard
control-mode and vehicle-command messages.  The `enabled` parameter defaults
to `true`; its purpose is to let MPC initialize before the handoff, not to
change normal direct-start behavior.

For the same handoff as one command, run this workspace-level script:
```bash
./run_pid_to_mpc.sh
```
It launches PX4 `gz_x500`, waits for the PID log line `Reached final reference
point`, starts the moving-target publisher and a disabled MPC, then stops PID
and enables MPC. On shutdown it also turns `target.log` into
`target_tracking.png` and `target_tracking_attitude.png` under
`logs/pid_to_mpc_*`.
For an existing PX4 instance, use `START_PX4=0 ./run_pid_to_mpc.sh`; all
supported position, target, timeout, and parameter-file overrides are listed
by `./run_pid_to_mpc.sh --help`.

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

`px4_mpc_node` receives `VehicleAttitudeSetpoint` on
`target_attitude_setpoint_topic`. Its `q_d` field is the desired NED/FRD
quaternion, and `thrust_body[2]` is the negative normalized physical thrust.
The node converts that physical thrust to the MPC's configured motor-command
range before optimizing torque commands.

`moving_target_publisher` calculates this setpoint from vehicle odometry and
the simulated moving target. It applies configured direction/range noise,
points yaw toward the target, and limits tilt with `max_tilt_deg`. Its thrust
parameters include `uav_mass`, `thrust_constant`, `min_rotor_speed`, and
`max_rotor_speed`; this preserves correct hover compensation with a nonzero
minimum rotor speed.

### Target setpoint example

Publish a stationary level-attitude reference with 57% normalized physical
thrust:
```bash
ros2 topic pub --once --qos-reliability reliable --qos-durability transient_local \
  /target/attitude_setpoint px4_msgs/msg/VehicleAttitudeSetpoint \
  "{q_d: [1.0, 0.0, 0.0, 0.0], thrust_body: [0.0, 0.0, -0.57]}"
```

The debug publisher computes a moving world target, listens to vehicle
odometry, and publishes noisy attitude/thrust setpoints. Run it directly:
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
hold collective thrust at its baseline. `log_interval_sec:=0.0` disables the
human-readable status log. The publisher also writes JSON telemetry records at
`telemetry_log_interval_sec` (default 0.1 s), containing synchronized vehicle
and target NED positions, desired/observed quaternions and Euler angles, and
thrust. It writes a collision record when the distance first enters
`collision_radius_m` (default 0.5 m).

Capture its output and produce a 3D quadcopter/target/collision plot plus a
desired-versus-observed roll, pitch, and yaw plot:
```bash
ros2 run quadrotor_acados moving_target_publisher --ros-args \
  -p telemetry_log_interval_sec:=0.05 2>&1 | tee ros2_log_target

python3 utils/analyze_target_tracking_log.py \
  --target-log ros2_log_target \
  --save-plot target_tracking.png --no-show
```

The second image is saved as `target_tracking_attitude.png`; override it with
`--save-attitude-plot`. The first collision is a red star and later collisions
are orange crosses. The analyzer retains legacy controller-log parsing when
new telemetry is unavailable.

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
