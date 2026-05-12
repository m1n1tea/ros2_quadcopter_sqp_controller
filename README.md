# Quadrotor Formation using Model Predictive Control

## ROS 2 PX4 MPC Node
This repository now includes a ROS 2 package `quadrotor_acados` that:
- subscribes to `nav_msgs/msg/Path` (reference trajectory),
- subscribes to `px4_msgs/msg/VehicleOdometry` (current state),
- publishes `px4_msgs/msg/ActuatorMotors` (motor command).

The launch file uses `quadrotor_acados/config/x500.yaml` by default.

Build and run:
```bash
colcon build --packages-select quadrotor_acados
source install/setup.bash
ros2 launch quadrotor_acados launch_mpc.py
```

### Python dependencies
Install pip dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

Topic names and node runtime parameters are defined in `quadrotor_acados/config/x500.yaml`.

If you run the node directly, load the same config explicitly:
```bash
ros2 run quadrotor_acados px4_mpc_node --ros-args --params-file $(ros2 pkg prefix quadrotor_acados)/share/quadrotor_acados/config/x500.yaml
```

## Sample Trajectory Publisher
The package includes a ROS 2 node that publishes an Nx3 `.npy` reference trajectory as `nav_msgs/msg/Path`.

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

## Install Acados
To build Acados from source, see instructions [here](https://docs.acados.org/python_interface/index.html) or as follows:

Clone acados and its submodules by running:
```
$ git clone https://github.com/acados/acados.git
$ cd acados
$ git submodule update --recursive --init
```

Install acados as follows:

```
$ mkdir -p build
$ cd build
$ cmake -DACADOS_WITH_QPOASES=ON ..
$ make install -j4
```

Install acados_template Python package:
```
$ cd acados
$ pip install -e interfaces/acados_template
```
***Note:*** The ```<acados_root>``` is the full path from ```/home/```.

Add two paths below to ```~/.bashrc``` in order to add the compiled shared libraries ```libacados.so```, ```libblasfeo.so```, ```libhpipm.so``` to ```LD_LIBRARY_PATH``` (default path is ```<acados_root/lib>```):

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"<acados_root>/lib"
export ACADOS_SOURCE_DIR="<acados_root>"
```
