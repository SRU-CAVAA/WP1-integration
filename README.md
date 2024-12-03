# Motivational autonomy. Navigating motivational cognitive maps.

## Description

This repository includes a Webots simulation environment where an epuck robot navigates different environments in search of different rewards.

The repository includes different executable files:
* **epuck_launch.py**:
	- Launches the **Webots simulation** considering the hyperparameters specified in the config.ini file.
	- Launches the **supervisor** node that allows simulation reset, metrics and plotting.
	- Launches the **data gathering** node. This node gathers simulation metrics to save it at the end of the experiment in a CSV file.

* **experiment.py**: Launches the **control architecture** that will control de agent. Through the hyperparameters in config.ini the configuration of the agent and the layers involved can be defined. Also through the config.ini file experiment features are defined.


## Getting started

This repository requires the previous installation of **ROS2 Iron Irwini** in WSL-Ubuntu 22.04. 
It also requires to install [webots_ros2](https://docs.ros.org/en/iron/Tutorials/Advanced/Simulators/Webots/).
Then create a ROS2 workspace in WSL-Ubuntu and copy the content of this repository in the src folder.

Instructions to run this repository:

Open two WSL-Ubuntu 22.04 terminals and:
*colcon build
1. Source each terminal: source install/local_setup.bash
2. Terminal 1: ros2 launch ca_webots_pkg epuck_launch.py
3. Terminal 2: ros2 run ca_architecture_pkg experiment



Python version = 3.10.12
CUDA = V11.5.119
Torch version = 2.1.1+cu118
Torchvision = 0.16.1+cu118
Torchaudio = 2.1.1+cu118


## Project WSL

This project is developed in WSL 2. Please, contact me if you would like to have access to this WSL.

### Imporint Project WSL
1. Import WSL
2. Using Windows Defender Firewall with Advance Security, create an Inboud rule for port 1234.
3. Edit etc/wsl.conf. Add:

		[boot]
		systemd=true

		[network]
		generateResolvConf = false

4. Edit etc/resolv.conf. Add:

		nameserver 172.xx.xxx.x (your Windows IP)

		search imec.be


