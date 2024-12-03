import os
import launch
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from webots_ros2_driver.webots_launcher import WebotsLauncher
from webots_ros2_driver.webots_controller import WebotsController
from launch_ros.actions import Node

from pathlib import Path
ws_path = str(Path(__file__).parents[5])
print(ws_path)

import configparser
parameters = configparser.ConfigParser()
parameters.read(ws_path + '/src/config.ini')

webots_world = int(parameters.get('Experiment', 'webots_world'))
if webots_world == 0:
    world = 'Open_arena'
if webots_world == 1:
    world = 'Linear_Track'
if webots_world == 2:
    world = 'Tmaze'
if webots_world == 3:
    world = 'Double_Tmaze'



def get_ros2_control_spawners(*args):
    # Declare here all nodes that must be restarted at simulation reset
    ros_control_node = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['diffdrive_controller']
    )
    return [
        ros_control_node
    ]


def generate_launch_description():
    package_dir = get_package_share_directory('ca_webots_pkg')
    robot_description_path = os.path.join(package_dir, 'resource', 'webots_epuck.urdf')

    webots = WebotsLauncher(
        world=os.path.join(package_dir, 'worlds', world+'.wbt'),
        ros2_supervisor=True
    )

    epuck_agent = WebotsController(
        robot_name='my_epuck',
        parameters=[
            {'robot_description': robot_description_path}
        ],
        # Every time one resets the simulation the controller is automatically respawned
        respawn=True
    )

    supervisor_node = Node(
        package='ca_webots_pkg',
        executable='supervisor',
        output='screen',
    )

    experiment_node = Node(
        package='ca_architecture_pkg',
        executable='experiment',
        output='screen',
        )

    data_gathering_node = Node(
        package='ca_webots_pkg',
        executable='data_gathering',
        output='screen',
    )


    # Declare the reset handler that respawns nodes when robot_driver exits
    reset_handler = launch.actions.RegisterEventHandler(
        event_handler=launch.event_handlers.OnProcessExit(
            target_action=epuck_agent,
            on_exit=get_ros2_control_spawners,
        )
    )

    return LaunchDescription([
        webots,
        webots._supervisor,
        epuck_agent,
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=webots,
                on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
            )
        )
    ])