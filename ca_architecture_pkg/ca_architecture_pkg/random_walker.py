import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Range
from ca_msgs_pkg.msg import EpuckMotors
from webots_ros2_msgs.msg import FloatStamped
import random

from pathlib import Path
ws_path = str(Path(__file__).parents[6])


import configparser
parameters = configparser.ConfigParser()
parameters.read(ws_path + '/src/config.ini')


class RandomWalker(Node):
    def __init__(self):
        super().__init__('allostatic_controller')

        

        self.wheel_r = 0.1
        self.wheel_l = 0.5


        self.get_logger().info('------------- ----------------- allostatic_controller -------------- -------------')

        self.Motor__publisher = self.create_publisher(EpuckMotors, 'epuck_agent/motor_cmd', 1)
        
        self.create_subscription(Range, 'epuck_agent/ps0', self.__ps0_sensor_callback, 1)


        


    def __ps0_sensor_callback(self, message):
        self.ps0 = message.range
        print(self.ps0)

        command_message = EpuckMotors()
        command_message.right_motor = self.wheel_r
        command_message.left_motor = self.wheel_l

        self.Motor__publisher.publish(command_message)



def main(args=None):
    rclpy.init(args=args)
    walker = RandomWalker()
    rclpy.spin(walker)
    walker.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()