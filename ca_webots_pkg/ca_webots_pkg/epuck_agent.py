import os
import sys
import rclpy
from rosgraph_msgs.msg import Clock
from ca_msgs_pkg.msg import EpuckMotors

from pathlib import Path
ws_path = str(Path(__file__).parents[6])

import configparser
parameters = configparser.ConfigParser()
parameters.read(ws_path + '/src/config.ini')


class MyEpuckAgent:
    def init(self, webots_node, properties):
        self.state_action_rate = float(parameters.get('Reactive_Layer', 'state_action_rate'))

        self.time = 0
        self.action_init_time = 0
        self.action_current_time = 0
        self.action_2_time = 0
        self.elapsed_time = 0
        self.action_1_time = 0

        self.__robot = webots_node.robot

        self.__left_motor = self.__robot.getDevice('left wheel motor')
        self.__right_motor = self.__robot.getDevice('right wheel motor')

        self.__left_motor.setPosition(float('inf'))
        self.__left_motor.setVelocity(0)

        self.__right_motor.setPosition(float('inf'))
        self.__right_motor.setVelocity(0)

        self.__epuck_motors = EpuckMotors()

        rclpy.init(args=None)
        self.__node = rclpy.create_node('epuck_agent')
        self.__node.create_subscription(EpuckMotors, 'epuck_agent/motor_cmd', self.__cmd_vel_callback, 1)
        self.__node.create_subscription(Clock, '/clock', self.clock_callback, 1)

        self.__node.get_logger().info('------------- ----------------- epuck_agent -------------- -------------')

    def __cmd_vel_callback(self, data):
        self.__epuck_motors.right_motor = data.right_motor
        self.__epuck_motors.left_motor = data.left_motor
        self.action_init_time = self.time
        self.action_current_time = self.time - self.action_init_time



    def clock_callback(self, message):
        self.clock = message.clock
        self.secs = self.clock.sec
        self.nanosecs = self.clock.nanosec/1000000000
        self.time = self.secs + self.nanosecs
        self.action_current_time = self.time - self.action_init_time


    def step(self):
        if self.action_current_time >= self.state_action_rate:
            self.__node.get_logger().info('Warning - ACTION TIME SURPASSED: ' + str(self.action_current_time))
            self.__epuck_motors.right_motor = 0
            self.__epuck_motors.left_motor = 0

        rclpy.spin_once(self.__node, timeout_sec=0)

        command_motor_right = self.__epuck_motors.right_motor
        command_motor_left = self.__epuck_motors.left_motor

        self.__left_motor.setVelocity(command_motor_left)
        self.__right_motor.setVelocity(command_motor_right)