import sys
import ast
import time
import rclpy
import random
import numpy as np
import matplotlib.pyplot as plt

from rclpy.node import Node
from std_msgs.msg import String
from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import PointStamped
from webots_ros2_msgs.msg import FloatStamped
from ca_msgs_pkg.msg import TrialINFO, TimeINFO, RewardID, ExpStatus, AEunits, HomeostaticStates, Replay, FrameID

from webots_ros2_msgs.srv import SpawnNodeFromString

from pathlib import Path
ws_path = str(Path(__file__).parents[6])
print(ws_path)

import configparser
parameters = configparser.ConfigParser()
parameters.read(ws_path + '/src/config.ini')


class Supervisor(Node):

    def __init__(self):
        super().__init__('supervisor')
        self.get_logger().info('------------- ----------------- Supervisor -------------- -------------')


        #----------------- ROS TOPIC SUBSCRIPTIONS AND PUBLICATIONS -----------------
        self.create_subscription(PointStamped, '/my_epuck/gps', self.agent_GPS_callback, 1)
        self.create_subscription(FloatStamped, '/my_epuck/compass/bearing', self.agent_Compass_callback, 1)
        self.create_subscription(AEunits, '/epuck_agent/AEunits', self.AEunits_callback, 1)
        self.create_subscription(HomeostaticStates, '/epuck_agent/homeostatic_states', self.Internal_state_callback, 1)
        self.create_subscription(Replay, '/epuck_agent/Replay', self.Replay_callback, 1)
        self.create_subscription(TimeINFO, '/experiment/Time', self.exp_time_callback, 1)
        self.create_subscription(Clock, '/clock', self.clock_callback, 1)
        self.create_subscription(FrameID, 'epuck_agent/frame_ID', self.frame_ID_callback, 1)
        self.create_subscription(ExpStatus, '/experiment/status', self.experiment_status_callback, 1)

        self.pub_trial_INFO = self.create_publisher(TrialINFO, 'Supervisor/Trial_INFO', 1)
        self.pub_reward_ID = self.create_publisher(RewardID, 'Supervisor/Reward_ID', 1)
        self.pub_remove_Node = self.create_publisher(String, '/Ros2Supervisor/remove_node', 1)
        self.pub_exp_status = self.create_publisher(ExpStatus, 'experiment/status', 1)


        #----------------- ROS SERVICES -----------------
        self.cli = self.create_client(SpawnNodeFromString, '/Ros2Supervisor/spawn_node_from_string')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.SpawnNode_req = SpawnNodeFromString.Request()


        #----------------- SUPERVISOR HYPERPAPRAMETERS -----------------
        self.webots_world = int(parameters.get('Experiment', 'webots_world'))
        self.agent_mode = int(parameters.get('Agent', 'agent_mode'))
        self.total_trials = int(parameters.get('Experiment', 'total_trials'))
        self.reset_reward_margin = float(parameters.get('Environment', 'reset_reward_margin'))
        self.trial_time_limit = int(parameters.get('Experiment', 'trial_time_limit'))
        self.plotting = int(parameters.get('Experiment', 'plotting'))
        self.frame_limit = int(parameters.get('Experiment', 'frame_limit'))
        n_hidden_list = ast.literal_eval(parameters.get('Adaptive_Layer', 'n_hidden').replace('[', '["').replace(']', '"]').replace(', ', '", "'))
        if parameters.get('Reactive_Layer', 'img_saving') == 'True':
            self.img_saving = True
        else:
            self.img_saving = False

        if parameters.get('Experiment', 'rand_start') == 'True':
            self.rand_start = True
        else:
            self.rand_start = False

        if self.webots_world == 0:
            self.n_hidden = int(n_hidden_list[0])
            self.Epuck_translation = parameters.get('Environment', 'openarena_epuck_translation')
            self.Epuck_rotation = parameters.get('Environment', 'openarena_epuck_rotation')
            self.Green_reward_X = float(parameters.get('Environment', 'openarena_green_reward_X'))
            self.Green_reward_Y = float(parameters.get('Environment', 'openarena_green_reward_Y'))

        if self.webots_world == 1:
            self.n_hidden = int(n_hidden_list[1])
            self.Epuck_translation = parameters.get('Environment', 'lineartrack_epuck_translation')
            self.Epuck_rotation = parameters.get('Environment', 'lineartrack_epuck_rotation')
            self.Green_reward_X = float(parameters.get('Environment', 'lineartrack_green_reward_X'))
            self.Green_reward_Y = float(parameters.get('Environment', 'lineartrack_green_reward_Y'))

        if self.webots_world == 2:
            self.n_hidden = int(n_hidden_list[2])
            self.Epuck_translation = parameters.get('Environment', 'tmaze_epuck_translation')
            self.Epuck_rotation = parameters.get('Environment', 'tmaze_epuck_rotation')
            self.Red_reward_X = float(parameters.get('Environment', 'tmaze_red_reward_X'))
            self.Red_reward_Y = float(parameters.get('Environment', 'tmaze_red_reward_Y'))
            self.Blue_reward_X = float(parameters.get('Environment', 'tmaze_blue_reward_X'))
            self.Blue_reward_Y = float(parameters.get('Environment', 'tmaze_blue_reward_Y'))

        if self.webots_world == 3:
            self.n_hidden = int(n_hidden_list[3])
            self.Epuck_translation = parameters.get('Environment', 'double_tmaze_epuck_translation')
            self.Epuck_rotation = parameters.get('Environment', 'double_tmaze_epuck_rotation')
            self.Red_reward_X = float(parameters.get('Environment', 'double_tmaze_red_reward_X'))
            self.Red_reward_Y = float(parameters.get('Environment', 'double_tmaze_red_reward_Y'))
            self.Blue_reward_X = float(parameters.get('Environment', 'double_tmaze_blue_reward_X'))
            self.Blue_reward_Y = float(parameters.get('Environment', 'double_tmaze_blue_reward_Y'))
            self.Purple_reward_X = float(parameters.get('Environment', 'double_tmaze_purple_reward_X'))
            self.Purple_reward_Y = float(parameters.get('Environment', 'double_tmaze_purple_reward_Y'))
            self.Orange_reward_X = float(parameters.get('Environment', 'double_tmaze_orange_reward_X'))
            self.Orange_reward_Y = float(parameters.get('Environment', 'double_tmaze_orange_reward_Y'))

        self.agent_x_position = -9999
        self.agent_y_position = -9999
        self.agent_orientation = 0

        if self.agent_mode == 3:
            self.block_lenght = 5000
        else:
            self.block_lenght = 200

        self.list_X = []
        self.list_Y = []
        self.list_Z = []

        self.trial_number = 0
        self.reward_ID = 0
        self.last_reward = 0
        self.sim_time = 0
        self.experiment_status = 0
        self.spawned_rewards = [0, 0, 0, 0, 0, 0]

        self.AEunits = np.random.rand(self.n_hidden)
        self.AEunits_colorbar = None
        self.AEprediction = np.random.rand(120, 160, 3)

        self.time_list = []

        self.internal_state = [0,0,0,0]
        self.state1_list = []
        self.state2_list = []
        self.state3_list = []
        self.state4_list = []


        if self.plotting == 1:
            plt.ion()
            plt.style.use('seaborn')
            self.fig1, self.ax1 = plt.subplots(1, 1, figsize=(7, 4))
            self.fig2, self.ax2 = plt.subplots(1, 1, figsize=(10, 4))
            self.fig3, self.ax3 = plt.subplots(1, 1, figsize=(8, 6))

#----------------- ROS CALLBACK FUNCTIONS -----------------

    def agent_GPS_callback(self, message):
        self.agent_x_position = message.point.x
        self.agent_y_position = message.point.y

    def agent_Compass_callback(self, message):
        self.agent_orientation = message.data
            
    def AEunits_callback(self, message):
        self.AEunits = message.units
        self.AEunits = np.array(self.AEunits)
        self.AEprediction = np.array(message.prediction).reshape((120, 160, 3))

    def Internal_state_callback(self, message):
        self.internal_state = [message.state_1, message.state_2, message.state_3, message.state_4]


    def Replay_callback(self, message):
        self.get_logger().info('Replays completed')
        self.n_replays = message.n_replays
        self.update_experiment_status(3) #Data_gathering save trial

        self.end_trial()

    def exp_time_callback(self, message):
        self.sim_time = message.time

    def frame_ID_callback(self, message):
        self.frame_ID = message.frame_id
        if self.img_saving == True:
            print("Picture: " + str(self.frame_ID) + "/" + str(self.frame_limit))


    def experiment_status_callback(self, message):
        self.experiment_status = message.status



    def clock_callback(self, message):

        if self.experiment_status == 0:
            self.spawn_Epuck()
            self.check_rewards_spawned(self.reward_ID)
            self.update_experiment_status(1)
            print()
            self.get_logger().info("Starting Trial: " + str(self.trial_number + 1) + "/" + str(self.total_trials))




        #---------------------  EXPERIMENT LOOP  ---------------------
        if self.experiment_status == 1:
            self.trial_count()
            self.monitor_simulation_reset()
            if self.plotting == 1:
                self.plot_AEunits()
                self.plot_internal_state()
                self.plot_AE_prediction()


        if self.experiment_status == 5:
            self.list_X = []
            self.list_Y = []
            self.list_Z = []
            self.trial_number = 0
            self.AEunits = np.random.rand(self.n_hidden)
            self.time_list = []
            self.internal_state = [0,0,0,0]
            self.state1_list = []
            self.state2_list = []
            self.state3_list = []
            self.state4_list = []
            self.update_experiment_status(0)




    #---------------------  SUPERVISOR FUNCTIONS  ---------------------

    def plot_AEunits(self):
        if self.n_hidden == 1000: n_hidden_dim = [51, 21]
        if self.n_hidden == 400: n_hidden_dim = [21, 21]
        if self.n_hidden == 200: n_hidden_dim = [21, 11]
        
        x, y = np.meshgrid(np.arange(1, n_hidden_dim[0]), np.arange(1, n_hidden_dim[1]))
        x = x.flatten()
        y = y.flatten()


        self.ax1.cla()
        self.ax1.grid(False)
        scatter = self.ax1.scatter(x, y, c=self.AEunits, cmap='viridis', s=150, edgecolors='black', linewidth=1)
        self.ax1.set_title('Unit activation', fontsize= 15)
        self.ax1.set_xticks(np.arange(1, n_hidden_dim[0]))
        self.ax1.set_yticks(np.arange(1, n_hidden_dim[1]))

        if self.AEunits_colorbar:
            self.AEunits_colorbar.remove()

        self.AEunits_colorbar = self.fig1.colorbar(scatter, ax=self.ax1)
        self.AEunits_colorbar.set_label('AEunits')
        self.fig1.canvas.flush_events()


    def plot_internal_state(self):
        self.time_list.append(self.sim_time)

        if len(self.time_list) >= 100: self.time_list.pop(0)
        if len(self.state1_list) >= 100: self.state1_list.pop(0)
        if len(self.state2_list) >= 100: self.state2_list.pop(0)
        if len(self.state3_list) >= 100: self.state3_list.pop(0)
        if len(self.state4_list) >= 100: self.state4_list.pop(0)

        self.ax2.cla()
        if self.webots_world == 0:
            self.state1_list.append(self.internal_state[0])
            self.ax2.plot(self.state1_list, color='green', label='state1')
        if self.webots_world == 2:
            self.state1_list.append(self.internal_state[0])
            self.state2_list.append(self.internal_state[1])
            self.ax2.plot(self.state1_list, color='blue', label='state1')
            self.ax2.plot(self.state2_list, color='red', label='state2')
        if self.webots_world == 3:
            self.state1_list.append(self.internal_state[0])
            self.state2_list.append(self.internal_state[1])
            self.state3_list.append(self.internal_state[2])
            self.state4_list.append(self.internal_state[3])
            self.ax2.plot(self.state1_list, color='blue', label='state1')
            self.ax2.plot(self.state2_list, color='red', label='state2')
            self.ax2.plot(self.state3_list, color='purple', label='state3')
            self.ax2.plot(self.state4_list, color='orange', label='state4')
        self.ax2.set_title('Internal states', fontsize= 15)
        self.ax2.set_ylim(-0.1, 1.1)
        ticks = np.arange(0, 101, 10)
        labels = np.linspace(min(self.time_list), max(self.time_list), len(ticks))
        rounded_labels = [round(label, 2) for label in labels]
        self.ax2.set_xticks(ticks)
        self.ax2.set_xticklabels(rounded_labels)
        self.ax2.set_xlabel("Time (s)", fontsize=12)
        self.fig2.canvas.flush_events()


    def plot_AE_prediction(self):
        self.ax3.cla()
        self.ax3.imshow(self.AEprediction)
        self.ax3.axis('off')
        self.fig3.canvas.flush_events()




    def monitor_simulation_reset(self):
        if self.img_saving == False:
            self.Reward_catching()
        self.unblocking_reset()
        self.Time_reset()

    def Reward_catching(self):
        if self.webots_world == 0 and self.experiment_status == 1:
            if self.agent_x_position > self.Green_reward_X - self.reset_reward_margin and self.agent_x_position < self.Green_reward_X + self.reset_reward_margin:
                if self.agent_y_position > self.Green_reward_Y - self.reset_reward_margin and self.agent_y_position < self.Green_reward_Y + self.reset_reward_margin:
                    self.reward_ID = 1
                    rewardball = 'rewardball_green'
                    print('------- GREEN REWARD CAPTURED -------')

        if self.webots_world == 1 and self.experiment_status == 1:
            if self.agent_x_position > self.Green_reward_X - self.reset_reward_margin and self.agent_x_position < self.Green_reward_X + self.reset_reward_margin:
                if self.agent_y_position > self.Green_reward_Y - self.reset_reward_margin and self.agent_y_position < self.Green_reward_Y + self.reset_reward_margin:
                    self.reward_ID = 1
                    rewardball = 'rewardball_green'
                    print('------- GREEN REWARD CAPTURED -------')

        if self.webots_world >= 2 and self.experiment_status == 1:
            if self.agent_x_position > self.Blue_reward_X - self.reset_reward_margin:
                if self.agent_y_position > self.Blue_reward_Y - self.reset_reward_margin and self.agent_y_position < self.Blue_reward_Y + self.reset_reward_margin:
                    self.reward_ID = 2
                    rewardball = 'rewardball_blue'
                    print('------- BLUE REWARD CAPTURED -------')

            if self.agent_x_position < self.Red_reward_X + self.reset_reward_margin:
                if self.agent_y_position > self.Red_reward_Y - self.reset_reward_margin and self.agent_y_position < self.Red_reward_Y + self.reset_reward_margin:
                    self.reward_ID = 3
                    rewardball = 'rewardball_red'
                    print('------- RED REWARD CAPTURED -------')
                
        if self.webots_world >= 3 and self.experiment_status == 1:
            if self.agent_x_position < self.Purple_reward_X + self.reset_reward_margin:
                if self.agent_y_position > self.Purple_reward_Y - self.reset_reward_margin and self.agent_y_position < self.Purple_reward_Y + self.reset_reward_margin:
                    self.reward_ID = 4
                    rewardball = 'rewardball_purple'
                    print('------- PURPLE REWARD CAPTURED -------')

            if self.agent_x_position > self.Orange_reward_X - self.reset_reward_margin:
                if self.agent_y_position > self.Orange_reward_Y - self.reset_reward_margin and self.agent_y_position < self.Orange_reward_Y + self.reset_reward_margin:
                    self.reward_ID = 5
                    rewardball = 'rewardball_orange'
                    print('------- ORANGE REWARD CAPTURED -------')


        if self.reward_ID != 0:
            self.last_reward = self.reward_ID
            self.remove_Node(rewardball)
            msg = RewardID()
            msg.reward_id = self.reward_ID
            self.pub_reward_ID.publish(msg)
            self.reward_ID = 0

            self.update_experiment_status(2)



    def trial_count(self):
        msg = TrialINFO()
        msg.trial = self.trial_number
        self.pub_trial_INFO.publish(msg)


    def check_rewards_spawned(self, last_reward):

        if self.webots_world == 0: rewards = [0,1,0,0,0,0]
        if self.webots_world == 1: rewards = [0,1,0,0,0,0]
        if self.webots_world == 2: rewards = [0,0,1,1,0,0]
        if self.webots_world == 3: rewards = [0,0,1,1,1,1]

        missing_rewards = [i for i in range(len(rewards)) if rewards[i] != self.spawned_rewards[i]]

        if last_reward != 0:
            missing_rewards = [last_reward]

        for reward_color in missing_rewards:
            if reward_color == 1: color = 'Green'
            if reward_color == 2: color = 'Blue'
            if reward_color == 3: color = 'Red'
            if reward_color == 4: color = 'Purple'
            if reward_color == 5: color = 'Orange'

            self.spawn_Rewardball(color)
            self.spawned_rewards[reward_color] = 1


    def update_experiment_status(self, exp_status):
        self.experiment_status = exp_status
        msg = ExpStatus()
        msg.status = self.experiment_status
        self.pub_exp_status.publish(msg)


    def send_SpawnNode_request(self, request):
        self.SpawnNode_req.data =  request
        self.future = self.cli.call_async(self.SpawnNode_req) #call_async
        #rclpy.spin_until_future_complete(self, self.future, timeout_sec=5.0)
        #return self.future.result()

    def randomize_start_location(self):
        print('Randomizing start location')
        xy_limits = [1.15, 0.3]
        xpos = random.uniform(-xy_limits[0], xy_limits[0])
        ypos = random.uniform(-xy_limits[0], xy_limits[0])

        while xpos < xy_limits[1] and ypos < xy_limits[1]:
            xpos = random.uniform(-xy_limits[0], xy_limits[0])
            ypos = random.uniform(-xy_limits[0], xy_limits[0])
        zpos = random.uniform(0, 3.14)

        self.Epuck_translation = str(xpos) + ' ' + str(ypos) + ' 0.0'
        self.Epuck_rotation = '0.0 0.0 1.0 ' + str(zpos)
        

    def spawn_Epuck(self):
        if self.rand_start == True:
            self.randomize_start_location()
        request = "E-puck { name \"my_epuck\", translation " + self.Epuck_translation + ", rotation " + self.Epuck_rotation + ", controller \"<extern>\", camera_width 160, camera_height 120, distance_sensor_numberOfRays 3, turretSlot [Compass {} GPS {}]}"
        self.send_SpawnNode_request(request)
        time.sleep(2)
        '''response = self.send_SpawnNode_request(request)
        self.get_logger().info('Epuck spawned = ' + str(response.success))
        return response.success'''

    def remove_Node(self, node):
        msg = String()
        msg.data = node
        self.pub_remove_Node.publish(msg)


    def spawn_Rewardball(self, color):
        self.get_logger().info('Spawning reward ' + color)
        if color == 'Green':
            request = "RewardBall { name \"rewardball_green\", translation " + str(self.Green_reward_X) + " " + str(self.Green_reward_Y) + " 0.0396, colorOverride 0 1 0 }"
        if color == 'Blue':
            request = "RewardBall { name \"rewardball_blue\", translation " + str(self.Blue_reward_X) + " " + str(self.Blue_reward_Y) + " 0.0396, colorOverride 0 0 1 }"
        if color == 'Red':
            request = "RewardBall { name \"rewardball_red\", translation " + str(self.Red_reward_X) + " " + str(self.Red_reward_Y) + " 0.0396, colorOverride 1 0 0 }"
        if color == 'Purple':
            request = "RewardBall { name \"rewardball_purple\", translation " + str(self.Purple_reward_X) + " " + str(self.Purple_reward_Y) + " 0.0396, colorOverride 0.6 0 1 }"
        if color == 'Orange':
            request = "RewardBall { name \"rewardball_orange\", translation " + str(self.Orange_reward_X) + " " + str(self.Orange_reward_Y) + " 0.0396, colorOverride 1 0.4 0 }"

        self.send_SpawnNode_request(request)

        '''response = self.sex_SpawnNode_request(request)
        self.get_logger().info('Epuck spawned = ' + str(response.success))
        return response.success'''


    def end_trial(self):
        self.trial_number += 1
        time.sleep(0.2)

        if self.trial_number >= self.total_trials or (self.img_saving == True and self.frame_ID >= self.frame_limit):
            self.update_experiment_status(4)
        else:
            self.update_experiment_status(0)
            
        self.remove_Node('my_epuck')

        self.agent_x_position = 9 #Arbitrary value outbound of arena
        self.agent_y_position = 9
        self.agent_orientation = 0

        self.spawn_Epuck()
        self.check_rewards_spawned(self.last_reward)

        self.time_list = []


    def reset_trial(self):
        self.get_logger().info("Reseting trial: " + str(self.trial_number) + "/" + str(self.total_trials))

        self.update_experiment_status(2) # 0 for not LTM uptade in fail trails
        self.remove_Node('my_epuck')
        self.spawn_Epuck()



    def unblocking_reset(self):
        self.list_X.append(self.agent_x_position)
        self.list_Y.append(self.agent_y_position)

        if self.experiment_status == 0:
            self.list_X = []
            self.list_Y = []

        if len(self.list_X) > self.block_lenght:
            self.list_X.pop(0)
            self.list_Y.pop(0)

            X_diff = max(self.list_X) - min(self.list_X)
            Y_diff = max(self.list_Y) - min(self.list_Y)

            if X_diff < 0.02 and Y_diff < 0.02 and self.experiment_status == 1:
                print("UNBLOCKING RESET")
                self.reset_trial()

                '''self.experiment_status = 2
                msg = ExpStatus()
                msg.status = self.experiment_status
                self.Experiment__publisher.publish(msg)'''


    def Time_reset(self):
        if self.trial_time_limit!= 0 and self.sim_time > self.trial_time_limit:
            print("TIME RESET")
            self.reset_trial()

            '''self.experiment_status = 2
            msg = ExpStatus()
            msg.status = self.experiment_status
            self.Experiment__publisher.publish(msg)'''





    



def main(args=None):
    rclpy.init(args=args)
    supervisor = Supervisor()
    rclpy.spin(supervisor)
    supervisor.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()





# Start - Spawn robot and rewards.
# Experiment - Monitorize robot reward catching and remove and respawn the robot and the catched reward.
# Experiment - Monitorize robot do not get stuck. If so, remove and respawn robot.
# Experiment - For each trial, restart simulation time.
# End - Finish experiment when number of trials are completed.