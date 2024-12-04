import os
import csv
import ast
import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import String
from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import PointStamped
from webots_ros2_msgs.msg import FloatStamped
from ca_msgs_pkg.msg import EpuckMotors, HomeostaticStates, TrialINFO, TimeINFO, RewardID, ExpStatus, FrameID, AEunits, RewardValues, Replay

from pathlib import Path
ws_path = str(Path(__file__).parents[6])
print(ws_path)

import configparser
parameters = configparser.ConfigParser()
parameters.read(ws_path + '/src/config.ini')


class Data_gathering(Node):
    def __init__(self):
        super().__init__('data_gathering')

        self.get_logger().info('------------- ----------------- Data Gathering -------------- -------------')

        self.create_subscription(PointStamped, '/my_epuck/gps', self.agent_GPS_callback, 1)
        self.create_subscription(FloatStamped, '/my_epuck/compass/bearing', self.agent_Compass_callback, 1)
        self.create_subscription(HomeostaticStates, '/epuck_agent/homeostatic_states', self.homeostasis_callback, 1)
        self.create_subscription(RewardValues, '/epuck_agent/reward_values', self.reward_value_callback, 1)
        self.create_subscription(TrialINFO, '/Supervisor/Trial_INFO', self.trial_INFO_callback, 1)
        self.create_subscription(RewardID, 'Supervisor/Reward_ID', self.rewardID_callback, 1)
        self.create_subscription(ExpStatus, '/experiment/status', self.experiment_status_callback, 1)
        self.create_subscription(FrameID, '/epuck_agent/frame_ID', self.frame_ID_callback, 1)
        self.create_subscription(AEunits, '/epuck_agent/AEunits', self.AEunits_callback, 1)
        self.create_subscription(Replay, '/epuck_agent/Replay', self.Replay_callback, 1)
        self.create_subscription(EpuckMotors, 'epuck_agent/motor_cmd', self.Robot_action_callback, 1)
        self.create_subscription(TimeINFO, '/experiment/Time', self.exp_time_callback, 1)
        #self.create_subscription(Clock, '/clock', self.clock_callback, 1)

        self.data_folder_empty = False
        self.first_trial = True
        self.data_saved = False
        self.data_classes = ast.literal_eval(parameters.get('Data_gathering', 'data_classes').replace('[', '["').replace(']', '"]').replace(', ', '", "'))
        '''if parameters.get('Data_gathering', 'save_ae_model') == 'True':
                                    self.save_ae_model = True
                                else:
                                    self.save_ae_model = False'''
        self.webots_world = int(parameters.get('Experiment', 'webots_world'))

        self.sim_time_diff = 0
        self.init_sim_time = 0

        self.list_X = []
        self.list_Y = []
        self.list_Z = []
        self.list_homeo_state = []
        self.list_trial_number = []
        self.list_trial_time = []
        self.list_reward_ID = []
        self.list_frame_ID = []
        self.list_embedding = []
        self.list_n_replays = []
        self.list_reward_value = []
        self.list_robot_action = []
        self.list_retrieved_action = []

        self.AEunits = []
        
        

        self.agent_x_position = 0
        self.agent_y_position = 0
        self.agent_orientation = 0
        self.homeostatic_states = [0, 0, 0, 0]
        self.reward_values = [0, 0, 0, 0]

        self.trial_number = 1
        self.trial_time = 0.0

        self.experiment_status = 0

        self.reward_ID = 0
        self.n_replays = 0
        self.robot_action = [0, 0]
        self.retrieved_action = False

        self.create_new_folder()



    def append_data(self):
        if self.experiment_status == 1:
            self.list_X.append(self.agent_x_position)
            self.list_Y.append(self.agent_y_position)
            self.list_Z.append(self.agent_orientation)
            self.list_homeo_state.append(self.homeostatic_states)
            self.list_trial_number.append(self.trial_number)
            self.list_trial_time.append(self.trial_time)
            self.list_reward_ID.append(self.reward_ID)
            self.list_frame_ID.append(self.frame_ID)
            self.list_embedding.append(self.AEunits)
            self.list_n_replays.append(self.n_replays)
            self.list_reward_value.append(self.reward_values)
            self.list_robot_action.append(self.robot_action)
            self.list_retrieved_action.append(self.retrieved_action)


    def agent_GPS_callback(self, message):
        self.agent_x_position = message.point.x
        self.agent_y_position = message.point.y

    def agent_Compass_callback(self, message):
        self.agent_orientation = message.data

    def experiment_status_callback(self, message):
        self.experiment_status = message.status

        if self.experiment_status == 1:
            self.data_saved = False

        if self.experiment_status == 3 and self.data_saved == False:
            print('Saving trial ' + str(self.trial_number + 1))
            self.save_data()
            self.data_saved = True
            self.clean_data()

        if self.data_saved == True and self.experiment_status == 4:
            print('Experiment Completed. Data saved in ' + self.data_path)

        if self.data_saved == True and self.experiment_status == 5:
            self.first_trial = True
            self.data_saved = False
            self.agent_x_position = 0
            self.agent_y_position = 0
            self.agent_orientation = 0
            self.homeostatic_states = [0, 0, 0, 0]
            self.reward_values = [0, 0, 0, 0]
            self.trial_number = 1
            self.trial_time = 0.0
            self.experiment_status = 0
            self.reward_ID = 0
            self.n_replays = 0
            self.robot_action = [0, 0]
            self.create_new_folder()



    def frame_ID_callback(self, message):
        self.frame_ID = message.frame_id
        self.append_data()

    
    def homeostasis_callback(self, message):
        if self.webots_world == 2:
            self.homeostatic_states = [message.state_1, message.state_2]
        if self.webots_world == 3:
            self.homeostatic_states = [message.state_1, message.state_2, message.state_3, message.state_4]

    def reward_value_callback(self, message):
        if self.webots_world == 2:
            self.reward_values = [message.reward_value_1, message.reward_value_2]
        if self.webots_world == 3:
            self.reward_values = [message.reward_value_1, message.reward_value_2, message.reward_value_3, message.reward_value_4]

    def trial_INFO_callback(self, message):
        self.trial_number = message.trial

    def rewardID_callback(self, message):
        self.reward_ID = message.reward_id
        if self.reward_ID != 0:
            self.append_data()
        self.reward_ID = 0

    def AEunits_callback(self, message):
        self.AEunits = message.units

    def Replay_callback(self, message):
        self.n_replays = message.n_replays

    def Robot_action_callback(self, message):
        self.robot_action = [message.right_motor, message.left_motor]
        self.retrieved_action = message.retrieved_action

    def exp_time_callback(self, message):
        self.trial_time = round(message.time, 3)


    '''def clock_callback(self, message):
        self.clock = message.clock
        self.secs = self.clock.sec
        self.nanosecs = self.clock.nanosec/1000000000
        self.sim_time = self.secs + self.nanosecs
        self.sim_time_diff = self.sim_time - self.init_sim_time

        #---------------------  EXPERIMENT LOOP  ---------------------
        if self.sim_time_diff > .5:
            if self.data_folder_empty == False:
                self.clean_data_folder()'''
                



    def clean_data_folder(self):
        self.data_folder_empty = True
        file_list = os.listdir(self.data_folder)

        # Iterate through the files and remove them
        for file_name in file_list:
            if file_name != "LTM.csv":
                file_path = os.path.join(self.data_folder, file_name)
                os.remove(file_path)


    def create_new_folder(self):
        if self.webots_world == 0:
            self.data_folder = ws_path + '/data/Experiments/OpenArena/'
        if self.webots_world == 1:
            self.data_folder = ws_path + '/data/Experiments/LinearTrack/'
        if self.webots_world == 2:
            self.data_folder = ws_path + '/data/Experiments/Tmaze/'
        if self.webots_world == 3:
            self.data_folder = ws_path + '/data/Experiments/DoubleTmaze/'

        folders = [int(name) for name in os.listdir(self.data_folder) if os.path.isdir(os.path.join(self.data_folder, name))]

        if folders == []:
            exp_number = 1
        else:
            exp_number = max(folders) + 1
        os.makedirs(self.data_folder + '/' + str(exp_number), exist_ok=True)

        self.data_path = self.data_folder + str(exp_number) + '/data.csv'


    def save_data(self):
        csv_namefile = self.data_path
        with open(csv_namefile, mode='a') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=self.data_classes)
            if self.first_trial == True:
                csv_writer.writeheader()
                self.first_trial = False
            for i in range(len(self.list_X)):
                csv_writer.writerow({self.data_classes[0]: self.list_trial_number[i], self.data_classes[1]: self.list_trial_time[i], self.data_classes[2]: self.list_frame_ID[i], self.data_classes[3]: self.list_embedding[i], self.data_classes[4]: self.list_robot_action[i], self.data_classes[5]: self.list_retrieved_action[i], self.data_classes[6]: self.list_X[i], self.data_classes[7]: self.list_Y[i], self.data_classes[8]: self.list_Z[i], self.data_classes[9]: self.list_homeo_state[i], self.data_classes[10]: self.list_reward_ID[i], self.data_classes[11]: self.list_reward_value[i], self.data_classes[12]: self.list_n_replays[i]})


    def clean_data(self):
        self.list_X = []
        self.list_Y = []
        self.list_Z = []
        self.list_homeo_state = []
        self.list_trial_number = []
        self.list_trial_time = []
        self.list_reward_ID = []
        self.list_frame_ID = []
        self.list_embedding = []
        self.list_n_replays = []
        self.list_reward_value = []
        self.list_robot_action = []
        self.list_retrieved_action = []




def main(args=None):
    rclpy.init(args=args)
    data_gathering = Data_gathering()
    rclpy.spin(data_gathering)
    data_gathering.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()