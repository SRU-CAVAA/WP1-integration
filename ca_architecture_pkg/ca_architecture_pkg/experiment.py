import os
import sys
import ast
import time
import rclpy
import random
import numpy as np
from rclpy.node import Node
from std_msgs.msg import String
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Range, Image
from webots_ros2_msgs.msg import FloatStamped
from ca_msgs_pkg.msg import ExpStatus, EpuckMotors, HomeostaticStates, FrameID, AEunits, RewardID, RewardValues, Replay, TimeINFO


from pathlib import Path
ws_path = str(Path(__file__).parents[6])
print(ws_path)

sys.path.append(ws_path + '/src/ca_architecture_pkg/ca_architecture_pkg/')
from Agents import ReactiveAgent, AdaptiveAgent, ContextualAgent

import configparser
parameters = configparser.ConfigParser()
parameters.read(ws_path + '/src/config.ini')


# 1. Experiment class constructor. Variable definition using config.ini file.
#   - Reactive, Adaptive and Contextual layers
#   - Build the Agent
#   - ROS Publisher and subscribers
#
# 2. ROS Callback functions
#
# 3. Experiment loop


class Experiment(Node):
    def __init__(self):
        super().__init__('experiment')
        self.get_logger().info('-------------- Experiment --------------')
        print()

        self.webots_world = int(parameters.get('Experiment', 'webots_world'))
        self.agent_mode = int(parameters.get('Agent', 'agent_mode'))
        self.exp_batch_size = int(parameters.get('Experiment', 'exp_batch_size'))
        self.experiment_status = 0
        self.frame_ID = 0
        self.reward_ID = 0
        self.start_time = 0
        self.sim_time = 0
        self.init_action_time = 0
        self.action = 4
        self.retrain_epochs = 0
        self.contextual_action = -1
        self.n_needs = 0
        self.retraining = False
        self.trial_initiated = False
        self.training_set = 1

        self.AEunit_max = 30000

#----------------- REACTIVE LAYER PAPRAMETERS -----------------
        self.state_action_rate = float(parameters.get('Reactive_Layer', 'state_action_rate'))
        if parameters.get('Reactive_Layer', 'obstacle_avoidance') == 'True':
            self.obstacle_avoidance = True
        else:
            self.obstacle_avoidance = False
        self.avoidance_dist = float(parameters.get('Reactive_Layer', 'avoidance_dist'))

        if parameters.get('Reactive_Layer', 'homeostasis') == 'True':
            if self.webots_world == 0 or self.webots_world == 1: self.n_needs = 1
            if self.webots_world == 2: self.n_needs = 2
            if self.webots_world == 3: self.n_needs = 4
            self.homeostasis = True
            self.homeo_discount = float(parameters.get('Reactive_Layer', 'homeo_discount'))
            self.homeo_state = []
        else:
            self.homeostasis = False

        if parameters.get('Reactive_Layer', 'img_saving') == 'True':
            self.img_saving = True
        else:
            self.img_saving = False

        self.frame_count = 0

        self.ps0, self.ps1, self.ps2, self.ps3, self.ps4, self.ps5, self.ps6, self.ps7 = 0,0,0,0,0,0,0,0
        self.RGB_img = 0
        self.frame = []

        self.reactive_layer_mode = int(parameters.get('Reactive_Layer', 'reactive_layer_mode'))

        if self.reactive_layer_mode == 0:
            self.reactive_layer_action_space = ast.literal_eval(parameters.get('Reactive_Layer', 'action_space_mode_0'))
        if self.reactive_layer_mode == 1:
            self.reactive_layer_action_space = ast.literal_eval(parameters.get('Reactive_Layer', 'action_space_mode_1'))
        if self.reactive_layer_mode == 2:
            self.reactive_layer_action_space = ast.literal_eval(parameters.get('Reactive_Layer', 'action_space_mode_2'))
        if self.reactive_layer_mode == 3:
            self.reactive_layer_action_space = ast.literal_eval(parameters.get('Reactive_Layer', 'action_space_mode_3'))
        if self.reactive_layer_mode == 4:
            self.reactive_layer_action_space = ast.literal_eval(parameters.get('Reactive_Layer', 'action_space_mode_4'))


#----------------- ADAPTIVE LAYER PAPRAMETERS -----------------
        self.embedding = []
        self.AE_prediction = []

        if parameters.get('Adaptive_Layer', 'motivational_autoencoder') == 'True':
            self.motivational_AE = True
        else:
            self.motivational_AE = False
        if parameters.get('Adaptive_Layer', 'AE_retrain') == 'True':
            self.AE_retrain = True
        else:
            self.AE_retrain = False

        model_list = ast.literal_eval(parameters.get('Adaptive_Layer', 'ae_models').replace('[', '["').replace(']', '"]').replace(', ', '", "'))
        n_hidden_list = ast.literal_eval(parameters.get('Adaptive_Layer', 'n_hidden').replace('[', '["').replace(']', '"]').replace(', ', '", "'))
        self.batch_size = int(parameters.get('Adaptive_Layer', 'batch_size'))
        self.C_factor = int(parameters.get('Adaptive_Layer', 'C_factor'))
        self.learning_rate = float(parameters.get('Adaptive_Layer', 'learning_rate'))
        self.alpha = float(parameters.get('Adaptive_Layer', 'alpha'))
        self.motiv_I_len_per_need = int(parameters.get('Adaptive_Layer', 'motivational_input_lenght'))
        self.motiv_I_len = self.motiv_I_len_per_need * self.n_needs


        if self.webots_world == 0:
            self.n_hidden = int(n_hidden_list[0])
            if self.motivational_AE: self.model_filename = ws_path + '/AE_models/' + model_list[0]
            else:self.model_filename = ws_path + '/AE_models/' + model_list[0]

        if self.webots_world == 1:
            self.n_hidden = int(n_hidden_list[1])
            if self.motivational_AE: self.model_filename = ws_path + '/AE_models/' + model_list[0]
            else:self.model_filename = ws_path + '/AE_models/' + model_list[0]

        if self.webots_world == 2:
            self.n_hidden = int(n_hidden_list[2])
            if self.motivational_AE: self.model_filename = ws_path + '/AE_models/' + model_list[1]
            else:self.model_filename = ws_path + '/AE_models/' + model_list[2]

        if self.webots_world == 3:
            self.n_hidden = int(n_hidden_list[3])
            if self.motivational_AE: self.model_filename = ws_path + '/AE_models/' + model_list[3]
            else:self.model_filename = ws_path + '/AE_models/' + model_list[4]
        


#----------------- CONTEXTUAL LAYER PAPRAMETERS -----------------
        self.STM_limit = int(parameters.get('Contextual_Layer', 'STM_limit'))
        self.LTM_limit = int(parameters.get('Contextual_Layer', 'LTM_limit'))
        self.alpha_tr = float(parameters.get('Contextual_Layer', 'alpha_tr'))
        self.coll_thres_abs = float(parameters.get('Contextual_Layer', 'coll_thres_abs'))
        self.coll_thres_prop = float(parameters.get('Contextual_Layer', 'coll_thres_prop'))
        self.tau_decay = float(parameters.get('Contextual_Layer', 'tau_decay'))
        if parameters.get('Contextual_Layer', 'load_LTM') == 'True':
            self.load_LTM = True
        else:
            self.load_LTM = False

        
#----------------- LOAD AGENT MODE -----------------
        if self.agent_mode == 1:
            self.agent = ReactiveAgent(mode=self.reactive_layer_mode, action_space=self.reactive_layer_action_space, obstacle_avoidance=self.obstacle_avoidance, avoidance_dist=self.avoidance_dist, img_saving=self.img_saving, n_needs=self.n_needs)
        
        if self.agent_mode == 2:
            self.agent = AdaptiveAgent(mode=self.reactive_layer_mode, action_space=self.reactive_layer_action_space, obstacle_avoidance=self.obstacle_avoidance, avoidance_dist=self.avoidance_dist, img_saving=self.img_saving, n_needs=self.n_needs)
            self.agent.load_AE_model(model=self.model_filename, n_hidden=self.n_hidden, motivational_AE=self.motivational_AE, motiv_I_len=self.motiv_I_len)

        
        if self.agent_mode == 3:
            self.agent = ContextualAgent(mode=self.reactive_layer_mode, webots_world=self.webots_world, action_space=self.reactive_layer_action_space, obstacle_avoidance=self.obstacle_avoidance, avoidance_dist=self.avoidance_dist, img_saving=self.img_saving, n_needs=self.n_needs,
                STM_limit=self.STM_limit, LTM_limit=self.LTM_limit, n_hidden=self.n_hidden, alpha_tr=self.alpha_tr , coll_thres_abs=self.coll_thres_abs , coll_thres_prop=self.coll_thres_prop , tau_decay=self.tau_decay, load_LTM=self.load_LTM, ws_path=ws_path)
            self.agent.load_AE_model(model=self.model_filename, n_hidden=self.n_hidden, motivational_AE=self.motivational_AE, motiv_I_len=self.motiv_I_len)
            

#----------------- Experiment is ready -----------------
        print()
        self.get_logger().info('       ENVIRONMENT ----> ' + str(self.webots_world))
        self.get_logger().info('        AGENT MODE ----> ' + str(self.agent_mode))
        self.get_logger().info('OBSTACLE AVOIDANCE ----> ' + str(self.obstacle_avoidance))
        if self.agent_mode > 1: self.get_logger().info('AUTOENCODER VERSION ---> ' + self.model_filename)
        print()
        print()


        self.find_data_folder() #Data_gathering node has create a new folder.
        
            
#----------------- ROS SUBSCRIPTIONS AND PUBLICATIONS -----------------
        #self.Experiment__publisher = self.create_publisher(ExpStatus, 'experiment/status', 1)
        self.Motor__publisher = self.create_publisher(EpuckMotors, 'epuck_agent/motor_cmd', 1)
        self.Homeostasis__publisher = self.create_publisher(HomeostaticStates, 'epuck_agent/homeostatic_states', 1)
        self.RewardValue__publisher = self.create_publisher(RewardValues, 'epuck_agent/reward_values', 1)
        self.Frame__publisher = self.create_publisher(FrameID, 'epuck_agent/frame_ID', 1)
        self.AEunits__publisher = self.create_publisher(AEunits, 'epuck_agent/AEunits', 1)
        self.Replay__publisher = self.create_publisher(Replay, 'epuck_agent/Replay', 1)
        self.Time__publisher = self.create_publisher(TimeINFO, 'experiment/Time', 1)
        self.pub_exp_status = self.create_publisher(ExpStatus, 'experiment/status', 1)
        
        self.create_subscription(Range, 'epuck_agent/ps0', self.__ps0_sensor_callback, 1)
        self.create_subscription(Range, 'epuck_agent/ps1', self.__ps1_sensor_callback, 1)
        self.create_subscription(Range, 'epuck_agent/ps2', self.__ps2_sensor_callback, 1)
        self.create_subscription(Range, 'epuck_agent/ps3', self.__ps3_sensor_callback, 1)
        self.create_subscription(Range, 'epuck_agent/ps4', self.__ps4_sensor_callback, 1)
        self.create_subscription(Range, 'epuck_agent/ps5', self.__ps5_sensor_callback, 1)
        self.create_subscription(Range, 'epuck_agent/ps6', self.__ps6_sensor_callback, 1)
        self.create_subscription(Range, 'epuck_agent/ps7', self.__ps7_sensor_callback, 1)

        self.create_subscription(Image, 'epuck_agent/camera/image_color', self.__camera_callback, 1)
        self.create_subscription(ExpStatus, '/experiment/status', self.experiment_status_callback, 1)
        self.create_subscription(RewardID, 'Supervisor/Reward_ID', self.rewardID_callback, 1)
        self.create_subscription(String, '/Supervisor/Reset', self.reset_trial_callback, 1)
        self.create_subscription(Clock, '/clock', self.clock_callback, 1)



#----------------- ROS CALLBACK FUNCTIONS -----------------        
    def __ps0_sensor_callback(self, message):
        self.ps0 = message.range
    def __ps1_sensor_callback(self, message):
        self.ps1 = message.range
    def __ps2_sensor_callback(self, message):
        self.ps2 = message.range
    def __ps3_sensor_callback(self, message):
        self.ps3 = message.range
    def __ps4_sensor_callback(self, message):
        self.ps4 = message.range
    def __ps5_sensor_callback(self, message):
        self.ps5 = message.range
    def __ps6_sensor_callback(self, message):
        self.ps6 = message.range
    def __ps7_sensor_callback(self, message):
        self.ps7 = message.range


    def __camera_callback(self, message):
        self.frame = message.data


    def experiment_status_callback(self, message):
        self.experiment_status = message.status

    def rewardID_callback(self, message):
        self.reward_ID = message.reward_id

    def reset_trial_callback(self, message):
        pass

    def experiment_time(self, time):
        self.sim_time = self.secs + self.nanosecs - self.start_time #Current simulation time in secs. Every trial it starts at 0.
        self.action_time = self.sim_time #Current simulation time used for computing action init time and action time difference
        self.action_time_diff = self.action_time - self.init_action_time #Used to set the action rate.

        if self.experiment_status == 0:
            self.start_time = self.secs + self.nanosecs
            self.action_time = 0
            self.init_action_time = 0

        msg = TimeINFO()
        msg.time = self.sim_time
        self.Time__publisher.publish(msg)
            

    def clock_callback(self, message):
        self.clock = message.clock
        self.secs = self.clock.sec
        self.nanosecs = self.clock.nanosec/1000000000
        self.time = self.secs + self.nanosecs
        self.experiment_time(self.time)

        #---------------------  EXPERIMENT LOOP  ---------------------

        # Experiment Status 0: Waiting to initiate the trial
        if self.experiment_status == 0:
            self.retraining = False
            self.agent.reset_homeostasis()

        
        self.ros_finish_experiment()
        self.batch_training()


        #---------------------  STATE-ACTION LOOP  ---------------------
        if self.action_time_diff > self.state_action_rate:
            self.init_action_time = self.action_time

            # Experiment Status 1: Initiate the trial until reward is found
            if self.experiment_status == 1:
                #----------------- REACTIVE CONTROL -----------------
                if self.homeostasis == True:
                    self.homeo_state = self.agent.homeostasis(self.homeo_discount)
                    self.reward_values = self.agent.reward_value(self.homeo_state, self.homeo_discount)

                    self.ros_robot_homeostasis(self.homeo_state, self.reward_values)

                self.ps_data = [self.ps0, self.ps1, self.ps2, self.ps3, self.ps4, self.ps5, self.ps6, self.ps7]
                self.ros_update_robot_prox_sensors()
                self.frame_ID, self.RGB_img = self.agent.get_visual_observation(ws_path, self.frame, self.webots_world)
                self.ros_visual_observation()
                self.reactive_action = self.agent.random_action()

                if self.trial_initiated == False:
                    if self.agent_mode != 1:
                        self.embedding, self.AE_activation, self.AE_prediction = self.agent.encode_frame(self.RGB_img, self.motivational_AE, self.homeo_state, self.motiv_I_len_per_need, self.STM_limit)
                        self.ros_robot_autoencoder(self.embedding, self.AE_activation, self.AE_prediction)
                    self.trial_initiated = True
                
                
                #NOTE: TESTING adaptive control before contextual ------------------------------------------------------------------------------------------------------------------------------

                #----------------- ADAPTIVE CONTROL -----------------
                if self.agent_mode != 1:
                    self.embedding, self.AE_activation, self.AE_prediction = self.agent.encode_frame(self.RGB_img, self.motivational_AE, self.homeo_state, self.motiv_I_len_per_need, self.STM_limit)
                    self.ros_robot_autoencoder(self.embedding, self.AE_activation, self.AE_prediction)

                #----------------- CONTEXTUAL CONTROL -----------------
                if self.agent_mode == 3:
                    self.contextual_action = self.agent.check_state_similarity(self.embedding)
                    #self.agent.update_STM(self.embedding, self.action)
                    self.ros_move_robot(self.reactive_action, self.contextual_action)
                    self.agent.update_STM(self.embedding, self.action)
                else:
                    self.ros_move_robot(self.reactive_action)


                



            # Experiment Status 2: End of trial - Reward information and Replay
            if self.experiment_status == 2 and self.retraining == False:
                self.ros_move_robot(-1)
                self.retraining = True
                
                if self.agent_mode == 3:
                    if self.AE_retrain == True:
                        self.AE_model_retrain()
                    
                    if self.reward_ID != 0:
                        #self.reward_value = 0.99
                        if self.webots_world > 1: self.reward_value = self.reward_values[self.reward_ID - 2]
                        else: self.reward_value = self.reward_values[self.reward_ID - 1]
                    else:
                        self.reward_value = 0

                    if self.reward_value > 0.0001: 
                        self.agent.update_LTM(self.reward_value)
                    self.agent.reset_STM()
                    
                    self.trial_resume()

                msg = Replay()
                msg.n_replays = self.retrain_epochs
                self.Replay__publisher.publish(msg)

                self.reward_ID = 0
                
                




#----------------- REACTIVE LAYER COMMUNICATION FUNCTIONS ----------------- 
    def ros_update_robot_prox_sensors(self):
        self.agent.update_prox_sensors(self.ps_data)


    def ros_move_robot(self, reactive_action, contextual_action=-1):

        if contextual_action == -1:
            self.action = reactive_action
            wheel_r = self.reactive_layer_action_space[reactive_action][0]
            wheel_l = self.reactive_layer_action_space[reactive_action][1]

        else:
            print('Retrieved action = ', contextual_action)
            print()
            self.action = contextual_action
            wheel_r = self.reactive_layer_action_space[contextual_action][0]
            wheel_l = self.reactive_layer_action_space[contextual_action][1]

        if reactive_action == -1:
            wheel_r = 0.0
            wheel_l = 0.0

        command_message = EpuckMotors()
        command_message.right_motor = wheel_r
        command_message.left_motor = wheel_l

        self.Motor__publisher.publish(command_message)


    def ros_robot_homeostasis(self, homeo_state, reward_values):
        homeo_msg = HomeostaticStates()
        reward_msg = RewardValues()
        
        homeo_msg.state_1= self.homeo_state[0]
        reward_msg.reward_value_1 = self.reward_values[0]

        if self.webots_world >= 2:
            homeo_msg.state_2 = self.homeo_state[1]
            reward_msg.reward_value_2 = self.reward_values[1]

        if self.webots_world == 3:
            homeo_msg.state_3 , homeo_msg.state_4 = self.homeo_state[2], self.homeo_state[3]
            reward_msg.reward_value_3 , reward_msg.reward_value_4 = self.reward_values[2], self.reward_values[3]

        self.Homeostasis__publisher.publish(homeo_msg)
        self.RewardValue__publisher.publish(reward_msg)

    def ros_visual_observation(self):
        if self.frame != []:
            msg = FrameID()
            msg.frame_id = self.frame_ID
            self.Frame__publisher.publish(msg)





#----------------- ADAPTIVE LAYER COMMUNICATION FUNCTIONS -----------------
    def ros_robot_autoencoder(self, embedding, AE_activation, AE_prediction):
        if self.RGB_img != 0:
            embedding = embedding# / self.AEunit_max
            
            msg = AEunits()
            msg.units = embedding
            msg.prediction = AE_prediction.flatten()
            self.AEunits__publisher.publish(msg)

    def AE_model_retrain(self):
        self.retraining = True
        self.retrain_epochs = 10
        self.agent.retrain(self.motivational_AE, self.learning_rate, self.alpha, self.C_factor, self.retrain_epochs, self.batch_size)

        if self.model_filename[-14:] != "_retrained.pth":
            self.model_filename = self.model_filename[:-4] + "_retrained.pth"
        self.agent.save_model(self.model_filename)
        self.agent.load_AE_model(model=self.model_filename, n_hidden=self.n_hidden, motivational_AE=self.motivational_AE, motiv_I_len=self.motiv_I_len)





#----------------- EXPERIMENT CONTROL FUNCTIONS -----------------
    def find_data_folder(self):
        if self.webots_world == 0:
            self.data_folder = ws_path + '/data/Experiments/OpenArena/'
        if self.webots_world == 1:
            self.data_folder = ws_path + '/data/Experiments/LinearTrack/'
        if self.webots_world == 2:
            self.data_folder = ws_path + '/data/Experiments/Tmaze/'
        if self.webots_world == 3:
            self.data_folder = ws_path + '/data/Experiments/DoubleTmaze/'

        folders = [int(name) for name in os.listdir(self.data_folder) if os.path.isdir(os.path.join(self.data_folder, name))]
        exp_number = max(folders)

        self.exp_data_path = self.data_folder + str(exp_number)


    def ros_finish_experiment(self):
        if self.experiment_status == 4:
            command_message = EpuckMotors()
            command_message.right_motor = 0.0
            command_message.left_motor = 0.0
            self.Motor__publisher.publish(command_message)

    def update_experiment_status(self, exp_status):
        self.experiment_status = exp_status
        msg = ExpStatus()
        msg.status = self.experiment_status
        self.pub_exp_status.publish(msg)


    def batch_training(self):
        if self.experiment_status == 4 and self.training_set <= self.exp_batch_size:
            print("EXPERIMENT BATCH " + str(self.training_set) + "/" + str(self.exp_batch_size) + " FINISHED")
            self.training_set += 1
            self.ros_finish_experiment()
            self.find_data_folder()
            self.agent.save_LTM(self.exp_data_path)
            if self.training_set <= self.exp_batch_size:
                self.update_experiment_status(5)
            


        


    def trial_resume(self):
        print()
        print("------- TRIAL RESUME -------")
        print("Trial duration =", self.sim_time)
        print("Motivational state =", self.homeo_state.index(min(self.homeo_state)) + 1)
        print("Reward ID =", self.reward_ID)
        print("Reward value =", self.reward_value)
        print("------- ------- ------- ------- ------- -------")
        print()



def main(args=None):
    rclpy.init(args=args)
    experiment = Experiment()
    rclpy.spin(experiment)
    experiment.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()