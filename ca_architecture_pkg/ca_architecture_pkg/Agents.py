import sys
from Reactive_layer import ReactiveLayer
from Adaptive_layer import AdaptiveLayer
from Contextual_layer import ContextualLayer


class ReactiveAgent(object):
    def __init__(self, mode, action_space, obstacle_avoidance, avoidance_dist, img_saving, n_needs):
        self.Reactive_Layer = ReactiveLayer(mode=mode, action_space=action_space, obstacle_avoidance=obstacle_avoidance, avoidance_dist=avoidance_dist, img_saving=img_saving, n_needs=n_needs)
        

    def update_prox_sensors(self, ps_data):
        self.Reactive_Layer.update_prox_sens(ps_data)

    def get_visual_observation(self, ws_path, frame, env):
        frame_ID, RGB_img = self.Reactive_Layer.get_visual_observation(ws_path, frame, env)
        return frame_ID, RGB_img

    def random_action(self):
        reactive_action = self.Reactive_Layer.random_action()
        return reactive_action
        
    def sample_env(self, num_pictures, cam_interval):
        name_picture, picture_number = self.Reactive_Layer.camera_capture(num_pictures, cam_interval)
        return name_picture, picture_number

    def homeostasis(self, homeo_discount):
        homeostatic_state = self.Reactive_Layer.homeostasis(homeo_discount)
        return homeostatic_state

    def reward_value(self, homeostatic_state, reward_discount):
        reward_values = self.Reactive_Layer.reward_value(homeostatic_state, reward_discount)
        return reward_values

    def reset_homeostasis(self):
        self.Reactive_Layer.reset_homeostasis()




class AdaptiveAgent(ReactiveAgent):
    def __init__(self, mode, action_space, obstacle_avoidance, avoidance_dist, img_saving, n_needs):
        super().__init__(mode, action_space, obstacle_avoidance, avoidance_dist, img_saving, n_needs)
        self.Adaptive_Layer = AdaptiveLayer()


    def load_AE_model(self, model, n_hidden, motivational_AE, motiv_I_len):
        self.Adaptive_Layer.load_AE_model(model, n_hidden, motivational_AE, motiv_I_len)

    def encode_frame(self, frame, motivational_AE, homeo_state, motiv_I_len_per_need, STM_limit):
        embedding, activation, view_prediction = self.Adaptive_Layer.encode(frame, motivational_AE, homeo_state, motiv_I_len_per_need, STM_limit)
        return embedding, activation, view_prediction

    def retrain(self, motivational_AE, learning_rate, alpha, C_factor, num_epochs, batch_size):
        self.Adaptive_Layer.retrain(motivational_AE, learning_rate, alpha, C_factor, num_epochs, batch_size)

    def save_model(self, model_path):
        self.Adaptive_Layer.save_model(model_path)



class ContextualAgent(AdaptiveAgent):
    def __init__(self, mode, webots_world, action_space, obstacle_avoidance, avoidance_dist, img_saving, n_needs, STM_limit, LTM_limit, n_hidden, alpha_tr, coll_thres_abs, coll_thres_prop, tau_decay, load_LTM, ws_path):
        super().__init__(mode, action_space, obstacle_avoidance, avoidance_dist, img_saving, n_needs)
        self.Contextual_Layer = ContextualLayer(webots_world, action_space, STM_limit, LTM_limit, n_hidden, alpha_tr, coll_thres_abs, coll_thres_prop, tau_decay, load_LTM, ws_path)


    def check_state_similarity(self, AEunits):
        retrieved_action = self.Contextual_Layer.check_state_similarity(AEunits)
        return retrieved_action

    def update_STM(self, AEunits, action):
        self.Contextual_Layer.update_STM(AEunits, action)
        self.Contextual_Layer.update_sequential_bias()

    def reset_STM(self):
        self.Adaptive_Layer.trial_frame_list = []
        self.Adaptive_Layer.trial_homeostate_list = []
        self.Contextual_Layer.reset_STM()

    def update_LTM(self, reward_value):
        self.Contextual_Layer.update_LTM(reward_value)

    def save_LTM(self, exp_data_path):
        self.Contextual_Layer.save_LTM(exp_data_path)
