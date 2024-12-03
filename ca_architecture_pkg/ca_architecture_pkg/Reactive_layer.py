import random
from PIL import Image


class ReactiveLayer(object):

    def __init__(self, mode, action_space, obstacle_avoidance, avoidance_dist, img_saving, n_needs):
        self.mode = mode
        self.robot_action_space = len(action_space)
        self.obstacle_avoidance_ON = obstacle_avoidance
        self.avoidance_dist = avoidance_dist
        self.img_saving = img_saving
        self.n_needs = n_needs

        self.frame_counter = 0
        self.action_interval_counter = 0
        self.wheel_speed_r, self.wheel_speed_l = 0, 0
        self.obstacle = False
        self.obstacle_right = False

        self.homeostatic_state = []
        self.reward_values = []
        self.reset_homeostasis()
        self.reward_discount = 0



    def update_prox_sens(self, ps_data):
        self.prox_sensors = [ps_data[0], ps_data[1], ps_data[2],ps_data[5], ps_data[6], ps_data[7]]

        self.right_prox_sensors = [ps_data[0], ps_data[1], ps_data[2]]
        self.left_prox_sensors = [ps_data[5], ps_data[6], ps_data[7]]


    def get_visual_observation(self, ws_path, frame, env):
        if env == 0:
            path = '/data/OpenArena/'
        if env == 1:
            path = '/data/LinearTrack/'
        if env == 2:
            path = '/data/Tmaze/'
        if env == 3:
            path = '/data/DoubleTmaze/'

        path = ws_path + path

        self.frame = frame

        self.frame_counter += 1
        self.frame_ID = str(self.frame_counter)
        while len(self.frame_ID) < 5:
            self.frame_ID = '0' + self.frame_ID

        byte_data = self.frame.tobytes()
        width, height = 160, 120

        img = Image.frombytes('RGBA', (width, height), byte_data)

        r, g, b, a = img.split()
        img = Image.merge('RGBA', (b, g, r, a))
        self.RGB_img = img.convert('RGB')

        if self.img_saving == True:
            self.RGB_img.save(path + self.frame_ID + '.jpg')

        return self.frame_counter, self.RGB_img


    def detect_obstacle(self):
        if any(sensor < self.avoidance_dist  for sensor in self.prox_sensors):
            self.obstacle = True
            if sum(self.right_prox_sensors) > sum(self.left_prox_sensors):
                self.obstacle_right = True
            else:
                self.obstacle_right = False

        else:
            self.obstacle = False


    def obstacle_avoidance(self):

        if self.obstacle == True:
            action = random.randrange(0,3)
            if self.obstacle_right == False:
                action += 6
        return(action)


    def continuous_random_action(self):
        if self.obstacle_avoidance_ON == True:
            self.detect_obstacle()

        if self.obstacle == True:
            self.obstacle_avoidance()

        else:
            self.action_interval_counter = 0
            self.wheel_speed_r = self.wheel_speed_r + random.gauss(0,1)
            self.wheel_speed_l = self.wheel_speed_l + random.gauss(0,1)

            if self.wheel_speed_r < 3: self.wheel_speed_r = 3
            if self.wheel_speed_r > self.robot_action_space[0]: self.wheel_speed_r = self.robot_action_space[0]
            if self.wheel_speed_l < 3: self.wheel_speed_l = 3
            if self.wheel_speed_l > self.robot_action_space[0]: self.wheel_speed_l = self.robot_action_space[0]

        return([self.wheel_speed_r, self.wheel_speed_l])


    def discrete_random_action(self):
        if self.obstacle_avoidance_ON == True:
            self.detect_obstacle()

        if self.obstacle == True:
            action = self.obstacle_avoidance()

        else:
            action = random.randrange(self.robot_action_space)

        return(action)
        


    def random_action(self):
        if self.mode == 0: 
            output_action = self.continuous_random_action()
        else:
            action = self.discrete_random_action()

        return action



    def homeostasis(self, homeo_discount):
        discount_factor = homeo_discount

        for i in range(len(self.homeostatic_state)):
            if self.homeostatic_state[i] > 0:
                self.homeostatic_state[i] = round(self.homeostatic_state[i] - discount_factor, len(str(discount_factor)))
            if self.homeostatic_state[i] <= 0:
                self.homeostatic_state[i] = 0.0001

        return self.homeostatic_state


    def reward_value(self, homeostatic_state, homeo_discount):
        self.reward_discount += homeo_discount*2

        homeostatic_error = [1-x for x in homeostatic_state]
        reward_values = [x-self.reward_discount for x in homeostatic_error]
        reward_values = [0.0001 if x < 0 else x for x in reward_values]

        return reward_values


    def reset_homeostasis(self):
        self.homeostatic_state = []
        self.reward_values = []
        self.reward_discount = 0
        for i in range(self.n_needs-1):
            self.homeostatic_state.append(1)
        #self.homeostatic_state.append(random.uniform(0, 1))
        self.homeostatic_state.append(0.5)
        random.shuffle(self.homeostatic_state)

        for i in range(self.n_needs):
            if self.homeostatic_state[i] > 0.5:
                self.reward_values.append(0)
            else:
                self.reward_values.append(1)