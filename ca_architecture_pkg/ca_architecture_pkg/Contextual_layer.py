import csv
import numpy as np
from scipy.spatial.distance import euclidean

class ContextualLayer(object):

    def __init__(self, webots_world, action_space, STM_limit, LTM_limit, n_hidden, alpha_tr, coll_thres_abs, coll_thres_prop, tau_decay, load_LTM, ws_path, sequential_bias=True):
        self.webots_world = webots_world
        self.action_space = len(action_space)
        self.sequential_bias = sequential_bias
        self.STM_limit = STM_limit
        self.LTM_limit = LTM_limit
        self.n_hidden = n_hidden
        self.alpha_tr = alpha_tr
        self.coll_thres_abs = coll_thres_abs
        self.coll_thres_prop = coll_thres_prop
        self.tau_decay = tau_decay

        self.STM = [[[0] * self.n_hidden , 0] for i in range(self.STM_limit)]
        self.LTM = [[],[],[]] # state, action, reward
        self.triggers = []
        self.selected_states_indx = []
        self.LTM_loaded = False
        self.LTM_saved = False
        self.ws_path = ws_path

        self.max_collectors = [] #NOTE: To remove

        if load_LTM == True:
            self.LTM = self.load_LTM()


    def euclidean_similarity(self, array1, array2):
        # Only consider positions where either array has non-zero values
        mask = (array1 > 0) | (array2 > 0)
        array1_filtered = array1[mask]
        array2_filtered = array2[mask]
        return 1 / (1 + euclidean(array1_filtered, array2_filtered))


    def check_state_similarity(self, AEunits):
        #Collector stores the similarity of each state in memory with respect the current state.
        similarity_measure = 0
        if len(self.LTM[0]) > 0:
            # np.abs(AEunits - np.array(self.LTM[0])) = Compares unit by unit the current embedding to each embedding stored in LTM. Shape is n_Sequences x lenght_Sequences x units_in_enbedding
            # np.sum(np.abs(AEunits - np.array(self.LTM[0])), axis=2) = Sum unit differences for comparing current and stored embeddings. Shape is n_Sequences x lenght_Sequences
            # (np.sum(np.abs(AEunits - np.array(self.LTM[0])), axis=2)) / len(AEunits) = Normalize by the number of units in embedding. Not clear why.

            if similarity_measure == 0:
                self.collectors = (1 - (np.sum(np.abs(AEunits - np.array(self.LTM[0])), axis=2)) / len(AEunits))
            else:
                #New similarity measure
                self.collectors = []
                for experiment_states in self.LTM[0]:
                    experiment_similarities = [
                        self.euclidean_similarity(AEunits, stored_state) for stored_state in experiment_states
                    ]
                    self.collectors.append(experiment_similarities)
            



            if self.sequential_bias:
                bias = np.array(self.triggers)
                #print("bias = ", bias)
                #print("bias.shape = ", bias.shape)
                #print("self.collectors = ", self.collectors)
                
                self.collectors *= bias
                #print("self.collectors = ", self.collectors)
                #print()
                #print()

            retrieved_action = self.memory_retrieval()
        else: retrieved_action = -1

        return retrieved_action

    def memory_retrieval(self):
        #From the collector, only those states that surpass an absolute and proportional threshold are retrieved and contribute to action.
         #####print('max(self.collectors) =', max(self.collectors.flatten()))
         #####self.max_collectors.append(max(self.collectors.flatten()))
         #####print('Mean of max collector = ' +  str(np.mean(self.max_collectors)) + ', for ' + str(len(self.max_collectors)) + ' collectors')
        self.selected_states_indx = (self.collectors > self.coll_thres_abs) & ((self.collectors/self.collectors.max()) > self.coll_thres_prop)
        
         #####count_true = sum(sum(row) for row in self.selected_states_indx)
         #####print("count_true =", count_true)

        # Reduce the collector to those states that surpasses the similarity thresholds.
        self.collectors = self.collectors[self.selected_states_indx]
        

        if np.any(self.selected_states_indx):
            # Given the state you are, consider similar states and choose actions based on those you took previously and the reward they returned
            actions = np.array(self.LTM[1])[self.selected_states_indx]
            #np.nonzero returns two array (idx of sequences and idx of states)
            rewards = np.array(self.LTM[2])[(np.nonzero(self.selected_states_indx)[0])] #Get the reward obtained in those sequences where the selected states pertains to
            rewards = rewards/rewards.max()

            #distances (normalized) are based on the action indexes within the sequence and their proxemity to reward
            distances = (self.STM_limit - np.nonzero(self.selected_states_indx)[1])/self.STM_limit

            # Probability matrix: define a probability for each retrieved action from memory based on its result in reward 
            prob_matrix = np.zeros((len(actions), self.action_space))
        
            #np.exp() = e^x, where e is Euler's number (2.71828) NOTE: Why?
            #prob_matrix gives probabilities for actions based on distance to reward and reward value.
            prob_matrix[np.arange(len(actions)), actions[:]] = self.collectors*(rewards*np.exp(-distances/self.tau_decay))


            prob_matrix = np.sum(prob_matrix, axis=0)
            prob_matrix = prob_matrix/prob_matrix.sum()  #proportion of being selected based on the action's relative reward based on the stored experiences
            q = prob_matrix.flatten()
            #print('max(self.collectors) =', max(self.collectors.flatten()))
            #print('max(q) =', max(q.flatten()))


            #Select action based on policy
            retrieved_action = np.random.choice(np.arange(q.shape[0]), p=q)

        else:
            retrieved_action = -1

        return retrieved_action


    def update_sequential_bias(self): #NOTE: understand this function
        self.last_actions_indx = np.copy(self.selected_states_indx).tolist()  # Updates the last action indexes with the current actions indexes.

        # Update trigger values.
        if (len(self.triggers) > 0) and self.sequential_bias:
            self.triggers = (np.array(self.triggers) * (1. - self.alpha_tr)) + self.alpha_tr # trigger values decay to 1 by default
            self.triggers[(self.triggers < 1.)] = 1.       # all trigger values below 1 are reset to 1.
            tr_last_actions_indx = np.array(self.last_actions_indx)
            self.triggers[tr_last_actions_indx] = 1.    # NEW: the trigger value of previously selected segments are reset to 1!!!
            last_actions_shifted = np.roll(self.last_actions_indx, 1, axis=1) # shift the matrix one step to the right
            last_actions_shifted[:, 0] = False  # set the first value of each sequence to False

            # Increase ONLY the trigger value of the next element in sequence (after the ones selected before)!
            tr_change_indx = np.array(last_actions_shifted)
            self.triggers[tr_change_indx] += 0.01    # NEW: increase by an arbitrary amount (this amount should be tuned or modified).
            self.triggers = self.triggers.tolist()




    def update_STM(self, AEunits, action):
        if AEunits != []:
            couplet = [AEunits, action]
            self.STM.append(couplet)
            self.STM.pop(0)



    def update_LTM(self, reward_value):
        states = [s[0] for s in self.STM]
        actions = [a[1] for a in self.STM]
        self.LTM[0].append(np.array(states))
        self.LTM[1].append(np.array(actions))
        if reward_value < 0.000001: reward_value = 0.000001 #For fail trials
        self.LTM[2].append(reward_value)

        self.triggers.append(np.ones(self.STM_limit).tolist()) #Initiate sequential bias as an identity matrix

        #FIFO forgetting rule
        if len(self.LTM[0]) > self.LTM_limit:
            self.LTM[0].pop(0)
            self.LTM[1].pop(0)
            self.LTM[2].pop(0)
            self.triggers = np.delete(np.array(self.triggers),0,0).tolist()
            self.selected_states_indx = np.delete(np.array(self.selected_states_indx),0,0).tolist()
            self.last_actions_indx = np.delete(np.array(self.last_actions_indx),0,0).tolist()


        print('Sequences stored in LTM = ' + str(len(self.LTM[0])))
        print('Rewards stored in LTM = ' + str(self.LTM[2]))

        

    def reset_STM(self):
        print('Number of State-Action couplets = ' + str(len(self.STM)))
        self.STM = [[[0] * self.n_hidden , 0] for i in range(self.STM_limit)]

    def save_LTM(self, exp_data_path):
        csv_namefile = exp_data_path + '/LTM.csv'

        with open(csv_namefile, mode='w') as csv_file:
            fieldnames = ['Sequence', 'State', 'Action', 'Reward', 'Seq_bias']
            csv_writer = csv.DictWriter(csv_file, fieldnames= fieldnames)
            csv_writer.writeheader()
            for i in range(len(self.LTM[0])):
                for j in range(len(self.LTM[0][i])):
                    state_str = ','.join(map(str, self.LTM[0][i][j]))
                    csv_writer.writerow({'Sequence':i, 'State': state_str, 'Action': self.LTM[1][i][j], 'Reward': self.LTM[2][i], 'Seq_bias': self.triggers[i][j]})
                    
        print("Long term memory stored in " + csv_namefile)

        self.LTM = [[],[],[]] # state, action, reward
        self.triggers = []


    def load_LTM(self):
        if self.LTM_loaded == False:
            loaded_LTM = [[], [], []]
            self.LTM_loaded = True

            if self.webots_world == 0:
                csv_namefile = self.ws_path + '/data/OpenArena/LTM.csv'
            if self.webots_world == 1:
                csv_namefile = self.ws_path + '/data/LinearTrack/LTM.csv'
            if self.webots_world == 2:
                csv_namefile = self.ws_path + '/data/Tmaze/LTM.csv'
            if self.webots_world == 3:
                csv_namefile = self.ws_path + '/data/DoubleTmaze/LTM.csv'

            with open(csv_namefile, mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                current_sequence = -1

                for row in csv_reader:
                    sequence = int(row['Sequence'])
                    state = list(map(float, row['State'].split(',')))  # Convert state back to a list of floats
                    action = int(row['Action'])
                    reward = float(row['Reward'])

                    if sequence != current_sequence:
                        loaded_LTM[0].append([])
                        loaded_LTM[1].append([])
                        current_sequence = sequence

                    loaded_LTM[0][sequence].append(state)
                    loaded_LTM[1][sequence].append(action)

                    if len(loaded_LTM[2]) <= sequence:
                        loaded_LTM[2].append(reward)

            loaded_LTM = self.adjust_loaded_LTM(loaded_LTM)
            self.triggers = self.load_triggers(loaded_LTM)
            print("Long term memory loaded from " + csv_namefile)

            return loaded_LTM

    def adjust_loaded_LTM(self, loaded_LTM):
        adjusted_LTM = [[], [], []]
        LTM = loaded_LTM
        # if loaded LTM is longer than current LTM limit.
        if len(LTM[0]) > self.LTM_limit:
            LTM = [LTM[0][-self.LTM_limit:], LTM[1][-self.LTM_limit:], LTM[2][-self.LTM_limit:]]

        for i in range(len(LTM[0])):
            states = LTM[0][i]
            actions = LTM[1][i]
            reward = LTM[2][i]

            # if loaded STMs are longer than current STM limit
            if len(states) > self.STM_limit:
                # Keep only the last max_sequence_steps states and actions
                states = states[-self.STM_limit:]
                actions = actions[-self.STM_limit:]

            # if loaded STMs are shorter than current STM limit
            elif len(states) < self.STM_limit:
                padding_states = [[0.0] * len(states[0])] * (self.STM_limit - len(states))
                states.extend(padding_states)


            adjusted_LTM[0].append(states)
            adjusted_LTM[1].append(actions)
            adjusted_LTM[2].append(reward)

        return adjusted_LTM


    #def load_triggers(self, loaded_LTM):




