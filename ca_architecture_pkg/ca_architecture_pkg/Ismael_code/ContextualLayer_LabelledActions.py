import numpy as np
import pickle as pkl 

class ContextualLayer(object):

    def __init__(self, action_space=4, emb=20, stm=50, ltm=500,
        sequential_bias=True, value_function='default', forget="NONE",
        coll_threshold_act=0.98, coll_threshold_proportion=0.995,
        alpha_trigger=0.05, tau_decay=0.9, load_ltm=False):

        self.stm_length = stm # STM sequence length
        self.ltm_length = ltm # LTM buffer capacity: Total n of sequences stored in LTM
        self.emb_length = emb        # EMB = embedding length: Size of the state vector

        self.sequential_bias = sequential_bias   # sequential bias
        self.value_function = value_function  # TODO check it, it used to be ==
        self.forget = forget    # can be "FIFO", "SING" or "PROP"
        self.forget_ratio = 0.1

        print("STM length: ", self.stm_length)
        print("LTM length: ", self.ltm_length)
        print("EMB length: ", self.emb_length)
        print("Sequential Bias: ", self.sequential_bias)
        print("Forgetting: ", self.forget)
        print('Value function: ', self.value_function)

        self.coll_thres_act = coll_threshold_act    # default 0.9
        self.coll_thres_prop = coll_threshold_proportion    #default 0.95
        self.alpha_tr = alpha_trigger       #default 0.005
        self.alpha_dis = alpha_discrepancy      #default 0.01
        self.tau_decay = 0.9

        self.action_space = action_space                                     # can be a list "[3, 3]" or a integer "6"
        #print("CL action_space: ", self.action_space)
        #print("action_space type: ", type(self.action_space))
        self.action = 0

        self.STM = [[[0] * self.emb_length , 0] for i in range(self.stm_length)]
        self.LTM = [[],[],[]]

        self.memory_full = False
        self.memory_threshold =  memory_threshold
        self.forget_ratio = 0.01 # 1% of memories will be erased when using Forgetting PROP

        self.tr = []
        self.last_actions_indx = []
        self.selected_actions_indx = []

        self.entropy = 0.
        self.steps = 0
        #self.count = 0

        if load_ltm: self.load_LTM()

    def advance(self, state):
        # SEC retrieval phase. e (state/embeding). 
        # Main loop of CL: (1) Select memories similar to current state. (2) Compute policy based on selected memories (3) Select action based on policy
        q = np.ones(self.action_space) / self.action_space
        state_array = np.array(state)

        if len(self.LTM[0]) > 0:

            # Initialize sequential bias if active
            bias = 1
            if self.sequential_bias:
                bias = np.array(self.tr)
                #print("bias: ", bias)
                #print("bias length: ", len(bias[0])) # proportional to sequence's length, n = LTM sequences

            # (1) Select those memories that surpass a similarity threshold with the current state + those selected by the sequential bias.
            #collectors = (1 - (np.sum(np.abs(e - self.LTM[0]), axis=2)) / len(e)) * bias
            collectors = (1 - (np.sum(np.abs(state - np.array(self.LTM[0])), axis=2)) / len(state)) * bias
            #print ("collectors ", collectors) # proportional to sequence's length, n = LTM sequences
            #print ("collectors length", len(collectors[0])) 
            #print ("collectors relative", collectors/collectors.max())

            # Collector values must be above both thresholds (absolute and relative) to contribute to action.
            self.selected_actions_indx = (collectors > self.coll_thres_act) & ((collectors/collectors.max()) > self.coll_thres_prop) # proportional to sequence's length, n = LTM sequences
            #print ("selected_actions_indx ", self.selected_actions_indx)
            #print ("selected_actions_indx length", len(self.selected_actions_indx))
            #print ("selected_actions_indx length [0]", len(self.selected_actions_indx[0]))

            if np.any(self.selected_actions_indx):

                actions = np.array(self.LTM[1])[self.selected_actions_indx]
                # chooose (normalized, or relative) rewards of sequences with actions selected 
                rewards = np.array(self.LTM[2])[(np.nonzero(self.selected_actions_indx)[0])]
                rewards = rewards/rewards.max()
                # choose (normalized) distances of each action selected within its sequence -- distance from action to reward at the end of the sequence
                distances = (self.stm_length - np.nonzero(self.selected_actions_indx)[1])/self.stm_length
                # choose collector info about the actions selected (that take euclidean distance of current state and collector's selected prototypes)
                collectors = collectors[self.selected_actions_indx]

                # map each selected action-vector into a matrix of N dimensions where N are the dimensions of the action space
                #m = np.zeros((len(actions), self.action_space[0], self.action_space[1]))
                m = np.zeros((len(actions), self.action_space))
                #m[np.arange(len(actions)), actions[:,0].astype(int), actions[:,1].astype(int)] = ((collectors*rewards)/distances)
                #m[np.arange(len(actions)), actions[:,0].astype(int), actions[:,1].astype(int)] = collectors*(rewards*np.exp(-distances/self.tau_decay))

                # (2) Generate policy from selected state-action couplets based on 3 factors: reward value, distance to reward, state similarity
                if self.value_function == 'default':
                    #print('COMPUTING ACTIONS CLASSIC SEC...')
                    #m[np.arange(len(actions)), actions[:,0].astype(int), actions[:,1].astype(int)] = collectors*(rewards*np.exp(-distances/self.tau_decay))
                    m[np.arange(len(actions)), actions[:].astype(int)] = collectors*(rewards*np.exp(-distances/self.tau_decay))
                #Ablations
                if self.value_function == 'default':
                    #print('COMPUTING ACTIONS CLASSIC SEC...')
                    m[np.arange(len(actions)), actions[:].astype(int)] = collectors*(rewards*np.exp(-distances/self.tau_decay))
                if self.value_function == 'noGi':
                    #print('COMPUTING ACTIONS WITHOUT SIMILARITY...')
                    m[np.arange(len(actions)), actions[:].astype(int)] = rewards*np.exp(-distances/self.tau_decay)
                if self.value_function == 'noDist':
                    #print('COMPUTING ACTIONS WITHOUT DISTANCE...')
                    m[np.arange(len(actions)), actions[:].astype(int)] = collectors*rewards
                if self.value_function == 'noRR':
                    #print('COMPUTING ACTIONS WITHOUT REWARD...')
                    m[np.arange(len(actions)), actions[:].astype(int)] = collectors*np.exp(-distances/self.tau_decay)
                if self.value_function == 'soloGi':
                    #print('COMPUTING ACTIONS WITH ONLY SIMILARTY...')
                    m[np.arange(len(actions)), actions[:].astype(int)] = collectors
                if self.value_function == 'soloDist':
                    #print('COMPUTING ACTIONS WITH ONLY DISTANCE...')
                    m[np.arange(len(actions)), actions[:].astype(int)] = np.exp(-distances/self.tau_decay)
                if self.value_function == 'soloRR':
                    #print('COMPUTING ACTIONS WITH ONLY RELATIVE REWARD...')
                    m[np.arange(len(actions)), actions[:].astype(int)] = rewards

                q = np.sum(m, axis=0)
                #q = q + np.abs(q.min())+1 # NEW
                q = q/q.sum()  #proportion of being selected based on the action's relative reward based on the stored experiences
                q = q.flatten()

                # Entropy of the prob distr for policy stability. (The sum of the % distribution multiplied by the logarithm -in base 2- of p)
                self.compute_entropy(q)

                #(3) Select action based on policy
                #ac_indx = np.random.choice(np.arange(int(self.action_space[0]*self.action_space[1])), p=q)
                #self.action = [int(ac_indx/self.action_space[0]), int(ac_indx%self.action_space[1])]
                self.action = np.random.choice(np.arange(q.shape[0]), p=q)

            else:
                self.action = -1 # Null action (indicates the agent that contextual did not generate a valid action)

            self.selected_actions_indx = self.selected_actions_indx.tolist()
            #print ("selected_actions_indx ", self.selected_actions_indx)
            
        return self.action #* self.enabled


    def compute_entropy(self, q):
        # Entropy of the prob distr for policy stability. (The sum of the % distribution multiplied by the logarithm -in base 2- of p)
        #q = policy
        #qlog = np.log2(q)
        #infs = np.where(np.isinf(qlog))
        #qlog[infs] = 0.
        #qqlog = q*qlog
        #qsum = -np.sum(qqlog)
        #self.entropy = qsum
        self.entropy = np.sum(-policy * np.log2(policy + 1e-12))  # avoid log(0) by adding a small constant

    # Couplet expects a list with [state, action]; Goal is -1 or 1 indicating aversive or appetitive goal has been reached.
    def update_STM(self, couplet=[]):

        # Update STM buffer with the new couplet (FIFO).
        self.STM.append(couplet)
        self.STM = self.STM[1:] # renew the STM buffer by removing the first value of the STM
        #print ("STM: ", self.STM[-1])

    def update_sequential_bias(self):
        # NEW: Update the last actions index first!
        self.last_actions_indx = np.copy(self.selected_actions_indx).tolist()  # Updates the last action indexes with the current actions indexes.
        #print ("last_actions_indx ", self.last_actions_indx)

        # Update trigger values.
        if (len(self.tr) > 0) and self.sequential_bias:
            self.tr = (np.array(self.tr) * (1. - self.alpha_tr)) + self.alpha_tr  # trigger values decay by default
            self.tr[(self.tr < 1.)] = 1.       # all trigger values below 1 are reset to 1.
            tr_last_actions_indx = np.array(self.last_actions_indx)
            self.tr[tr_last_actions_indx] = 1.    # NEW: the trigger value of previously selected segments are reset to 1!!!
            last_actions_shifted = np.roll(self.last_actions_indx, 1, axis=1) # shift the matrix one step to the right
            last_actions_shifted[:, 0] = False  # set the first value of each sequence to False

            # NEW: increase ONLY the trigger value of the next element in sequence (after the ones selected before)!
            tr_change_indx = np.array(last_actions_shifted)
            self.tr[tr_change_indx] += 0.01    # NEW: increase by an arbitrary amount (this amount should be tuned or modified).
            self.tr = self.tr.tolist()

            ## TO-DO ADD FORGETTING OF SEQUENCES BASED ON TRIGGER VALUES.

    def reset_memory(self):
        # MEMORY RESET when finishing an episode
        self.reset_STM()
        self.reset_sequential_bias()

    def reset_STM(self):
        # Reset STM when beggining a new episode
        #self.STM = [[np.zeros(self.emb_length), np.zeros(2)] for _ in range(self.stm_length)] # pl = prototype length (i.e. dimension of the state vector)
        #self.STM = [[np.zeros(self.emb_length), np.zeros(1)] for _ in range(self.stm_length)] # pl = prototype length (i.e. dimension of the state vector)
        if type(self.action_space) != list:
            #self.STM = [[np.zeros(self.emb_length), np.zeros(1)] for _ in range(self.stm_length)] # pl = prototype length (i.e. dimension of the state vector)
            self.STM = [[[0] * self.emb_length , 0] for i in range(self.stm_length)]
            #print(self.STM)
        else:
            #self.STM = [[np.zeros(self.emb_length), np.zeros(2)] for _ in range(self.stm_length)] # pl = prototype length (i.e. dimension of the state vector)
            self.STM = [[[0] * self.emb_length , [0, 0]] for i in range(self.stm_length)]
            #print(self.STM)

    def reset_sequential_bias(self):
        # Reset trigger values when beggining a new episode
        if (len(self.tr) > 0):
            self.tr = np.array(self.tr)
            self.tr[:] = 1.0
            self.tr = self.tr.tolist()


    def update_LTM(self, reward=0):
        # Update LTM if reached goal state and still have free space in LTM.
        if (reward > 0.) and (len(self.LTM[2]) < self.ltm_length):
            print ("GOAL STATE REACHED! REWARD: ", reward)
            #print ("N STEPS TO REACH REWARD:", self.count)
            self.LTM[0].append([s[0] for s in self.STM])  #append prototypes of STM couplets.
            self.LTM[1].append([a[1] for a in self.STM])  #append actions of STM couplets.
            self.LTM[2].append(reward)
            self.tr.append(np.ones(self.stm_length).tolist())
            self.selected_actions_indx.append(np.zeros(self.stm_length, dtype='bool').tolist())
            self.last_actions_indx.append(np.zeros(self.stm_length, dtype='bool').tolist())
            print("Sequences in LTM", len(self.LTM[2]), ", Sequence length:", len(self.STM))

        # Remove sequences when LTM is full
        if (len(self.LTM[2]) >= self.ltm_length) and self.forget != "NONE":
            print ("LTM IS FULL! FORGETTING ACTIVATED...", self.forget)
            #print ("CURRENT LTM rewards: ", self.LTM[2])

            if self.forget == "FIFO":
                self.LTM[0] = np.delete(np.array(self.LTM[0]),0,0).tolist()
                self.LTM[1] = np.delete(np.array(self.LTM[1]),0,0).tolist()
                self.LTM[2] = np.delete(np.array(self.LTM[2]),0,0).tolist()
                self.tr = np.delete(np.array(self.tr),0,0).tolist()
                self.selected_actions_indx = np.delete(np.array(self.selected_actions_indx),0,0).tolist()
                self.last_actions_indx = np.delete(np.array(self.last_actions_indx),0,0).tolist()
                #print ("FIRST MEMORY SEQUENCE FORGOTTEN")
                #print ("UPDATED LTM rewards: ", self.LTM[2])
            elif self.forget == "SING":
                idx = np.argsort(self.LTM[2])
                self.LTM[0] = np.delete(np.array(self.LTM[0]),idx[0],0).tolist()
                self.LTM[1] = np.delete(np.array(self.LTM[1]),idx[0],0).tolist()
                self.LTM[2] = np.delete(np.array(self.LTM[2]),idx[0],0).tolist()
                self.tr = np.delete(np.array(self.tr),idx[0],0).tolist()
                self.selected_actions_indx = np.delete(np.array(self.selected_actions_indx),idx[0],0).tolist()
                self.last_actions_indx = np.delete(np.array(self.last_actions_indx),idx[0],0).tolist()
                #print ("LOWEST REWARD SEQUENCE FORGOTTEN")
                #print ("UPDATED LTM rewards: ", self.LTM[2])
            elif self.forget == "PROP":
                maxfgt = int(len(self.LTM[2]) * self.forget_ratio)
                idx = np.argsort(self.LTM[2])
                self.LTM[0] = np.delete(np.array(self.LTM[0]),idx[0:maxfgt],0).tolist()
                self.LTM[1] = np.delete(np.array(self.LTM[1]),idx[0:maxfgt],0).tolist()
                self.LTM[2] = np.delete(np.array(self.LTM[2]),idx[0:maxfgt],0).tolist()
                self.tr = np.delete(np.array(self.tr),idx[0:maxfgt],0).tolist()
                self.selected_actions_indx = np.delete(np.array(self.selected_actions_indx),idx[0:maxfgt],0).tolist()
                self.last_actions_indx = np.delete(np.array(self.last_actions_indx),idx[0:maxfgt],0).tolist()
                #print ("NUMBER OF FORGOTTEN SEQUENCES: ", maxfgt)
                #print ("UPDATED LTM rewards: ", self.LTM[2])

    def save_LTM(self, savePath, ID, n=1):
        with open(savePath+ID+'ltm'+str(len(self.LTM[2]))+'_'+str(n)+'.pkl','wb') as f:
            pkl.dump(self.LTM, f)

    def load_LTM(self, filename):
        ID = '/LTMs/'+filename
        #ID = '/LTMs/LTM100_N961.pkl'
        # open a file, where you stored the pickled data
        file = open(ID, 'rb')
        # load information from that file
        self.LTM = pkl.load(file)
        print("LTM loaded!! Memories retrieved: ", len(self.LTM[2]))
        for s in (self.LTM[2]):
            self.tr.append(np.ones(self.stm_length).tolist())
            self.selected_actions_indx.append(np.zeros(self.stm_length, dtype='bool').tolist())
            self.last_actions_indx.append(np.zeros(self.stm_length, dtype='bool').tolist())

        file.close()