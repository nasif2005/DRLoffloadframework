# https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/gridworld.py
import random
import numpy


import xlrd
import numpy as np
from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from collections import deque



class DQNAgent:
    def __init__(self,env):
        #self.state_size = self.env.observation_space.shape[0]
        #self.action_size = self.env.action_space.n
        self.memory  = deque(maxlen=2000)
        self.env = env
        self.model = self.createModel()
        self.target_model = self.createModel()
        

    def createModel(self):
        model  = Sequential()
        model.add(Dense(512, input_dim = 132, activation="relu"))
        model.add(Dense(512, activation="relu"))
        model.add(Dense(512, activation="relu"))
        model.add(Dense(3))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=1e-3))
        print(model.summary())
        return model       
    
    def findAction(self, state):
        #print ("act")
        #print ("#####1#")
        #print (state.shape)
        #print (state)
        #print ("#####1#")
        state.resize(1, 132)
        #print ("Action_list: ", self.model.predict(state))# Action_list:  [[0.00092562 0.0005148 ]]
        #print (np.argmax(self.model.predict(state)))
        return (np.argmax(self.model.predict(state)))

    def remember(self, state, action, next_state, reward, done):    
        self.memory.append([state, action, reward, next_state, done])
        #print ("memory:", self.memory)
     
    def replay(self):
        #print("replay") 
        #next_state.resize(1, 4)
        #print (max(self.target_model.predict(next_state)))
        #print (max(self.target_model.predict(next_state)[0]))
        if len(self.memory) < 10:
            return
        
        # Randomly sample minibatch from the memory
        samples = random.sample(self.memory, 10)
        
        for sample in samples:
            state, action, reward, next_state, done = sample
            
            next_state.resize(1, 132)
            state.resize(1, 132)
            #target_actionQvalue_listN = self.target_model.predict(next_state) 
            # next state
            eval_actionQvalue_listN = self.model.predict(next_state) 
            # current state
            eval_actionQvalue_Currest_list = self.model.predict(state) 
            
            
            #print ("eval_actionQvalue_listN: ", eval_actionQvalue_listN)
            next_action = np.argmax(eval_actionQvalue_listN)
            #print ("next_action: ", next_action)
            #print ("target_actionQvalue_listN: ", target_actionQvalue_listN)
            my_value =  eval_actionQvalue_listN[0][next_action] 
            #print ("my_nextvalue: ", my_value)
            newQ_value = reward + 0.85 * my_value
            #print ("newQ_value: ", newQ_value)
            eval_actionQvalue_Currest_list[0][action] = reward + 0.85 * my_value
            #print ("eval_actionQvalue_listN: ", eval_actionQvalue_listN)
            #print ("eval_actionQvalue_Currest_list: ", eval_actionQvalue_Currest_list)
            self.model.fit(state, eval_actionQvalue_Currest_list, epochs=1, verbose=0)
            #print("Z-Z")
            """
            state.resize(1, 132)
            target_action_list = self.target_model.predict(state) 
            # e.g. target:  [[-0.25961223  0.05271231]]
            #print("target_action_list: ", target_action_list)
            if done:
                target_action_list[0][action] = reward
            else:
                #print ("*************************D")
                next_state.resize(1, 132)
                Q_future = max(self.target_model.predict(next_state)[0])# arrange action max to min# Action_list:  [[0.00092562(*) 0.0005148 ]]
                #print ("Q_future: ", Q_future)
                target_action_list[0][action] = reward + Q_future * 0.85
                #print("target_action_list[0][action]: ", target_action_list[0][action])
                # training
                self.model.fit(state, target_action_list, epochs=1, verbose=0)
            """

    def target_train(self):     
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * 0.125 + target_weights[i] * (1 - 0.125)
        
        self.target_model.set_weights(target_weights)
      
    def save_model(self):
            self.model.save("gridworld-dqni10.h5")
            
    def load(self, name):
        self.model = load_model(name)
        
        
    def test(self):
        #self.load("gridworld-dqn6000512-i2L.h5")
        #self.load("gridworld-dqn6000512.h5")
        self.load("gridworld-dqn-i10.h5")
        
        for i in range(1):
            state, reward, done, info = self.env.reset_for_test(0)
            state.resize(1, 132)
            done = False
            j = 0
            #while not done:
                #self.env.render()
            action = np.argmax(self.model.predict(state))
            print ("prediction action: ", action)
            #action = 2
            #action = random.randint(0, 2)
            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            state.resize(1, 132)
            j =  j + reward
                #if done:
            print("episode: {}/{}, reward: {}".format(i, 10, j))
                    #break
    

########################################################################

class myGridWorld(object):
    
    def __init__(self):
        #print('Environment initialized')
        #print(custom_data)
        global counter 
        counter = 0
        
    
    def step(self, action):
        #print('Step successful!', action)
        next_state = self._take_action(action)
        reward = self._take_reward(action)
         # when done is False
        done = False    
        info = {}
        return next_state, reward, done, info
    
    
    def resetimage(self, col):
        global counter
        global image_data, image_accuracy

        ### READ Image Data info from excel file
        loc = ("image.xlsx")
        wb = xlrd.open_workbook(loc)
        sheet = wb.sheet_by_index(0)
        sheet.cell_value(0, 0)
        image_data = []
        
        for i in range(sheet.nrows):
            sheet.cell_value(i, col)
            image_data = np.append(image_data, sheet.cell_value(i, col)) # [row, col]
        image_data.resize(1, 128)  
        
        ### READ Inage Accuracy info from excel file
        loc = ("imageaccuracy.xlsx")
        wb = xlrd.open_workbook(loc)
        sheet = wb.sheet_by_index(0)
        sheet.cell_value(0, 0)
        image_accuracy = []
            
        for j in range(sheet.nrows):
            sheet.cell_value(j, col)
            image_accuracy = np.append(image_accuracy, sheet.cell_value(j, col)) # [row, col]
     
        #print("image_accuracy: ", image_accuracy)

    
    def reset(self):
        #print('Environment reset')
        global  total_reward 
        total_reward = 0
        global Tsmart, Tedge, Tcore
        global weight 
        global image_data
        global counter
        
        Tsmart = round(random.uniform(0.1, 1.5), 1) #0.20 #(0~1) processing time
        Tedge = round(random.uniform(0.1, 1.5), 1) #0.98 #(0~1) processing time
        Tcore = round(random.uniform(0.1, 2.5), 1) #0.90 # (1.5~2.5) processing time
        
        weight = round(random.uniform(0, 1), 1) #0.4 #(0~1) processing time
        """
        if (counter == 1):
            Tsmart = 0.80 #(0~1) processing time
            Tedge = 0.60  #(0~1) processing time
            Tcore = 1.70  # (1.5~2.5) processing time
            
            weight = 0.7 #(0~1) processing time  
        
        
        counter = counter + 1
        """
            
        ob = numpy.concatenate((image_data, Tsmart, Tedge, Tcore, weight), axis=None)
        
        done = False
        info = {}
        return ob, total_reward, done, info

    def reset_for_test(self, col):
        print('Environment reset')
        global total_reward 

        total_reward = 3
        global Tsmart, Tedge, Tcore 
        global weight
        global image_data, image_accuracy
        
        Tsmart = 0.1
        Tedge = 0.9
        Tcore = 0.9
        weight = 0.1
        
        print("Tsmart: ", Tsmart, "Tedge: ", Tedge, "Tcore: ",Tcore)
        
        # Read image data from file
        loc = ("image.xlsx")
        wb = xlrd.open_workbook(loc)
        sheet = wb.sheet_by_index(0)
        sheet.cell_value(0, 0)
        image_data = []
        
        for i in range(sheet.nrows):
            sheet.cell_value(i, col)
            image_data = np.append(image_data, sheet.cell_value(i, col)) # [row, col]
        image_data.resize(1, 128)  
        
        # Read Image Accuracy info from excel file
        loc = ("imageaccuracy.xlsx")
        wb = xlrd.open_workbook(loc)
        sheet = wb.sheet_by_index(0)
        sheet.cell_value(0, 0)
        image_accuracy = []
            
        for i in range(sheet.nrows):
            sheet.cell_value(i, col)
            image_accuracy = np.append(image_accuracy, sheet.cell_value(i, col)) # [row, col]
        
        ob = numpy.concatenate((image_data, Tsmart, Tedge, Tcore, weight), axis=None)
        
        done = False
        info = {}
        return ob, total_reward, done, info

   
    def _take_action(self, action):
        global Tsmart, Tedge, Tcore
        global weight
        global image_data
        
        if (action == 0):
            #print("smartphone_action")    
            ob = numpy.concatenate((image_data, Tsmart, Tedge, Tcore, weight), axis=None)
            return ob
        
        elif (action == 1):
            #print("edgecloud_action")
            ob = numpy.concatenate((image_data, Tsmart, Tedge, Tcore, weight), axis=None)
            return ob
        
        elif (action == 2):
            #print("corecloud_action") 
            ob = numpy.concatenate((image_data, Tsmart, Tedge, Tcore, weight), axis=None)
            return ob
        
        
    def _take_reward(self, action):
        global Tsmart, Tedge, Tcore, weight
        global image_accuracy
        
        Esmart = 8.736
        Eedge = 1
        Ecore = 1
        
        Csmart = 0
        Cedge = 0.000046 * Tedge 
        Ccore = 0.0000555 * Tcore
        
        #print ( Csmart, Cedge, Ccore)
        
        Asmart = image_accuracy[0]
        Aedge = image_accuracy[1]
        Acore = image_accuracy[2]
        
        
        time_max = max (Tsmart, Tedge, Tcore)
        energy_max = max (Esmart, Eedge, Ecore)
        cost_max = max (Csmart, Cedge, Ccore)
        
        Tsmart0 = 1 - (Tsmart / time_max )
        Tedge0 = 1 - (Tedge / time_max )
        Tcore0 = 1 - (Tcore / time_max )
        
        Esmart0 = 1 - (Esmart / energy_max)
        Eedge0 = 1 - (Eedge / energy_max)
        Ecore0 = 1 - (Ecore / energy_max)
        
        Csmart0 = 1 - (Csmart / cost_max)
        Cedge0 = 1 - (Cedge / cost_max)
        Ccore0 = 1 - (Ccore / cost_max)
        
        smartphone_performance1 = Tsmart0 + Esmart0 + Csmart0
        edge_performance1 = Tedge0 + Eedge0 + Cedge0
        core_performance1 = Tcore0 + Ecore0 + Ccore0
        
        #print ("*************************************************")
        
        #print("time_max: ", time_max)
        ###
        #print("energy_max: ", energy_max)
        #print("cost_max: ", cost_max)
        #print ("counter::::", counter)
        """
        print("Asmart: ", Asmart)
        print("Aedge: ", Aedge)
        print ("Acore:", Acore)
        
        print("Tsmart: ", Tsmart)
        print("Tedge: ", Tedge)
        print ("Tcore:", Tcore)
        print ("*************************************************")
        print("smartphone_performance1(t,e,c): ", smartphone_performance1/3)
        print("edge_performance1(t,e,c): ", edge_performance1/3)
        print("core_performance1(t,e,c): ", core_performance1/3)
        """
        Ps = (smartphone_performance1/3) * weight  + (Asmart) * ( 1 - weight )
        Pe = (edge_performance1/3) * weight + (Aedge ) * ( 1 - weight )
        Pc = (core_performance1/3) * weight + (Acore ) * ( 1 - weight )  
        """
        print ("*************************************************")
        print("Ps: ", Ps)
        print("Pe: ", Pe)
        print("Pc: ", Pc)
        """
        ###
        if (action == 0):
            #print("smartphone_reward")
            reward =  Ps
            return reward

        elif (action == 1):
            #print("edgecloud_reward")       
            reward =  Pe
            return reward
        
        elif (action == 2):
            #print("corecloud_reward")
            reward = Pc
            return reward
        
        
        
   
###########################################################################3
if __name__ == '__main__':
    env = myGridWorld()

    eplison = 0.3
    agent = DQNAgent(env)

    #agent.test()
    ######state = env.reset()

    for j in range(10):      # j = image id
        env.resetimage(j) 
        
        for i in range(8000):  # i = no. of time variation
            print ("i: ", i, "j: ", j)
            done = False
            state, reward, done, info = env.reset()
            #print ("######")
            #print (state)
            #print (state.shape)
            #print ("######")
            for i in range(50):  # memorize
                # Select an action (randomly or NN)
                if (random.uniform(0, 1) > eplison):
                    #print("use NN")
                    action = agent.findAction(state)
                    #print ("prediction action: ", action)
                else:
                    #print ("random choice")
                    action = random.randint(0, 2) #env.action_space.sample()
                # Apply selected action on 'env'
                
                next_state, reward, done, _ = env.step(action)
                #print ("##############")
                #print (next_state)
                #print (reward)
                #print ("##############")
                # store in the memory
                agent.remember(state, action, next_state, reward, done)
             
            agent.replay()       # internally iterates default (prediction) model
            
            #agent.target_train() # transfer WEIGHT of 'target_model' to 'model'
            
            state = next_state
    
    agent.save_model()
 