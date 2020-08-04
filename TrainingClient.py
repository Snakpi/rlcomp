import sys
from A2CModel import ActorCritic # A class of creating a deep q-learning model
from MinerEnv import MinerEnv # A class of creating a communication environment between the DQN model and the GameMiner environment (GAME_SOCKET_DUMMY.py)

import pandas as pd
import datetime 
import numpy as np


HOST = "localhost"
PORT = 1111
if len(sys.argv) == 3:
    HOST = str(sys.argv[1])
    PORT = int(sys.argv[2])

# Create header for saving learning file
now = datetime.datetime.now() #Getting the latest datetime
header = ["Ep", "Step", "Reward", "Total_reward", "Action", "Epsilon", "Done", "Termination_Code"] #Defining header for the save file
filename = "Data/data_" + now.strftime("%Y%m%d-%H%M") + ".csv" 
with open(filename, 'w') as f:
    pd.DataFrame(columns=header).to_csv(f, encoding='utf-8', index=False, header=True)

# Parameters for training a DQN model
N_EPISODE = 10000 #The number of episodes for training
MAX_STEP = 1000   #The number of steps for each episode
BATCH_SIZE = 32   #The number of experiences for each replay 
MEMORY_SIZE = 100000 #The size of the batch for storing experiences
SAVE_NETWORK = 100  # After this number of episodes, the DQN model is saved for testing later. 
INITIAL_REPLAY_SIZE = 1000 #The number of experiences are stored in the memory batch before starting replaying
INPUTNUM = 198 #The number of input values for the DQN model
ACTIONNUM = 6  #The number of actions output from the DQN model
MAP_MAX_X = 21 #Width of the Map
MAP_MAX_Y = 9  #Height of the Map

# Initialize a DQN model and a memory batch for storing experiences
A2CAgent = ActorCritic(INPUTNUM, ACTIONNUM)
ac_optim = optim.RMSProp(A2CAgent.parameters(), lr=A2CAgent.learning_rate)
# Initialize environment
minerEnv = MinerEnv(HOST, PORT) #Creating a communication environment between the DQN model and the game environment (GAME_SOCKET_DUMMY.py)
minerEnv.start()  # Connect to the game

train = False #The variable is used to indicate that the replay starts, and the epsilon starts decrease.
#Training Process
#the main part of the deep-q learning agorithm 
for episode in range(N_EPISODE):
    try:
        log_probs, values, rewards = [], [], []
        
        entropy_term = 0
        all_lengths = []
        average_lengths = []
        all_rewards = []

        # Choosing a map in the list
        mapID = np.random.randint(1, 6) #Choosing a map ID from 5 maps in Maps folder randomly
        posID_x = np.random.randint(MAP_MAX_X) #Choosing a initial position of the agent on X-axes randomly
        posID_y = np.random.randint(MAP_MAX_Y) #Choosing a initial position of the agent on Y-axes randomly
        #Creating a request for initializing a map, initial position, the initial energy, and the maximum number of steps of the DQN agent
        request = ("map" + str(mapID) + "," + str(posID_x) + "," + str(posID_y) + ",50,100") 
        #Send the request to the game environment (GAME_SOCKET_DUMMY.py)
        minerEnv.send_map_info(request)

        # Getting the initial state
        minerEnv.reset() #Initialize the game environment
        s = minerEnv.get_state()#Get the state after reseting. 
                                #This function (get_state()) is an example of creating a state for the DQN model 
        total_reward = 0 #The amount of rewards for the entire episode
        terminate = False #The variable indicates that the episode ends
        maxStep = minerEnv.state.mapInfo.maxStep #Get the maximum number of steps for each episode in training
        #Start an episde for training
        for step in range(maxStep):
            
            value, policy_dist = A2CAgent.forward(s)
            value = value.detach().numpy()[0,0]
            dist = policy_dist.detach().numpy()
            action = np.random.choice(ACTIONNUM, p=np.squeeze(dist)) # Getting an action from the DQN model from the state (s)
            
            minerEnv.step(str(action))  # Performing the action in order to obtain the new state
            
            log_prob = torch.log(policy_dist.squeeze(0)[action])
            entropy = -np.sum(np.mean(dist) * np.log(dist))
            
            s_next = minerEnv.get_state()  # Getting a new state
            reward = minerEnv.get_reward()  # Getting a reward
            terminate = minerEnv.check_terminate()  # Checking the end status of the episode
            
            values.append(value)
            log_probs.append(log_probs)
            rewards.append(reward) #Plus the reward to the total rewad of the episode
            entropy_term += entropy
            s = s_next #Assign the next state for the next step.

            if step == MAX_STEP - 1:
                Qval, _ = actor_critic.forward(new_state)
                Qval = Qval.detach().numpy()[0,0]
                all_rewards.append(np.sum(rewards))
                all_lengths.append(step)
                average_lengths.append(np.mean(all_lengths[-10:]))
                if episode % 40 == 0:                    
                    sys.stdout.write("episode: {}, reward: {}, total length: {}, average length: {} \n".format(episode, np.sum(rewards), steps, average_lengths[-1]))
                break

            # Saving data to file
            save_data = np.hstack(
                [episode + 1, step + 1, reward, total_reward, action, A2CAgent.epsilon, terminate]).reshape(1, 7)
            with open(filename, 'a') as f:
                pd.DataFrame(save_data).to_csv(f, encoding='utf-8', index=False, header=False)
            
            if terminate == True:
                #If the episode ends, then go to the next episode
                break
        
        # compute Q values
        Qvals = np.zeros_like(values)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + A2CAgent.gamma * Qval
            Qvals[t] = Qval
  
        #update actor critic
        values = torch.FloatTensor(values)
        Qvals = torch.FloatTensor(Qvals)
        log_probs = torch.stack(log_probs)
        
        advantage = Qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

        ac_optim.zero_grad()
        ac_loss.backward()
        ac_optim.step()

        # Iteration to save the network weights
        if (np.mod(episode_i + 1, SAVE_NETWORK) == 0 and train == True):
            now = datetime.datetime.now() #Get the latest datetime
            A2CAgent.save_actor("TrainedModels/",
                                "A2C_weights_" + now.strftime("%Y%m%d-%H%M") + "_ep" + str(episode + 1))

        #Print the training information after the episode
        print('Episode %d ends. Number of steps is: %d. Accumulated Reward = %.2f. Epsilon = %.2f .Termination code: %d' % (
            episode + 1, step + 1, total_reward, A2CAgent.epsilon, terminate))
        
        #Decreasing the epsilon if the replay starts
        if train == True:
            A2CAgent.update_epsilon()

    except Exception as e:
        import traceback

        traceback.print_exc()
        # print("Finished.")
        break
