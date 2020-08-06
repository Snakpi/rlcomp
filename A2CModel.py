# -*- coding: utf-8 -*-
from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from utils import v_wrap, set_init, push_and_pull, record
from shared_adam import SharedAdam

import math, os, sys
import numpy as np

from MinerEnv import MinerEnv

###########################################################################################################################

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 10000
MAX_EP_STEP = 1000
SAVE_NETWORK = 100  # After this number of episodes, the model is saved for testing later. 
INPUTNUM = 198 #The number of input values for the model
ACTIONNUM = 6  #The number of actions output from the model
MAP_MAX_X = 21 #Width of the Map
MAP_MAX_Y = 9  #Height of the Map

class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.al1 = nn.Linear(s_dim, 200)
        self.al2 = nn.Linear(200, 100)
        self.al3 = nn.Linear(100, 50)
        self.al4 = nn.Linear(50, a_dim)

        a_ls = [self.al1, self.al2, self.al3, self.al4]

        self.cl1 = nn.Linear(s_dim, 200)
        self.cl2 = nn.Linear(200, 100)
        self.cl3 = nn.Linear(100, 50)
        self.cl4 = nn.Linear(50, 1)

        cr_ls = [self.cl1, self.cl2, self.cl3, self.cl4]

        set_init(a_ls + cr_ls)

        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        a1 = F.relu(self.al1(x))
        a2 = F.relu(self.al2(a1))
        a3 = F.relu(self.al3(a2))
        a4 = F.relu(self.al4(a3))
        a_probs = F.softmax(a4, dim=0)

        c1 = F.relu(self.cl1(x))
        c2 = F.relu(self.cl2(c1))
        c3 = F.relu(self.cl3(c2))
        v = F.relu(self.cl4(c3))

        return a_probs, v

    def choose_action(self, s, epsilon=.3):
        self.training = False
        a_probs, _ = self.forward(s)
        sample_0_1 = np.random.sample()
        if sample_0_1 > epsilon:
            a_ps_np = a_probs.detach().numpy()
            a = np.argmax(a_ps_np)
        else:
            a = np.random.choice(a_probs.shape[0], 1).item()
            #a = np.random.choice(a_probs.shape[0], 1, p=a_probs.tolist()).item() #### Semi-greedy epsilon-greedy algorithm (explores the actions by softmax probs)
        return a

        
    def loss_func(self, s, a, v_t):
        self.train()
        a_probs, v = self.forward(s)
        td = v_t - v
        c_loss = td.pow(2)

        m = self.distribution(a_probs)
        log_prob = m.log_prob(a)
        exp_v = log_prob * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (a_loss + c_loss).mean()
        return total_loss


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name, env):
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(INPUTNUM, ACTIONNUM)
        self.env = env


    def run(self):
        total_step = 1

        while self.g_ep.value < MAX_EP:
            try:
              mapID = np.random.randint(1, 6) #Choosing a map ID from 5 maps in Maps folder randomly
              posID_x = np.random.randint(MAP_MAX_X) #Choosing a initial position of the DQN agent on X-axes randomly
              posID_y = np.random.randint(MAP_MAX_Y) #Choosing a initial position of the DQN agent on Y-axes randomly
              #Creating a request for initializing a map, initial position, the initial energy, and the maximum number of steps of the DQN agent
              request = ("map" + str(mapID) + "," + str(posID_x) + "," + str(posID_y) + ",50,100") 
              #Send the request to the game environment (GAME_SOCKET_DUMMY.py)
              self.env.send_map_info(request)
              self.env.reset()
              s = self.env.get_state()
              print(s.shape)
              buffer_s, buffer_a, buffer_r = [], [], []
              ep_r = 0.
              for t in range(MAX_EP_STEP):
                  a = self.lnet.choose_action(v_wrap(s[None, :]), .5)
                  self.env.step(str(a))
                  s_ = self.env.get_state()
                  r = self.env.get_reward()
                  done = self.env.check_terminate()

                  if t == MAX_EP_STEP - 1:
                      done = True
                  ep_r += r
                  buffer_a.append(a)
                  buffer_s.append(s)
                  buffer_r.append(r)

                  if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                      # sync
                      push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                      buffer_s, buffer_a, buffer_r = [], [], []

                      if done:  # done and print information
                          record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                          break
                  s = s_
                  total_step += 1

            except Exception as e:
              import traceback
              traceback.print_exc()
              break

        self.res_queue.put(None)

    def gnet_opt_save(self, ep_r, loss):
        torch.save(
            {
            'global_ep': self.g_ep,
            'global_ep_r': self.g_ep_r,
            'gnet state_dict': self.gnet.state_dict(),
            'opt state_dict': self.opt.state_dict(),
            }, self.name + '.pth')
    
    def epsilon_update(self, drop_rate):
      pass

########################################################################################################



# Create header for saving learning file
"""
now = datetime.datetime.now() #Getting the latest datetime
header = ["Ep", "Step", "Reward", "Total_reward", "Action", "Epsilon", "Done", "Termination_Code"] #Defining header for the save file
filename = "Data/data_" + now.strftime("%Y%m%d-%H%M") + ".csv" 
with open(filename, 'w') as f:
    pd.DataFrame(columns=header).to_csv(f, encoding='utf-8', index=False, header=True)
"""

# Initialize environment
HOST = "localhost"
PORT = 1111
if len(sys.argv) == 3:
    HOST = str(sys.argv[1])
    PORT = int(sys.argv[2])
minerEnv = MinerEnv(HOST, PORT)
minerEnv.start()
#train = False #The variable is used to indicate that the epsilon starts to decrease.

#Training Process
if __name__ == '__main__':
    gnet = Net(INPUTNUM, ACTIONNUM)        # global network
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.95, 0.999))  # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i, minerEnv) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    res = []
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    import matplotlib.pyplot as plt
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()
