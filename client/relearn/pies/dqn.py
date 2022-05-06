
import os
import numpy as np

import torch as T
import torch.nn as nn
#import torch.optim as optim
#import torch.nn.functional as F

from copy import deepcopy


class QnetRELUn(nn.Module):
    def __init__(self, state_dim, LL, action_dim):
        super(QnetRELUn, self).__init__()
        #self.flatten = nn.Flatten()
        self.n_layers = len(LL)
        if self.n_layers < 1:
            raise ValueError('need at least 1 layers')
        layers = [nn.Linear(state_dim, LL[0]), nn.ReLU()]
        for i in range(self.n_layers-1):
            layers.append(nn.Linear(LL[i], LL[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(LL[-1], action_dim))
        self.SEQL = nn.Sequential(*layers)

    def get_weights(self):
        return self.state_dict()

    def forward(self, x):
        logits = self.SEQL(x)
        return logits


class PIE:

    """ 
        Implements DQN based Policy

        state_dim       Observation Space Shape
        LL              List of layer sizes for eg. LL=[32,16,8]
        action_dim      Action Space (should be discrete)
        opt             torch.optim     (eg - torch.optim.Adam)
        cost            torch.nn.<loss> (eg - torch.nn.MSELoss)
        lr              Learning Rate for DQN Optimizer ()
        dis             discount factor
        mapper          a mapper from state space to DQN input eg - lambda x: x.reshape(a,b)
        tuf             target update frequency (if tuf==0 then doesnt use target network)
        double          uses double DQN algorithm (with target network)
        device          can be 'cuda' or 'cpu' 
                        #self.device = 'cuda' if T.cuda.is_available() else 'cpu'

        Note:
            # single DQN can either be trained with or without target
            # if self.tuf > 0 then it means target T exists and need to updated, otherwise T is same as Q
            # Note that self.T = self.Q if self.tuf<=0


    """

    def __init__(self, state_dim, LL, action_dim, opt, cost, lr, dis, mapper, double=False, tuf=0,  device='cpu', seed=None):

        if double and tuf <= 0:
            raise ValueError(
                "double DQN requires a target network, set self.tuf>0")
        self.lr, self.dis = lr, dis
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rand = np.random.default_rng(seed)
        self.tuf = tuf
        self.double = double
        self.mapper = mapper
        self.device = device
        self.opt = opt
        self.cost = cost
        self.base_model = QnetRELUn(state_dim, LL, action_dim).to(self.device)
        self.Q = QnetRELUn(state_dim, LL, action_dim).to(self.device)
        self.T = QnetRELUn(state_dim, LL, action_dim).to(
            self.device) if (self.tuf > 0) else self.Q
        self.clear()

    def clear(self):
        self._clearQ()
        self.optimizer = self.opt(
            self.Q.parameters(), lr=self.lr)  # opt = optim.Adam
        self.loss_fn = self.cost()  # cost=nn.MSELoss()
        self.train_count = 0
        self.update_count = 0

    def _clearQ(self):
        self.Q.load_state_dict(self.base_model.state_dict())
        self.Q.eval()
        if (self.tuf > 0):
            self.T.load_state_dict(self.base_model.state_dict())
            self.T.eval()

    def predict(self, state):
        st = T.tensor(self.mapper(state), dtype=T.float32)
        qvals = self.Q(st)
        m, i = T.max(qvals, dim=0)
        return i.item()

    def _prepare_batch(self, memory, size):
        batch = memory.sample(size)
        #print("Batch", batch)
        steps = len(batch)
        cS, nS, act, reward, done = [], [], [], [], []
        for i in batch:
            cSi, nSi, acti, rewardi, donei = memory.mem[i]
            cS.append(self.mapper(cSi))
            nS.append(self.mapper(nSi))
            act.append(acti)
            reward.append(rewardi)
            done.append(int(donei))
        return steps, np.arange(steps), \
            T.tensor(cS, dtype=T.float32).to(self.device), \
            T.tensor(nS, dtype=T.float32).to(self.device), \
            np.array(act), \
            T.tensor(reward, dtype=T.float32).to(self.device), \
            T.tensor(done, dtype=T.float32).to(self.device)

    def learn(self, memory, batch_size):
        steps, indices, cS, nS, act, reward, done = self._prepare_batch(
            memory, batch_size)
        target_val = self.T(nS)
        if not self.double:
            updater, _ = T.max(target_val, dim=1)
        else:
            # tensor.max returns indices as well
            _, target_next = T.max(self.Q(nS), dim=1)
            updater = T.zeros(steps, dtype=T.float32)
            updater[indices] = target_val[indices, target_next[indices]]
        updated_q_values = reward + self.dis * updater * (1 - done)

        # Compute prediction and loss
        pred = self.Q(cS)
        target = pred.detach().clone()
        target[indices, act[indices]] = updated_q_values[indices]
        loss = self.loss_fn(pred, target)  # T.tensor()
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.train_count+=1

        if (self.tuf>0):
            if self.train_count % self.tuf == 0:
                self.T.load_state_dict(self.Q.state_dict())
                self.update_count+=1

    def render(self, mode=0, p=print):
        p('=-=-=-=-==-=-=-=-=\nQ-NET\n=-=-=-=-==-=-=-=-=')
        p(self.Q)
        p('Train Count:', self.train_count)
        if (self.tuf > 0):
            p('Update Count:', self.update_count)
        p('=-=-=-=-==-=-=-=-=!Q-net=-=-=-=-==-=-=-=-=')
        return

    def save(self, path):
        T.save(self.Q, path)
        return

    def load(self, path):
        self.base_model = T.load(path)
        self._clearQ()
        return
