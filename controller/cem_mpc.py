import ray
import numpy as np
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pathlib import Path
from controller.value_net_utils import *


class CEMMPC_uni_neural(Base_uni_neural):

    def __init__(
        self,
        params,
        load_dir,
        use_time=True,
        lr=1e-3,
        tau=0.01,
        device=torch.device('cuda'),
        ):
        
        super().__init__(
            params,
            load_dir,
            use_time,
            lr,
            tau,
            device
        )

        self.top_k = params["control"]["top_k"]
        self.iter = params["control"]["iter"]


    def optimize(self, x_init, target, obs_pts, tc):

        for _ in range(self.iter):
            
            self.sample_u()

            err_pred = self.rs_pred_cost(x_init, self.u_seq, target, tc)

            self.du_mu = self.du_seq[:, :, np.argsort(err_pred)[:self.top_k]].mean(axis=2)

            self.du_std = self.du_seq[:, :, np.argsort(err_pred)[:self.top_k]].std(axis=2)

        self.u_ex = self.u_seq[0, :, np.argmin(err_pred)].reshape([self.u_dim, -1])

        self.reset_mu_std()

        return self.u_seq[:, :, np.argsort(err_pred)[:self.top_k]].mean(axis=2), self.x_predict[:, :, np.argsort(err_pred)[:self.top_k]]
    


class CEMMPC_uni_shered_redq(CEMMPC_uni_neural):

    def __init__(
        self,
        params,
        load_dir,
        use_time=True,
        lr=1e-3,
        tau=0.01,
        device=torch.device('cuda'),
        ):
        
        super().__init__(
            params,
            load_dir,
            use_time,
            lr,
            tau,
            device
        )
        
        self.Ne = params["control"]["Ne"]
        self.beta_cost = params["control"]["beta"]
        
        
    def build_value_net(self):
        
        if self.use_time:
        
            self.cost_func = EnsembleValueNet(self.x_dim, 256, self.Ne).to(self.device)
            self.cost_func_target = EnsembleValueNet(self.x_dim, 256, self.Ne).to(self.device)
        
        else:
        
            self.cost_func = EnsembleValueNet(self.x_dim-1, 256, self.Ne).to(self.device)
            self.cost_func_target = EnsembleValueNet(self.x_dim-1, 256, self.Ne).to(self.device)
        
        self.cost_func_target.load_state_dict(self.cost_func.state_dict())
        
        self.critic_optimizer = optim.Adam(
            list(self.cost_func.parameters()), lr=self.lr, eps=1e-4)
        
        
    def pred_single_cost(self, idx_iqn, states, trainable=True):

        if trainable:
                
            cost = self.cost_func(states)[:, idx_iqn].view([-1, 1])

        else:
            cost = self.cost_func_target(states)[:, idx_iqn].view([-1, 1])

        return cost
    
        
    def calc_cost(self, states):
            
        self.cost_func.eval()
        
        cost_ensemble = self.cost_func(states)
        
        mean_cost = cost_ensemble.mean(dim=1, keepdim=True)
        
        std_cost = cost_ensemble.std(dim=1, keepdim=True)
        
        cost_conservative = mean_cost - self.beta_cost * std_cost
        
        return cost_conservative
    
    
    def train(self):
        
        if self.sample_enough:
            
            for g_update in range(self.G):
                
                if self.n_step==1:
                    
                    mini_batch = self.buffer.sample(self.minibatch_size)
                    
                else:
                
                    mini_batch = self.buffer.main_buffer.sample(self.minibatch_size)
                        
                mini_batch = np.array(mini_batch, dtype=object)
                states = torch.Tensor(np.vstack(mini_batch[:, 0]).reshape([self.minibatch_size, -1])).float().to(self.device)
                actions = list(mini_batch[:, 1])
                rewards = torch.Tensor(list(mini_batch[:, 2])).float().to(self.device)
                next_states = torch.Tensor(np.vstack(mini_batch[:, 3]).reshape([self.minibatch_size, -1])).float().to(self.device)
                masks = torch.Tensor(list(mini_batch[:, 4])).float().to(self.device)
                
                idx = np.random.choice(self.Ne, 2, replace=False)
                
                if self.use_time:
                    
                    ts = torch.Tensor(list(mini_batch[:, 5])).float().to(self.device).view([self.minibatch_size, 1])
                    
                else:
                    
                    pass
                
                if self.use_time:
                    value_target_1 = self.pred_single_cost(idx[0], torch.cat([next_states[:, :2], ts+self.dt], 1), trainable=False)
                    value_target_2 = self.pred_single_cost(idx[1], torch.cat([next_states[:, :2], ts+self.dt], 1), trainable=False)
                
                else:
                    value_target_1 = self.pred_single_cost(idx[0], next_states[:, :2], trainable=False)
                    value_target_2 = self.pred_single_cost(idx[1], next_states[:, :2], trainable=False)
                
                min_value_target = torch.min(value_target_1, value_target_2)
                
                if self.n_step == 1:

                    target = rewards.view([-1,1]) + masks.view([-1,1]) * self.gamma * min_value_target
                
                else:
                    
                    target = rewards.view([-1,1]) + masks.view([-1,1]) * (self.gamma**self.n_step) * min_value_target
                
                loss = 0
                criterion = torch.nn.MSELoss()
                
                self.cost_func.train()
                
                if self.use_time:
                
                    value_trainable = self.cost_func(torch.cat([states[:, :2], ts], 1))
                    
                else:
                
                    value_trainable = self.cost_func(states[:, :2])
                
                for ne in range(self.Ne):
                    
                    loss = loss + criterion(value_trainable[:, ne].view([-1,1]), target.detach())
                    
                loss = loss / self.Ne
                
                self.critic_optimizer.zero_grad()
                loss.backward()
                self.critic_optimizer.step()
                
                self.soft_target_update(self.cost_func, self.cost_func_target)
                
            self.eps = np.maximum(self.eps*0.999, 0.001)
        
        else:
            
            pass
        
