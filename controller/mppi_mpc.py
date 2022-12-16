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


class MPPIMPC_uni_neural(Base_uni_neural):

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
        
        self.cost_lambda = 0.1
        
        
    def build_value_net(self):
        
        if self.use_time:
        
            self.cost_func = ValueNet(self.x_dim, 256).to(self.device)
            self.cost_func_target = ValueNet(self.x_dim, 256).to(self.device)
            
        else:
        
            self.cost_func = ValueNet(self.x_dim-1, 256).to(self.device)
            self.cost_func_target = ValueNet(self.x_dim-1, 256).to(self.device)
        
        self.cost_func_target.load_state_dict(self.cost_func.state_dict())
        
        self.critic_optimizer = optim.Adam(
            list(self.cost_func.parameters()), lr=self.lr, eps=1e-4)


    def clip_du(self, n_idx, du_seq=None):
        
        if du_seq is None:

            upper_bool = self.du_seq[n_idx, 0, :] > self.du_max[0]

            self.du_seq[n_idx, 0, upper_bool] = self.du_max[0]

            lower_bool = self.du_seq[n_idx, 0, :] <= -self.du_max[0]

            self.du_seq[n_idx, 0, lower_bool] = -self.du_max[0]

            upper_bool = self.du_seq[n_idx, 1, :] > self.du_max[1]

            self.du_seq[n_idx, 1, upper_bool] = self.du_max[1]

            lower_bool = self.du_seq[n_idx, 1, :] <= -self.du_max[1]

            self.du_seq[n_idx, 1, lower_bool] = -self.du_max[1]
            
        else:
            
            upper_bool = du_seq[n_idx, 0, :] > self.du_max[0]

            du_seq[n_idx, 0, upper_bool] = self.du_max[0]

            lower_bool = du_seq[n_idx, 0, :] <= -self.du_max[0]

            du_seq[n_idx, 0, lower_bool] = -self.du_max[0]

            upper_bool = du_seq[n_idx, 1, :] > self.du_max[1]

            du_seq[n_idx, 1, upper_bool] = self.du_max[1]

            lower_bool = du_seq[n_idx, 1, :] <= -self.du_max[1]

            du_seq[n_idx, 1, lower_bool] = -self.du_max[1]
            
            return du_seq


    def optimize(self, x_init, target, obs_pts, tc):

        self.sample_u()

        err_pred = self.rs_pred_cost(x_init, self.u_seq, target, tc)
        
        self.u_seq = np.tile(self.u_ex, (self.N, 1)).reshape([self.N, self.u_dim, -1])
        
        for i in range(self.N):
            
            max_err = np.max(err_pred)
        
            weights_cost = np.exp(-self.cost_lambda*(err_pred-max_err)) / np.sum(np.exp(-self.cost_lambda*(err_pred-max_err)))

            self.du_mu[i, :] = (weights_cost * self.du_seq[i, :, :]).sum(axis=1)
        
            du_w_seq = np.copy(self.du_mu[i, :]).reshape([1, self.u_dim, -1])

            du_w_seq = self.clip_du(0, du_w_seq)
            
            if i==0:
                self.u_seq[i, :, :] += du_w_seq[0, :, :]
                
            else:
                self.u_seq[i, :, :] = self.u_seq[i-1, :, :] + du_w_seq[0, :, :]
                
            self.clip_u(i)
        
        u_final = self.u_seq

        self.u_ex = self.u_seq[0, :].reshape([self.u_dim, -1])

        self.reset_mu_std()

        return u_final, self.x_predict[:, :, np.argsort(weights_cost)[-1]].reshape([-1, self.x_dim, 1])
    
    
    def train(self):
        
        if self.sample_enough:
            
            for _ in range(self.G):
        
                criterion = torch.nn.MSELoss()
                
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
                
                if self.use_time:
                    
                    ts = torch.Tensor(list(mini_batch[:, 5])).float().to(self.device).view([self.minibatch_size, 1])
                    
                else:
                    
                    pass
                
                self.cost_func.train()
                self.cost_func_target.eval()
                
                if self.use_time:
                    value_trainable = self.cost_func(torch.cat([states[:, :2], ts], 1))
                    value_target = self.cost_func_target(torch.cat([next_states[:, :2], ts+self.dt], 1))
                
                else:
                    value_trainable = self.cost_func(states[:, :2])
                    value_target = self.cost_func_target(next_states[:, :2])
                
                
                if self.n_step == 1:

                    target = rewards.view([-1,1]) + masks.view([-1,1]) * self.gamma * value_target
                
                else:
                    
                    target = rewards.view([-1,1]) + masks.view([-1,1]) * (self.gamma**self.n_step) * value_target
                    
                loss = criterion(value_trainable, target.detach())
                self.critic_optimizer.zero_grad()
                loss.backward()
                self.critic_optimizer.step()
                        
                self.soft_target_update(self.cost_func, self.cost_func_target)
                
            self.eps = np.maximum(self.eps*0.999, 0.001)
        
        else:
            
            pass


class MPPIMPC_uni_shered_redq(MPPIMPC_uni_neural):

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
        
