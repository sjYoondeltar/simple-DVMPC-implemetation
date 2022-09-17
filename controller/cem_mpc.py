import ray
import numpy as np
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pathlib import Path

def continuous_dynamics_uni(state, u):

    theta = state[2, :]
    v = u[0, :]
    w = u[1, :]

    derivative = np.zeros_like(state)

    derivative[0, :] = v*np.cos(theta)
    derivative[1, :] = v*np.sin(theta)
    derivative[2, :] = w

    return derivative


def discrete_dynamics_uni(state, u, dt):
    state = state + continuous_dynamics_uni(state, u)*dt
    
    return state


class CEMMPC_uni_neural(object):

    def __init__(
        self,
        params,
        load_dir,
        use_time=True,
        lr=1e-3,
        tau=0.01,
        device=torch.device('cuda'),
        ):
                
        self.N = params["control"]["N"]
        self.K = params["control"]["K"]

        self.dt = params["control"]["dt"]
        self.x_dim = 3
        self.u_dim = 2
        self.u_ex = np.array([0.0, 0.0]).reshape([self.u_dim, -1])

        self.top_k = params["control"]["top_k"]
        self.iter = params["control"]["iter"]
        
        self.lr = lr

        self.gamma = 0.99
        self.cost_gamma = params["control"]["cost_gamma"]

        self.u_min= np.array(params["control"]["u_min"])
        self.u_min[1] = self.u_min[1]*np.pi/180
        self.u_max= np.array(params["control"]["u_max"])
        self.u_max[1] = self.u_max[1]*np.pi/180
        self.du_max = np.array(params["control"]["du_max"])
        self.du_max[1] = self.du_max[1]*np.pi/180

        self.du_mu = np.zeros((self.N, self.u_dim))
        self.du_std = np.ones((self.N, self.u_dim))

        self.reset_mu_std()
        
        self.x_predict = np.zeros((self.N+1, self.x_dim, self.K))
        
        self.n_step = 1
        
        self.buffer = ExperienceReplayMemory(buffer_size=params["control"]["buffer_size"])
        
        self.device = device
        
        self.use_time = use_time
        
        self.coef_target_cost = params["control"]["target_coef"]
        
        self.G=params["control"]["G"]
        
        self.sample_enough = False
        self.exploration_step = params["control"]["exploration_step"]
        self.minibatch_size = params["control"]["minibatch_size"]
        
        self.tau = tau
        
        self.eps = 1
        
        self.save_dir = Path(params["learning_process"]["save_dir"])
        self.load_dir = load_dir
        
        
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
        
    
    def soft_target_update(self, critic, target_critic):
        for param, target_param in zip(critic.parameters(), target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)


    def clip_u(self, n_idx):

        upper_bool = self.u_seq[n_idx, 0, :] > self.u_max[0]

        self.u_seq[n_idx, 0, upper_bool] = self.u_max[0]

        lower_bool = self.u_seq[n_idx, 0, :] <= self.u_min[0]

        self.u_seq[n_idx, 0, lower_bool] = self.u_min[0]

        upper_bool = self.u_seq[n_idx, 1, :] > self.u_max[1]

        self.u_seq[n_idx, 1, upper_bool] = self.u_max[1]

        lower_bool = self.u_seq[n_idx, 1, :] <= self.u_min[1]

        self.u_seq[n_idx, 1, lower_bool] = self.u_min[1]


    def clip_du(self, n_idx):

        upper_bool = self.du_seq[n_idx, 0, :] > self.du_max[0]

        self.du_seq[n_idx, 0, upper_bool] = self.du_max[0]

        lower_bool = self.du_seq[n_idx, 0, :] <= -self.du_max[0]

        self.du_seq[n_idx, 0, lower_bool] = -self.du_max[0]

        upper_bool = self.du_seq[n_idx, 1, :] > self.du_max[1]

        self.du_seq[n_idx, 1, upper_bool] = self.du_max[1]

        lower_bool = self.du_seq[n_idx, 1, :] <= -self.du_max[1]

        self.du_seq[n_idx, 1, lower_bool] = -self.du_max[1]


    def sample_u(self):
        
        self.u_seq = np.zeros((self.N, self.u_dim, self.K))
    
        self.du_seq = np.tile(self.du_mu.reshape([self.N, self.u_dim, -1]), (1, 1, self.K)) + \
            np.tile(self.du_std.reshape([self.N, self.u_dim, -1]), (1, 1, self.K)) * np.random.randn(self.N, self.u_dim, self.K)

        self.clip_du(0)

        self.u_seq[0, :, :] = np.tile(self.u_ex, (1, self.K)) + self.du_seq[0, :, :]

        self.clip_u(0)

        for i in range(1, self.N):

            self.clip_du(0)

            self.u_seq[i, :, :] = self.u_seq[i-1, :, :] + self.du_seq[i, :, :]

            self.clip_u(i)

    
    def reset_mu_std(self):

        self.du_mu = np.zeros((self.N, self.u_dim))
        self.du_mu[:-1, :] = self.du_mu[1:, :]
        self.du_std = self.du_max*np.ones((self.N, self.u_dim))
        self.du_std[:, 0] = 0.5*self.du_max[0]
        self.du_std[:, 1] = 0.5*self.du_max[1]

        
    def rs_pred_cost(self, x, u_seq, target, t):

        self.x_predict[0, :, :] = np.tile(x, (1, self.K))

        xt = np.tile(x, (1, self.K))
        
        tc = np.tile(t, (1, self.K))

        cost_k = np.ones((self.K, ))
        
        for i, u in enumerate(u_seq.tolist()):

            x_new = discrete_dynamics_uni(xt, np.array(u), self.dt)
            tc = tc + self.dt

            if x_new[2, 0] > np.pi:

                x_new[2, 0] -= 2*np.pi
            
            elif x_new[2, 0] < -np.pi:
            
                x_new[2, 0] += 2*np.pi

            else:

                pass
            
            #calc cost
            
            if self.use_time:
                cost_value_new = self.cost_func(torch.Tensor(np.concatenate([x_new[:2, :], tc]).T).float().to(self.device))
                
            else:
                cost_value_new = self.cost_func(torch.Tensor(x_new[:2, :].T).float().to(self.device))

            for k in range(self.K):
                    
                # cost_k[k] += -(1-self.coef_target_cost) * cost_value_new.reshape([1, ]) + self.coef_target_cost * (self.gamma**i+1)*np.sum(np.square(target.squeeze() - x_new[:, k].squeeze()[:2])) 
                if self.coef_target_cost > 0.001:
                    diff = target.squeeze() - x_new[:, k].squeeze()[:2]
                    
                    x_rel = diff[0]
                    y_rel = diff[1]
                    
                    diff_angle = np.arctan2(y_rel, x_rel)
                    
                    r_target = (1 - np.square(x_rel/20) - np.square(y_rel/20))
                
                    # cost_k[k] += -cost_value_new[k][0] * (self.gamma**(i+1)) + self.coef_target_cost * (self.gamma**i+1)*np.sum(np.square(diff_angle/3.14) + np.square(x_rel/10) + np.square(y_rel/10))
                    
                    cost_k[k] += -cost_value_new[k][0] * (self.cost_gamma**(i+1)) + self.coef_target_cost * (self.cost_gamma**i+1) * r_target
                    
                else:
                    
                    cost_k[k] += -cost_value_new[k][0] * (self.cost_gamma**(i+1))
                    
                    # cost_k[k] += -cost_value_new[k][0]

            xt = x_new

            self.x_predict[i+1, :, :] = xt

        return cost_k


    def cem_optimize(self, x_init, target, obs_pts, tc):

        for _ in range(self.iter):
            
            self.sample_u()

            err_pred = self.rs_pred_cost(x_init, self.u_seq, target, tc)

            self.du_mu = self.du_seq[:, :, np.argsort(err_pred)[:self.top_k]].mean(axis=2)

            self.du_std = self.du_seq[:, :, np.argsort(err_pred)[:self.top_k]].std(axis=2)

        self.u_ex = self.u_seq[0, :, np.argmin(err_pred)].reshape([self.u_dim, -1])

        self.reset_mu_std()

        return self.u_seq[:, :, np.argsort(err_pred)[:self.top_k]].mean(axis=2), self.x_predict[:, :, np.argsort(err_pred)[:self.top_k]]
    
    
    def push_samples(self, transition):
        
        if self.use_time:
            
            self.buffer.push(transition)
            
        else:
            
            self.buffer.push((transition[0], transition[1], transition[2], transition[3], transition[4]))

        if self.n_step==1:
            
            self.sample_enough = True if len(self.buffer.memory) > self.exploration_step else False
            
        else:

            self.sample_enough = True if len(self.buffer.main_buffer.memory) > self.exploration_step else False
    
    
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
        
    
    def save_value_net(self, file_path):
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.save_dir / file_path
        torch.save(self.cost_func.state_dict(), str(save_path))
        
    
    def load_value_net(self):
        self.cost_func.load_state_dict(torch.load(str(self.load_dir)))


class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super().__init__()

        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, states, actions):
        
        x1 = torch.cat([states, actions], dim=1)

        x1 = F.gelu(self.fc1(x1))

        x1 = F.gelu(self.fc2(x1))

        q_value = self.fc3(x1)

        return q_value


class ValueNet(nn.Module):
    def __init__(self, state_size, hidden_size):
        super().__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, states):
        
        x1 = F.leaky_relu(self.fc1(states))
        
        x1 = F.leaky_relu(self.fc2(x1))

        q_value = self.fc3(x1)

        return q_value


class ExperienceReplayMemory:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.buffer_size:
            self.memory.pop(0)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

class CEMMPC_uni_redq(CEMMPC_uni_neural):

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
        
        self.critic_list = []
        
        self.target_critic_list = []
        
        critic_params = []
        
        for _ in range(self.Ne):
        
            if self.use_time:
                
                cost_func = ValueNet(self.x_dim, 256).to(self.device)
                cost_func_target = ValueNet(self.x_dim, 256).to(self.device)
                
            else:
            
                cost_func = ValueNet(self.x_dim-1, 256).to(self.device)
                cost_func_target = ValueNet(self.x_dim-1, 256).to(self.device)
            
            cost_func_target.load_state_dict(cost_func.state_dict())
            
            self.critic_list.append(cost_func)
            self.target_critic_list.append(cost_func_target)
            
            critic_params = critic_params + list(cost_func.parameters())
            
        self.critic_optimizer = optim.Adam(
            critic_params, lr=self.lr, eps=1e-4)
        
        
    def pred_single_cost(self, idx_iqn, states, trainable=True):

        if trainable:
                
            cost = self.critic_list[idx_iqn](states)

        else:
            cost = self.target_critic_list[idx_iqn](states)

        return cost
    
        
    def pred_ensemble_cost(self, states):
        
        for idx_iqn in range(self.Ne):
            
            self.critic_list[idx_iqn].eval()
            
            cost = self.critic_list[idx_iqn](states)
            
            if idx_iqn == 0:
                
                cost_ensemble = cost
                
            else:
                
                cost_ensemble = torch.cat([cost_ensemble, cost], dim=1)
            
        mean_cost = cost_ensemble.mean(dim=1, keepdim=True)
        
        std_cost = cost_ensemble.std(dim=1, keepdim=True)
        
        cost_conservative = mean_cost - self.beta_cost * std_cost
        
        return cost_conservative
    
        
    def rs_pred_cost(self, x, u_seq, target, t):

        self.x_predict[0, :, :] = np.tile(x, (1, self.K))

        xt = np.tile(x, (1, self.K))
        
        tc = np.tile(t, (1, self.K))

        cost_k = np.ones((self.K, ))
        
        for i, u in enumerate(u_seq.tolist()):

            x_new = discrete_dynamics_uni(xt, np.array(u), self.dt)
            tc = tc + self.dt

            if x_new[2, 0] > np.pi:

                x_new[2, 0] -= 2*np.pi
            
            elif x_new[2, 0] < -np.pi:
            
                x_new[2, 0] += 2*np.pi

            else:

                pass
            
            #calc cost
            
            if self.use_time:
                cost_value_new = self.pred_ensemble_cost(torch.Tensor(np.concatenate([x_new[:2, :], tc]).T).float().to(self.device))
                
            else:
                cost_value_new = self.pred_ensemble_cost(torch.Tensor(x_new[:2, :].T).float().to(self.device))

            for k in range(self.K):
                    
                # cost_k[k] += -(1-self.coef_target_cost) * cost_value_new.reshape([1, ]) + self.coef_target_cost * (self.gamma**i+1)*np.sum(np.square(target.squeeze() - x_new[:, k].squeeze()[:2])) 
                if self.coef_target_cost > 0.001:
                    diff = target.squeeze() - x_new[:, k].squeeze()[:2]
                    
                    x_rel = diff[0]
                    y_rel = diff[1]
                    
                    diff_angle = np.arctan2(y_rel, x_rel)
                    
                    r_target = (1 - np.square(x_rel/20) - np.square(y_rel/20))
                    
                    cost_k[k] += -cost_value_new[k][0] * (self.cost_gamma**(i+1)) + self.coef_target_cost * (self.cost_gamma**i+1)*r_target
                    
                    # cost_k[k] += -cost_value_new[k][0] + self.coef_target_cost * r_target
                    
                else:
                    
                    cost_k[k] += -cost_value_new[k][0] * (self.cost_gamma**(i+1))
                    
                    # cost_k[k] += -cost_value_new[k][0]

            xt = x_new

            self.x_predict[i+1, :, :] = xt

        return cost_k
    
    
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
                
                for value_critic in self.critic_list:
                    value_critic.train()
                    
                    if self.use_time:
                    
                        value_trainable = value_critic(torch.cat([states[:, :2], ts], 1))
                        
                    else:
                    
                        value_trainable = value_critic(states[:, :2])
                        
                    loss = loss + criterion(value_trainable, target.detach())
                    
                self.critic_optimizer.zero_grad()
                loss.backward()
                self.critic_optimizer.step()
                        
                for critic, target_critic in zip(self.critic_list, self.target_critic_list):

                    self.soft_target_update(critic, target_critic)
                
            self.eps = np.maximum(self.eps*0.999, 0.001)
        
        else:
            
            pass
        
    
    def save_value_net(self, file_path):
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(self.Ne):
        
            save_path = self.save_dir / (file_path.replace('.pth', '_'+str(i)+'.pth'))
            torch.save(self.critic_list[i].state_dict(), str(save_path))
        
    
    def load_value_net(self):
        
        for i in range(self.Ne):
        
            self.critic_list[i].load_state_dict(torch.load(str(self.load_dir[i])))
        

