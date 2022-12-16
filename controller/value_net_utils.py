import random
import numpy as np
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


class EnsembleValueNet(nn.Module):
    def __init__(self, state_size, hidden_size, num_ensemble):
        super().__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2 * num_ensemble)
        self.fc3 = nn.Linear(hidden_size * 2 * num_ensemble, num_ensemble)

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


class Base_uni_neural(object):

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

    
    def reset_mu_std(self):

        self.du_mu = np.zeros((self.N, self.u_dim))
        self.du_mu[:-1, :] = self.du_mu[1:, :]
        self.du_std = self.du_max*np.ones((self.N, self.u_dim))
        self.du_std[:, 0] = 0.5*self.du_max[0]
        self.du_std[:, 1] = 0.5*self.du_max[1]
        
    
    def soft_target_update(self, critic, target_critic):
        for param, target_param in zip(critic.parameters(), target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
        
    def push_samples(self, transition):
        
        if self.use_time:
            
            self.buffer.push(transition)
            
        else:
            
            self.buffer.push((transition[0], transition[1], transition[2], transition[3], transition[4]))

        if self.n_step==1:
            
            self.sample_enough = True if len(self.buffer.memory) > self.exploration_step else False
            
        else:

            self.sample_enough = True if len(self.buffer.main_buffer.memory) > self.exploration_step else False
        

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

            self.clip_du(i)

            self.u_seq[i, :, :] = self.u_seq[i-1, :, :] + self.du_seq[i, :, :]

            self.clip_u(i)
        
        
    def calc_cost(self, states):
            
        self.cost_func.eval()
        
        return self.cost_func(states)
    
    
    def save_value_net(self, file_path):
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.save_dir / file_path
        torch.save(self.cost_func, str(save_path))
        
    
    def load_value_net(self):
        self.cost_func = torch.load(str(self.load_dir))
        
        
    def rs_pred_cost(self, x, u_seq, target, t):

        self.x_predict[0, :, :] = np.tile(x, (1, self.K))

        xt = np.tile(x, (1, self.K))
        
        tc = np.tile(t, (1, self.K))

        # cost_k = np.ones((self.N, self.K))
        
        cost_k = np.ones((self.K, ))
        
        for i, u in enumerate(u_seq.tolist()):
            
            # if i>0:
            #     cost_k[i, :] += cost_k[i-1, :] 

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
                cost_value_new = self.calc_cost(torch.Tensor(np.concatenate([x_new[:2, :], tc]).T).float().to(self.device))
                
            else:
                cost_value_new = self.calc_cost(torch.Tensor(x_new[:2, :].T).float().to(self.device))

            for k in range(self.K):
                    
                if self.coef_target_cost > 0.001:
                    diff = target.squeeze() - x_new[:, k].squeeze()[:2]
                    
                    x_rel = diff[0]
                    y_rel = diff[1]
                    
                    r_target = -np.exp(-np.square(x_rel)/100-np.square(y_rel)/100)
                    
                    # cost_k[i, k] += -cost_value_new[k][0] * (self.cost_gamma**(i+1)) + self.coef_target_cost * (self.cost_gamma**i+1) * r_target
                    
                    cost_k[k] += -cost_value_new[k][0] * (self.cost_gamma**(i+1)) + self.coef_target_cost * (self.cost_gamma**i+1) * r_target
                    
                else:
                    
                    # cost_k[i, k] += -cost_value_new[k][0] * (self.cost_gamma**(i+1))
                    
                    cost_k[k] += -cost_value_new[k][0] * (self.cost_gamma**(i+1))
                    
            xt = x_new

            self.x_predict[i+1, :, :] = xt

        return cost_k

