import sys
sys.path.append(".")

import time
import numpy as np
import torch
import json
import argparse
import random
from pathlib import Path
from vehicle_env.navi_maze_env_car import NAVI_ENV
from controller.cem_mpc import CEMMPC_uni_neural, CEMMPC_uni_shered_redq
from controller.mppi_mpc import MPPIMPC_uni_neural, MPPIMPC_uni_shered_redq


def set_seed(seed):
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def set_params(dir_path):
    
    with Path(dir_path).open('r') as f:
        
        if ENSEMBLE:
            params = json.load(f)["ensemble_value_net_params"]
        else:
            params = json.load(f)["value_net_params"]
    
    return params
    
    
def set_env(params):
    
    obs_list =[
        [0.0, 2.0, 4.0, 4.0],
        [9.0, -2.0, 4.0, 4.0],
        [-10.0, -2.0, 8.0, 4.0],
        # [-16.0, 0.0, 10.0, 8.0],
        [16.0, 0.0, 10.0, 8.0],
        [0.0, 12.0, 40.0, 16.0],
        [0.0, -12.0, 40.0, 16.0]
    ]
    
    obs_pts = np.array([[
        [obs[0]-obs[2]/2, obs[1]-obs[3]/2],
        [obs[0]+obs[2]/2, obs[1]-obs[3]/2],
        [obs[0]+obs[2]/2, obs[1]+obs[3]/2],
        [obs[0]-obs[2]/2, obs[1]+obs[3]/2],
        ] for obs in obs_list])
    
    train_episodes = params["learning_process"]["train_episodes"]
    
    env = NAVI_ENV(
        x_init=params["environment"]["x_init"],
        t_max=params["environment"]["t_max"],
        dT=0.1,
        u_min=[0, -np.pi/4],
        u_max=[2, np.pi/4],
        reward_type=params["environment"]["reward_type"],
        target_fix=params["environment"]["target"],
        level=2, obs_list=obs_list,
        coef_dis=params["environment"]["coef_dis"],
        coef_angle=params["environment"]["coef_angle"],
        terminal_cond=1,
        collision_panalty=params["environment"]["collision_panalty"],
        goal_reward=params["environment"]["goal_reward"],
        continue_after_collision=params["environment"]["continue_after_collision"]
    )
    
    return env, train_episodes, obs_pts


def set_ctrl(params):
    
    if ENSEMBLE:
        
        # if params["control"]["optimizer_type"] == "CEM":
            
        #     rsmpc = CEMMPC_uni_redq(
        #         params=params,
        #         load_dir=[Path(params["learning_process"]["save_dir"]) / Path(params["learning_process"]["load_model"] + f"_{i}.pth") for i in range(params["control"]["Ne"])],
        #         use_time=USE_TIME,
        #         device=device,
        #     )
            
        # else:
    
        #     rsmpc = MPPIMPC_uni_redq(
        #         params=params,
        #         load_dir=[Path(params["learning_process"]["save_dir"]) / Path(params["learning_process"]["load_model"] + f"_{i}.pth") for i in range(params["control"]["Ne"])],
        #         use_time=USE_TIME,
        #         device=device,
        #     )
        
        if params["control"]["optimizer_type"] == "CEM":
            
            rsmpc = CEMMPC_uni_shered_redq(
                params=params,
                load_dir=Path(params["learning_process"]["save_dir"]) / (params["learning_process"]["load_model"] + '.pth'),
                use_time=USE_TIME,
                device=device,
            )
            
        else:
    
            rsmpc = MPPIMPC_uni_shered_redq(
                params=params,
                load_dir=Path(params["learning_process"]["save_dir"]) / (params["learning_process"]["load_model"] + '.pth'),
                use_time=USE_TIME,
                device=device,
            )
        
    else:
        
        if params["control"]["optimizer_type"] == "CEM":
            
            rsmpc = CEMMPC_uni_neural(
                params=params,
                load_dir=Path(params["learning_process"]["save_dir"]) / (params["learning_process"]["load_model"] + '.pth'),
                use_time=USE_TIME,
                device=device,
            )
            
        else:
            
            rsmpc = MPPIMPC_uni_neural(
                params=params,
                load_dir=Path(params["learning_process"]["save_dir"]) / (params["learning_process"]["load_model"] + '.pth'),
                use_time=USE_TIME,
                device=device,
            )
    
    rsmpc.build_value_net()
    
    if LOAD:
        rsmpc.load_value_net()
        
    return rsmpc

def train_process(rsmpc):
    
    reach_history = []

    for eps in range(train_episodes):
        
        done = False

        x, target = env.reset()
        
        tc = 0.0
        
        u_ex = 0.0
        
        no_collision = True

        while not env.t_max_reach and not done :

            u_w, x_pred = rsmpc.optimize(x, target, obs_pts, tc)

            env.push_traj(np.transpose(x_pred, (2, 0, 1)))

            u = np.array(u_w[0, :]).reshape([-1, 1])
            
            if np.random.rand(1)<rsmpc.eps*float(EXPLORE):
                u[1][0] = np.random.rand(1) * (rsmpc.u_max[1]-rsmpc.u_min[1]) + rsmpc.u_min[1]
                
            else:
                
                pass

            xn, r, done = env.step(u)
            
            mask = 1 if not done else 0
            
            rsmpc.push_samples((x, u, r, xn, mask, tc))
            
            rsmpc.train()
            
            if RENDER:
                env.render()
            
            if env.reach:
                print(f"reach at {env.t}\n")
                break

            x = xn
            tc = tc + env.dT
            u_ex = u[1][0]
            
            if no_collision and (env.wall_contact or env.obs_contact):
                no_collision = False
                
        if no_collision:
            print(f"{eps:3d}th episode finished at {env.t} steps with no collision and stopped {env.dist:.4f} from targets\n")
        else:
            print(f"{eps:3d}th episode finished at {env.t} steps with collisions and stopped {env.dist:.4f} from targets\n")
            
        reach_history.append(float(no_collision))
        
        if len(reach_history) > params["learning_process"]["length_episode_history"]:
            reach_history.pop(0)
        else:
            pass
        
        if env.reach and np.mean(reach_history) > params["learning_process"]["threshold_success_rate"] and len(reach_history)==params["learning_process"]["length_episode_history"]:
            print(f"exiting the eps after reaching\n")
            rsmpc.save_value_net(f"value_net_{eps:03d}.pth")
            break
        else:
            pass
    
    if np.mean(reach_history) <= params["learning_process"]["threshold_success_rate"]:
        rsmpc.save_value_net(f"value_net_end_{eps:03d}.pth")
    else:
        pass


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Reach-Avoid')
    parser.add_argument('--params_dir', type=str, default='params/value_net_cem.json', help='directory of the parameters')
    parser.add_argument('--ensemble', action='store_true', help='use ensemble')
    parser.add_argument('--seed', type=int, default=1234,
                        help='the seed number of numpy and torch (default: 1234)')
    parser.add_argument('--render', action='store_true', default=False,
                        help='render the environment on training or inference')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    ENSEMBLE=args.ensemble
    
    params = set_params(args.params_dir)
    
    RENDER = params["learning_process"]["RENDER"]
    LOAD = params["learning_process"]["LOAD"]
    EXPLORE = params["learning_process"]["EXPLORE"]
    USE_TIME = params["learning_process"]["USE_TIME"]
    
    env, train_episodes, obs_pts = set_env(params)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rsmpc = set_ctrl(params)
    
    train_process(rsmpc)
    
    
