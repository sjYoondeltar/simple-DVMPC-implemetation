import sys
sys.path.append(".")

import numpy as np
import torch
import json
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_params(dir_path, is_ensemble=True):
    
    with Path(dir_path).open('r') as f:
        
        if is_ensemble:
            params = json.load(f)["ensemble_value_net_params"]
        else:
            params = json.load(f)["value_net_params"]
    
    return params
    
    
def set_env(params):
    
    obs_list =[
        [-18.0, 0.0, 4.0, 40.0],
        [12.0, 12.0, 24.0, 18.0],
        [-10.0, 16.0, 20.0, 8.0],
        [-10.0, -16.0, 20.0, 8.0],
        [12.0, -12.0, 24.0, 18.0]
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


def set_ctrl(params, device, is_ensemble=True):
    
    if is_ensemble:
        
        if params["control"]["optimizer_type"] == "CEM":
            
            rsmpc = CEMMPC_uni_shered_redq(
                params=params,
                load_dir=Path(params["learning_process"]["save_dir"]) / (params["learning_process"]["load_model"] + '.pth'),
                use_time=params["learning_process"]["USE_TIME"],
                device=device,
            )
            
        else:
    
            rsmpc = MPPIMPC_uni_shered_redq(
                params=params,
                load_dir=Path(params["learning_process"]["save_dir"]) / (params["learning_process"]["load_model"] + '.pth'),
                use_time=params["learning_process"]["USE_TIME"],
                device=device,
            )
        
    else:
        
        if params["control"]["optimizer_type"] == "CEM":
            
            rsmpc = CEMMPC_uni_neural(
                params=params,
                load_dir=Path(params["learning_process"]["save_dir"]) / (params["learning_process"]["load_model"] + '.pth'),
                use_time=params["learning_process"]["USE_TIME"],
                device=device,
            )
            
        else:
            
            rsmpc = MPPIMPC_uni_neural(
                params=params,
                load_dir=Path(params["learning_process"]["save_dir"]) / (params["learning_process"]["load_model"] + '.pth'),
                use_time=params["learning_process"]["USE_TIME"],
                device=device,
            )
    
    rsmpc.build_value_net()
    
    if params["learning_process"]["LOAD"]:
        rsmpc.load_value_net()
        
    return rsmpc
