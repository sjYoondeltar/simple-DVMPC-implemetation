import sys
sys.path.append(".")

import datetime
import numpy as np
import torch
import argparse
from pathlib import Path
from vehicle_env.recorder import Recorder
from common_utils import *


def test_process(rsmpc):
    
    reach_history = []

    for eps in range(train_episodes):
        
        done = False
        
        env.renew_init(params["environment"]["x_init"][int(eps%3)])
        
        rsmpc.reset_mu_std()

        x, target = env.reset()
        
        tc = 0.0
        
        u_ex = 0.0
        
        no_collision = True

        while not env.t_max_reach and not done :

            u_w, x_pred = rsmpc.optimize(x, target, obs_pts, tc)

            env.push_traj(np.transpose(x_pred, (2, 0, 1)))
            
            u = np.array(u_w[0, :]).reshape([-1, 1])

            xn, r, done = env.step(u)
            
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
            
        reach_history.append(float(env.reach))
        
        if len(reach_history) > params["learning_process"]["length_episode_history"]:
            reach_history.pop(0)
        else:
            pass
        
        if env.reach and np.mean(reach_history) > params["learning_process"]["threshold_success_rate"] and len(reach_history)==params["learning_process"]["length_episode_history"]:
            print(f"exiting the eps after reaching\n")
            break
        else:
            pass


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Reach-Avoid')
    parser.add_argument('--params_dir', type=str, default='params/ensemble_net_sparse.json', help='directory of the parameters')
    parser.add_argument('--ensemble', action='store_true', help='use ensemble')
    parser.add_argument('--seed', type=int, default=1234,
                        help='the seed number of numpy and torch (default: 1234)')
    parser.add_argument('--render', action='store_true', default=False,
                        help='render the environment on training or inference')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    ENSEMBLE=args.ensemble
    
    params = set_params(args.params_dir, ENSEMBLE)
    
    RENDER = True
    LOAD = True
    
    env, train_episodes, obs_pts = set_env(params)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rsmpc = set_ctrl(params, device, ENSEMBLE)
    
    test_process(rsmpc)
    
    
