
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import sys
sys.path.append(".")

from controller.cem_mpc import ValueNet

def main(dir_path, model_weights):
    
    with Path(dir_path).open('r') as f:
        params = json.load(f)
    
    use_time = params["visualization"]["USE_TIME"]
    tc = 30
    
    load_folder = Path("value_net")
    file_name = model_weights + ".pth"
    
    save_folder = Path("value_net_plot")        
    save_folder.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        
        if use_time:
            value_net = ValueNet(3, 256)
        else:
            value_net = ValueNet(2, 256)
        
        value_net.load_state_dict(torch.load(str(load_folder / file_name)))
        
        a, b = np.meshgrid(
            np.arange(params["visualization"]["x_min"], params["visualization"]["x_max"], params["visualization"]["x_resolution"]),
            np.arange(params["visualization"]["y_min"], params["visualization"]["y_max"], params["visualization"]["y_resolution"])
        )
        
        if use_time:
            c = tc * np.ones_like(a.reshape(-1, 1))*(-np.pi/2)
            ab = np.concatenate([a.reshape([-1,1]), b.reshape([-1,1]), c], axis=1)
            
            plot_name = file_name.replace(".pth", ".png").replace(".", f".{tc:2.2f}.")
        
        else:
            ab = np.concatenate([a.reshape([-1,1]), b.reshape([-1,1])], axis=1)
            
            plot_name = file_name.replace(".pth", ".png")
    
        preds_ab = value_net(torch.from_numpy(ab).float())
    
    cp = plt.contourf(a, b, preds_ab.detach().numpy().reshape(a.shape))

    plt.colorbar(cp)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.draw()
    plt.savefig(str(save_folder / plot_name))
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Reach-Avoid')
    parser.add_argument('--params_dir', type=str, default='params/vis_params.json', help='directory of the parameters')
    parser.add_argument('--model_weights', type=str, help='directory of the parameters')
    
    args = parser.parse_args()
    dir_path = args.params_dir
    model_weights = args.model_weights
    
    main(dir_path, model_weights)
    
    print("Done.")

