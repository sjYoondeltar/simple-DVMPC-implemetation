
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import sys
sys.path.append(".")

from controller.value_net_utils import ValueNet

def main(dir_path, model_weights):
    
    with Path(dir_path).open('r') as f:
        params = json.load(f)["visualization"]
    
    tc = 30
    
    file_name = model_weights + ".pth"
    
    save_folder = Path(model_weights).parent.joinpath("plot")
    save_folder.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        
        load_value_dict = torch.load(file_name)
        input_dim = list(load_value_dict.values())[0].shape[1]
        
        value_net = ValueNet(input_dim, 256)
        value_net.load_state_dict(load_value_dict)
        
        a, b = np.meshgrid(
            np.arange(params["x_min"], params["x_max"], params["x_resolution"]),
            np.arange(params["y_min"], params["y_max"], params["y_resolution"])
        )
        
        if input_dim == 3:
            c = tc * np.ones_like(a.reshape(-1, 1))*(-np.pi/2)
            ab = np.concatenate([a.reshape([-1,1]), b.reshape([-1,1]), c], axis=1)
            
            plot_name = file_name.replace(".pth", ".png").replace(".", f".{tc:2.2f}.").split("/")[-1]
        
        else:
            ab = np.concatenate([a.reshape([-1,1]), b.reshape([-1,1])], axis=1)
            
            plot_name = file_name.replace(".pth", ".png").split("/")[-1]
    
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

