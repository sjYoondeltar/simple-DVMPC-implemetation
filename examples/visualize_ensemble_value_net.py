
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import sys
sys.path.append(".")

from controller.value_net_utils import EnsembleValueNet

def main(dir_path, model_weights):
    
    with Path(dir_path).open('r') as f:
        params = json.load(f)["visualization"]
    
    tc = 15
    
    load_folder = Path("ensemble_value_net")
    file_name = model_weights + ".pth"
    
    save_folder = Path("ensemble_value_net_plot")
    save_folder.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        
        load_value_dict = torch.load(str(load_folder / file_name))
        input_dim = list(load_value_dict.values())[0].shape[1]
        output_dim = list(load_value_dict.values())[-1].shape[0]
        
        value_net = EnsembleValueNet(input_dim, 256, output_dim)
        value_net.load_state_dict(load_value_dict)
        
        a, b = np.meshgrid(
            np.arange(params["x_min"], params["x_max"], params["x_resolution"]),
            np.arange(params["y_min"], params["y_max"], params["y_resolution"])
        )
        
        if input_dim == 3:
            c = tc * np.ones_like(a.reshape(-1, 1))*(-np.pi/2)
            ab = np.concatenate([a.reshape([-1,1]), b.reshape([-1,1]), c], axis=1)
    
            plot_mean_name = model_weights + f"_{tc:2.2f}_mean.png"
            plot_std_name = model_weights + f"_{tc:2.2f}_std.png"
        
        else:
            ab = np.concatenate([a.reshape([-1,1]), b.reshape([-1,1])], axis=1)
            
            plot_mean_name = model_weights + "_mean.png"
            plot_std_name = model_weights + "_std.png"
    
        results = np.zeros([output_dim, a.shape[0], a.shape[1]])
        
        preds_ab = value_net(torch.from_numpy(ab).float())
        
        for i in range(output_dim):
            results[i, :, :] = preds_ab[:, i].detach().numpy().reshape(a.shape)
        
        mean_results = np.mean(results, axis=0)
        std_results = np.std(results, axis=0)
        
    cp_m = plt.contourf(a, b, mean_results)

    plt.colorbar(cp_m)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.draw()
    plt.savefig(str(save_folder / plot_mean_name))
    
    plt.close()
    
    cp_s = plt.contourf(a, b, std_results)

    plt.colorbar(cp_s)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.draw()
    plt.savefig(str(save_folder / plot_std_name))
    
    plt.close()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Reach-Avoid')
    parser.add_argument('--params_dir', type=str, default='params/vis_params.json', help='directory of the parameters')
    parser.add_argument('--model_weights', type=str, help='directory of the parameters')
    
    args = parser.parse_args()
    dir_path = args.params_dir
    model_weights = args.model_weights
    
    main(dir_path, model_weights)
    
    print("Done.")

