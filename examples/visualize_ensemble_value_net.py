
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
sys.path.append(".")

from controller.cem_mpc import ValueNet

def main():
    
    with Path('params/ensemble_net.json').open('r') as f:
        params = json.load(f)["ensemble_value_net_params"]
    
    use_time = params["visualization"]["USE_TIME"]
    tc = 15
    
    load_folder = Path(params["visualization"]["save_dir"])
    file_single_name = params["visualization"]["load_model"]
    file_name_list = [file_single_name + f"_{i}.pth" for i in range(5)]
    
    save_folder = Path("ensemble_value_net_plot")
    save_folder.mkdir(parents=True, exist_ok=True)
    
    if use_time:
    
        plot_mean_name = file_single_name + f"_{tc:2.2f}_mean.png"
        plot_std_name = file_single_name + f"_{tc:2.2f}_std.png"
        
    else:
        
        plot_mean_name = file_single_name + "_mean.png"
        plot_std_name = file_single_name + "_std.png"
    
    with torch.no_grad():
        
        ensemble_value_list = []
        
        for file_name in file_name_list:
            if use_time:
                value_net = ValueNet(3, 256)
            else:
                value_net = ValueNet(2, 256)
            value_net.load_state_dict(torch.load(str(load_folder / file_name)))
            ensemble_value_list.append(value_net)
        
        a, b = np.meshgrid(
            np.arange(params["visualization"]["x_min"], params["visualization"]["x_max"], params["visualization"]["x_resolution"]),
            np.arange(params["visualization"]["y_min"], params["visualization"]["y_max"], params["visualization"]["y_resolution"])
        )
        
        if use_time:
            c = tc * np.ones_like(a.reshape(-1, 1))*(-np.pi/2)
            ab = np.concatenate([a.reshape([-1,1]), b.reshape([-1,1]), c], axis=1)
        
        else:
            ab = np.concatenate([a.reshape([-1,1]), b.reshape([-1,1])], axis=1)
    
        results = np.zeros([len(ensemble_value_list), a.shape[0], a.shape[1]])
        
        for i, value_net in enumerate(ensemble_value_list):
            preds_ab = value_net(torch.from_numpy(ab).float())
            results[i, :, :] = preds_ab.detach().numpy().reshape(a.shape)
        
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
    main()
    
    print("Done.")

