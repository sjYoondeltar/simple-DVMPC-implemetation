
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
sys.path.append(".")

from controller.cem_mpc import ValueNet

def main():
    
    with Path('params/value_net.json').open('r') as f:
        params = json.load(f)["value_net_params"]
    
    use_time = params["visualization"]["USE_TIME"]
    tc = 30
    
    load_folder = Path(params["visualization"]["save_dir"])
    file_name = params["visualization"]["load_model"] + ".pth"
    
    save_folder = Path("value_net_plot")        
    save_folder.mkdir(parents=True, exist_ok=True)
    
    if use_time:
        plot_name = file_name.replace(".pth", ".png").replace(".", f".{tc:2.2f}.")
    
    else:
        plot_name = file_name.replace(".pth", ".png")
    
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
        
        else:
            ab = np.concatenate([a.reshape([-1,1]), b.reshape([-1,1])], axis=1)
    
        preds_ab = value_net(torch.from_numpy(ab).float())
    
    cp = plt.contourf(a, b, preds_ab.detach().numpy().reshape(a.shape))

    plt.colorbar(cp)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.draw()
    plt.savefig(str(save_folder / plot_name))
    

if __name__ == '__main__':
    main()
    
    print("Done.")

