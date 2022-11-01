
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


class RewardPlotter(object):

    def __init__(
        self,
        save_dir=Path("plot")
        ):
        
        self.save_dir = save_dir
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        
    def calc_results(self, file_path):
        
        df = pd.read_csv(file_path)
        
        episode_array = np.array(df['episode'])
        reward_array = np.array(df['r'])
        reach_array = np.array(df['reach']).astype(np.float32)
        collision_array = np.array(df['reach'])
        
        epi_f = np.max(episode_array) + 1
        
        rewards_per_episode = []
        is_reach_per_episode = []
        
        for epi in range(epi_f):
            rewards_per_episode.append(np.sum(reward_array[episode_array == epi]))
            is_reach_per_episode.append(np.sum(reach_array[episode_array == epi]))
        
        return rewards_per_episode, is_reach_per_episode
    
        
    def save_reward_plot(self, file_path_list):
        
        fig = plt.figure(figsize=(20, 10))
        
        ax1 = fig.add_subplot(2, 1, 1)
        
        ax2 = fig.add_subplot(2, 1, 2)
        
        for file_path in file_path_list:
        
            rewards_per_episode, is_reach_per_episode = self.calc_results(file_path)
            
            model_name_list = file_path.split('/')
            
            ax1.plot(rewards_per_episode, label=model_name_list[1] + "+" + model_name_list[2])
        
            ax2.plot(is_reach_per_episode)
        
        ax1.set_ylabel("cumulative rewards")
        
        ax1.grid()
        ax1.legend(loc='upper left')
        ax1.set_xlim([0, 40])
        
        ax2.set_ylabel("success to reach a goal")
        ax2.set_xlabel("episode")
        ax2.set_xlim([0, 40])
        ax2.grid()
        
        plt.draw()
        plt.savefig(self.save_dir / "reward_plot.png")
        plt.close()


if __name__ == '__main__':
    
    reward_plot_draw = RewardPlotter(save_dir=Path("runs/comparison"))
    
    logs_list = [
        "runs/value_net/cem_dense/logs/20221030_222322.csv",
        # "runs/value_net/cem_sparse/logs/20221031_225509.csv",
        "runs/ensemble_value_net/mppi_dense/logs/20221031_005341.csv",
        # "runs/ensemble_value_net/mppi_sparse/logs/20221031_132405.csv"
    ]
    
    reward_plot_draw.save_reward_plot(logs_list)
    
