
import numpy as np
from pathlib import Path
import pandas as pd


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
    
        
    def save_reward(self):
        
        episode_array = np.array(self.df['episode'])
        reward_array = np.array(self.df['r'])
        
        epi_f = np.max(episode_array) + 1
        
        for epi in range(epi_f):
            self.episode_rewards.append(np.sum(reward_array[episode_array == epi]))



if __name__ == '__main__':
    
    # df = pd.read_csv("runs/value_net/cem_dense/logs/20221030_175756.csv")
    
    reward_plot_draw = RewardPlotter(save_dir=Path("runs/value_net/cem_dense/plot"))
    
    rewards_per_episode, is_reach_per_episode = reward_plot_draw.calc_results("runs/value_net/cem_dense/logs/20221030_175756.csv")
    
    print(rewards_per_episode)
    
    print(is_reach_per_episode)
    