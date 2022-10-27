import numpy as np
from pathlib import Path
import csv

class Recorder(object):

    def __init__(
        self,
        save_dir=Path("logs"),
        ):
        
        self.save_dir = save_dir
        
        self.n_episode = 0
        
        self.episode_memory = []
        self.results_memory = []

        
    def record_episode(self, info):
        self.episode_memory.append((self.n_episode, info))
        
    def record_results(self):
        self.results_memory.append(self.episode_memory)
        self.episode_memory = []
        self.n_episode += 1
        
    def save_logs(self):
        
        with open(str(self.save_dir / 'tmp.csv'), 'w') as f:
            csv_writer = csv.writer(f)
            
            for rows in self.results_memory:
                csv_writer.writerows(rows)
                

if __name__ == '__main__':
    
    tmp_recorder = Recorder()
    
    # state, action, reward, is_collision, is_reach_goal, local_path
    
    tmp1 = np.array([1,2,3])
    tmp2 = np.array([4,5,6])
    tmp3 = np.array([7,8,9])
    
    tmp_recorder.record_episode((tmp1, tmp2, False))
    tmp_recorder.record_episode((tmp3, tmp2, True))
    
    tmp_recorder.record_results()
    
    tmp_recorder.record_episode((tmp3, tmp1, False))
    tmp_recorder.record_episode((tmp3, tmp3, False))
    
    tmp_recorder.record_results()
    
    tmp_recorder.save_logs()
    
