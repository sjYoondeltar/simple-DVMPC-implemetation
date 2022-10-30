import numpy as np
from pathlib import Path
import pandas as pd

class Recorder(object):

    def __init__(
        self,
        fieldnames,
        save_dir=Path("logs")
        ):
        
        self.save_dir = save_dir
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.n_episode = 0
        
        self.episode_memory = []
        self.results_memory = []
        
        self.fieldnames = fieldnames

        
    def record_episode(self, info):
        self.episode_memory.append((self.n_episode, *info))
        
    def record_results(self):
        self.n_episode += 1
        
    def save_logs(self, file_path):
        
        df = pd.DataFrame(self.episode_memory, columns=self.fieldnames)
        df.to_csv(str(self.save_dir / file_path), index=False)
        
        # with open(str(self.save_dir / file_path), 'w') as f:
        #     csv_writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            
        #     for rows in self.results_memory:
        #         csv_writer.writerows(rows)
                

if __name__ == '__main__':
    
    tmp_recorder = Recorder(fieldnames=['epi', 'tmp1', 'tmp2', 'tmp3'])
    
    # state, action, reward, is_collision, is_reach_goal, local_path
    
    # tmp_epi = 0
    # tmp_s = np.array([[-14.],
    #    [  0.],
    #    [  0.]], dtype=float32)
    # tmp_a = np.array([[2.], [0.38808072]])
    # tmp_r = -0.006770249999999998
    # tmp_m = 1, 
    tmp1 = np.array([1,2,3])
    tmp2 = np.array([4,5,6])
    tmp3 = np.array([7,8,9])
    
    tmp_recorder.record_episode((tmp1, tmp2, False))
    tmp_recorder.record_episode((tmp3, tmp2, True))
    
    tmp_recorder.record_results()
    
    tmp_recorder.record_episode((tmp3, tmp1, False))
    tmp_recorder.record_episode((tmp3, tmp3, False))
    
    tmp_recorder.record_results()
    
    tmp_recorder.save_logs('tmp.csv')
    
