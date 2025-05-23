import itertools
from train_PPO import train_and_save  
import os

param_grid = {
    'n_steps': [],
    'learning_rate': [1e-3, 2e-4],
    'batch_size': [64, 128],
    'gae_lambda': [0.9, 0.95],
    'clip_range': [0.2, 0.3],
    'gamma':[],
    'n_epochs': [],
    'ent_coef':[]
}


keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]


for i, combo in enumerate(combinations):
    log_dir = f"./grid_logs/run_{i}"
    model_path = f"./grid_models/model_{i}"
    
    print(f"\n🔧 Training combo {i+1}/{len(combinations)}: {combo}")
    
    train_and_save(
        env_id='CustomHopper-source-v0',
        log_dir=log_dir,
        model_path=model_path,
        use_udr=True,  
        custom_params=combo  
    )