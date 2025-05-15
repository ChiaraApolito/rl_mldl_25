"""Test two agent"""
import argparse

# import torch
import numpy as np

import gym
from env.custom_hopper import *

from stable_baselines3 import PPO


def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--model', default=None, type=str, help='Model path')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=10, type=int, help='Number of test episodes')
    
    return parser.parse_args()


def test_sb3_model(model_path, env_id, episodes=50, render=True):
    env = gym.make(env_id)
    model = PPO.load(model_path, env=env)
    
    returns = []
    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            if render:
                env.render()
        
        print(f"Episode {ep+1}: Return = {total_reward:.2f}")
        returns.append(total_reward)
        
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    print(f"Results on {env_id} with model {model_path}: Mean Return = {mean_return:.2f} ± {std_return:.2f}")
    env.close()
    return mean_return, std_return

def main():

	args = parse_args()
	
	test_cases = [
        ('source→source', './ppo_hopper_final_model_source.zip', 'CustomHopper-source-v0'),
        ('source→target', './ppo_hopper_final_model_source.zip', 'CustomHopper-target-v0'),
        ('target→target', './ppo_hopper_final_model_target.zip', 'CustomHopper-target-v0'),
    ]

	print(f"Running tests on fixed configurations with {args.episodes} episodes each\n")
     
	for label, model_path, env_id in test_cases:
         mean_ret, std_ret = test_sb3_model(model_path, env_id, episodes=args.episodes, render=args.render)
         print(f"{label} | Env: {env_id} | Mean Return: {mean_ret:.2f} ± {std_ret:.2f}")


if __name__ == '__main__':
	main()
