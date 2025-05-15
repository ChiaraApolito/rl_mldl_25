"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import gym
from env.custom_hopper import *

from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_learningcurves(file_path):
    df = pd.read_csv(file_path, skiprows=2, header=None, names=["r", "l", "t"])
    returns = df["r"].values

    window = 20
    smoothed = np.convolve(returns, np.ones(window) / window, mode="same")

    plt.figure(figsize=(8, 4))
    plt.plot(returns, label="Returns")
    plt.plot(smoothed, label=f"smoothed (w={window})")

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Learning Curve")
    plt.legend()
    plt.tight_layout()

    # Salva la figura nella directory desiderata
    if not os.path.exists('tlearning_curves'):
        os.makedirs('learning_curves', exist_ok=True)

    # Costruisci un nome file coerente in base al path del CSV
    base_name = os.path.basename(file_path).replace('.monitor.csv', '')
    plt.savefig(f"learning_curves/returns_plot_{base_name}.png")
    # plt.show()



def main():
    seed = 42
    train_env = gym.make('CustomHopper-source-v0')
    train_env.seed(seed)
    # Aggiungi il wrapper Monitor all'ambiente di train
    # per allenare e valutare un agente, è consigliato avvolgere l'ambiente con il Monitor wrapper, 
    # per evitare che venga modificata la durata degli episodi o le ricompense in modo non 
    # intenzionale da parte di altri wrapper
    train_env = Monitor(train_env, './ppo_hopper_logs/train_monitor', allow_early_resets=True)

    print('State space:', train_env.observation_space)  # state-space
    print('Action space:', train_env.action_space)  # action-space
    print('Dynamics parameters:', train_env.get_parameters())  # masses of each link of the Hopper

    #
    # TASK 4 & 5: train and test policies on the Hopper env with stable-baselines3
    #

    # Learning rate che va da 2.5e-4 a 0 durante il training
    lr_schedule = get_linear_fn(start=2.5e-4, end=0.0, end_fraction=1.0)

    # Callback per valutare la performance
    eval_env = gym.make('CustomHopper-source-v0')
    eval_env.seed(seed + 1)

    # Aggiungi il wrapper Monitor all'ambiente di valutazione
    eval_env = Monitor(eval_env, './ppo_hopper_logs/monitor', allow_early_resets=True)

    # Verifica che l’ambiente sia compatibile con Stable-Baselines3
    check_env(train_env)

    # Ogni eval_freq timesteps, il modello viene valutato.
    # Se la reward media è la migliore ottenuta finora, il modello viene salvato in ./ppo_hopper_logs/best_model.zip.
    # I risultati (media, deviazione standard, numero di episodi) vengono loggati in ./ppo_hopper_logs/
    eval_callback = EvalCallback(eval_env, 
                                 best_model_save_path='./ppo_hopper_logs/',
                                 log_path='./ppo_hopper_logs/', 
                                 eval_freq=5000,
                                 deterministic=True, 
                                 render=False)


    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        n_steps=2048,       # Numero di passi di simulazione raccolti prima di ogni aggiornamento della policy
        batch_size=64,      # Numero di esempi usati in ogni minibatch durante il training
        gae_lambda=0.90,    # Parametro per Generalized Advantage Estimation (GAE). Più vicino a 1 → meno bias, più varianza
        gamma=0.99,         # Fattore di sconto per le ricompense future
        n_epochs=10,        # Quante volte ogni batch di dati viene riutilizzato per aggiornare la policy 
        clip_range=0.2,     # Range di clipping per il rapporto tra policy attuale e precedente
        ent_coef=0.001,     # Coefficiente dell’entropia nella loss: Maggiore → più esplorazione
        vf_coef=0.5,        # Coefficiente per della value function nella loss
        max_grad_norm=0.5,  # Serve a prevenire problemi di esplosione del gradiente.
        learning_rate = lr_schedule   # Learning rate dinamico
    )

    model.learn(total_timesteps=300_000, callback=eval_callback)

    model.save("./ppo_hopper_final_model")

    mean_reward, std_reward = evaluate_policy(
        model, 
        env= eval_env, 
        n_eval_episodes=10, 
        deterministic=True, 
        render=False, 
        callback=None, 
        reward_threshold=None, 
        return_episode_rewards=False, 
        warn=True)
    
    # plot dei returns
    plot_learningcurves('./ppo_hopper_logs/train_monitor.monitor.csv')
    plot_learningcurves('./ppo_hopper_logs/monitor.monitor.csv')


if __name__ == '__main__':
    main()
