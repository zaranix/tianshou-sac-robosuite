from stable_baselines3 import SAC
from train import eval_success_rate  # adjust if your file has another name

# model = SAC.load("checkpoints/best_by_success_sb3.zip")
# sr = eval_success_rate(model, episodes=500)
# print(f"Best checkpoint success_rate over 500 eps: {sr:.3f}")
m=SAC.load('runs/Lift_SAC_SB3_seed0_h300_steps2000000_20260119-133559/checkpoints/best_by_success.zip')
print('Best checkpoint SR500 =', eval_success_rate(m, episodes=500, seed=999, horizon=300, control_freq=20, reward_shaping=True))