from stable_baselines3 import SAC
from train import eval_success_rate  # adjust if your file has another name

model = SAC.load("checkpoints/best_by_success_sb3.zip")
sr = eval_success_rate(model, episodes=500)
print(f"Best checkpoint success_rate over 200 eps: {sr:.3f}")
