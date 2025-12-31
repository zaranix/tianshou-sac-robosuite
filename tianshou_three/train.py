import argparse

import os

import datetime

import numpy as np

import torch

import gymnasium as gym

from gymnasium import spaces

from gymnasium.wrappers import NormalizeReward, TransformReward

from torch.utils.tensorboard import SummaryWriter



import robosuite as suite

from robosuite.wrappers import GymWrapper



from tianshou.data import Collector, VectorReplayBuffer

from tianshou.env import SubprocVectorEnv, VectorEnvNormObs

from tianshou.policy import SACPolicy

from tianshou.trainer import OffpolicyTrainer

from tianshou.utils import TensorboardLogger

from tianshou.utils.net.common import Net

from tianshou.utils.net.continuous import ActorProb, Critic



# =============================================================================

# 1. Custom Robosuite Wrapper (FIXED)

# =============================================================================

class TianshouRobosuiteWrapper(gym.Wrapper):

    """

    Wraps Robosuite to be fully compatible with Gymnasium and Tianshou.

    Casts all observations to Float32.

    """

    def __init__(self, env):

        super().__init__(env)

        assert isinstance(env.observation_space, spaces.Box), \
            "Environment must be wrapped with robosuite.wrappers.GymWrapper first."



        self.observation_space = spaces.Box(

            low=env.observation_space.low,

            high=env.observation_space.high,

            dtype=np.float32

        )

        self.action_space = spaces.Box(

            low=env.action_space.low,

            high=env.action_space.high,

            dtype=np.float32

        )



    def reset(self, seed=None, options=None):

        obs, info = self.env.reset(seed=seed, options=options)

        return obs.astype(np.float32), info



    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)

        return obs.astype(np.float32), reward, terminated, truncated, info



def make_robosuite_env(task_name, seed=0, training=True, video_record=False):

    # 1. Configuration

    has_offscreen = True if video_record else False

    use_camera = True if video_record else False

    

    env = suite.make(

        task_name,

        horizon=500,           # <--- CRITICAL: 500 steps gives enough time to lift

        robots="Panda",

        use_camera_obs=use_camera,

        has_offscreen_renderer=has_offscreen, 

        has_renderer=False,

        reward_shaping=True, 

        control_freq=20,

    )



    # 2. Key Selection (Standard Bundle)

    keys = ["robot0_proprio-state", "object-state", "gripper_to_cube_pos"]

    env = GymWrapper(env, keys=keys)

    

    # 3. Type Casting

    env = TianshouRobosuiteWrapper(env)



    # 4. Reward Normalization (CRITICAL FIX)

    # We only normalize rewards during training. 

    # For testing, we want the RAW score to see if it actually solved the task.

    if training:

        env = NormalizeReward(env)

        # Clip reward to prevent instability

        env = TransformReward(env, lambda r: np.clip(r, -10.0, 10.0))



    return env



# =============================================================================

# 2. Arguments

# =============================================================================

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='Lift')

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    

    # Visualization Arguments

    parser.add_argument('--watch', action='store_true', help='Watch/Visualize the agent instead of training')

    parser.add_argument('--resume-path', type=str, default=None, help='Path to .pth model for visualization')



    # Model Hyperparameters

    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[256, 256])

    parser.add_argument('--actor-lr', type=float, default=1e-4)

    parser.add_argument('--critic-lr', type=float, default=3e-4)

    parser.add_argument('--alpha-lr', type=float, default=3e-4) 

    parser.add_argument('--gamma', type=float, default=0.99)

    parser.add_argument('--tau', type=float, default=0.005)

    parser.add_argument('--alpha', type=float, default=0.2)

    parser.add_argument('--auto-alpha', default=True, action='store_true')



    # Training Hyperparameters

    parser.add_argument('--epoch', type=int, default=200)

    parser.add_argument('--step-per-epoch', type=int, default=5000)

    parser.add_argument('--step-per-collect', type=int, default=10)

    parser.add_argument('--update-per-step', type=float, default=1.0) # Increased update ratio

    parser.add_argument('--batch-size', type=int, default=256)

    parser.add_argument('--buffer-size', type=int, default=1000000)

    parser.add_argument('--training-num', type=int, default=4)  # Reduced to 4 for stability

    parser.add_argument('--test-num', type=int, default=10)

    parser.add_argument('--start-timesteps', type=int, default=10000) 



    parser.add_argument('--logdir', type=str, default='log')

    parser.add_argument('--save-interval', type=int, default=1) 



    return parser.parse_args()



# =============================================================================

# 3. Training Loop

# =============================================================================

def train_sac(args):

    # Note: We pass training=True/False to apply Reward Normalization correctly

    train_envs = SubprocVectorEnv(

        [lambda: make_robosuite_env(args.task, training=True) for _ in range(args.training_num)]

    )

    test_envs = SubprocVectorEnv(

        [lambda: make_robosuite_env(args.task, training=False) for _ in range(args.test_num)]

    )



    # Observation Normalization (Mean=0, Std=1)

    train_envs = VectorEnvNormObs(train_envs, update_obs_rms=True)

    test_envs = VectorEnvNormObs(test_envs, update_obs_rms=False)

    test_envs.set_obs_rms(train_envs.get_obs_rms())



    # Seeding

    np.random.seed(args.seed)

    torch.manual_seed(args.seed)

    train_envs.seed(args.seed)

    test_envs.seed(args.seed)



    # Get Shapes

    dummy_env = make_robosuite_env(args.task)

    state_shape = dummy_env.observation_space.shape

    action_shape = dummy_env.action_space.shape

    print(f"Observation Shape: {state_shape}")

    print(f"Action Shape: {action_shape}")

    print(f"Device: {args.device}")



    # Network

    net_a = Net(state_shape, hidden_sizes=args.hidden_sizes, device=args.device)

    actor = ActorProb(net_a, action_shape, unbounded=True, device=args.device).to(args.device)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)



    net_c1 = Net(state_shape, action_shape, hidden_sizes=args.hidden_sizes, concat=True, device=args.device)

    critic1 = Critic(net_c1, device=args.device).to(args.device)

    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)



    net_c2 = Net(state_shape, action_shape, hidden_sizes=args.hidden_sizes, concat=True, device=args.device)

    critic2 = Critic(net_c2, device=args.device).to(args.device)

    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)



    if args.auto_alpha:

        target_entropy = -np.prod(action_shape)

        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)

        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)

        alpha = (target_entropy, log_alpha, alpha_optim)

    else:

        alpha = args.alpha



    # Policy

    policy = SACPolicy(

        actor=actor,

        actor_optim=actor_optim,

        critic1=critic1,

        critic1_optim=critic1_optim,

        critic2=critic2,

        critic2_optim=critic2_optim,

        tau=args.tau,

        gamma=args.gamma,

        alpha=alpha,

        estimation_step=1,

        action_space=dummy_env.action_space

    )



    # Buffer & Collectors

    buffer = VectorReplayBuffer(args.buffer_size, buffer_num=len(train_envs))

    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)

    test_collector = Collector(policy, test_envs, exploration_noise=False)



    print(f"Warmup: collecting {args.start_timesteps} steps...")

    train_collector.collect(n_step=args.start_timesteps, random=True)



    # Logging

    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")

    log_path = os.path.join(args.logdir, args.task, "sac", t0)

    writer = SummaryWriter(log_path)

    logger = TensorboardLogger(writer)



    def save_best_fn(policy):

        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

        print("Best policy saved!")



    def save_checkpoint_fn(epoch, env_step, gradient_step):

        torch.save(policy.state_dict(), os.path.join(log_path, f"checkpoint_{epoch}.pth"))



    print("Starting training...")

    result = OffpolicyTrainer(

        policy=policy,

        train_collector=train_collector,

        test_collector=test_collector,

        max_epoch=args.epoch,

        step_per_epoch=args.step_per_epoch,

        step_per_collect=args.step_per_collect,

        episode_per_test=args.test_num,

        batch_size=args.batch_size,

        update_per_step=args.update_per_step,

        save_best_fn=save_best_fn,

        save_checkpoint_fn=save_checkpoint_fn,

        logger=logger,

        stop_fn=lambda mean_reward: mean_reward >= 500,

    ).run()



    print(f"Training finished in {result['duration']}.")

    print(f"Best reward: {result['best_reward']}")



    train_envs.close()

    test_envs.close()



# =============================================================================

# 4. Visualization Loop

# =============================================================================

def watch_sac(args):

    """

    Loads a policy and records video of the agent performance.

    """

    import imageio

    print(f"Setup Visualization Env (Task: {args.task})...")

    

    # 1. Create visualization env with offscreen rendering

    vis_env = suite.make(

        args.task,

        horizon=500,

        robots="Panda",

        use_camera_obs=False,

        has_offscreen_renderer=True,

        has_renderer=False,

        reward_shaping=True,

        control_freq=20,

    )

    keys = ["robot0_proprio-state", "object-state", "gripper_to_cube_pos"]

    vis_env = GymWrapper(vis_env, keys=keys)

    vis_env = TianshouRobosuiteWrapper(vis_env)

    

    # 2. Define Policy Structure

    state_shape = vis_env.observation_space.shape

    action_shape = vis_env.action_space.shape

    

    net_a = Net(state_shape, hidden_sizes=args.hidden_sizes, device=args.device)

    actor = ActorProb(net_a, action_shape, unbounded=True, device=args.device).to(args.device)

    

    net_c1 = Net(state_shape, action_shape, hidden_sizes=args.hidden_sizes, concat=True, device=args.device)

    critic1 = Critic(net_c1, device=args.device).to(args.device)

    net_c2 = Net(state_shape, action_shape, hidden_sizes=args.hidden_sizes, concat=True, device=args.device)

    critic2 = Critic(net_c2, device=args.device).to(args.device)

    

    policy = SACPolicy(

        actor=actor,

        actor_optim=torch.optim.Adam(actor.parameters()), 

        critic1=critic1,

        critic1_optim=torch.optim.Adam(critic1.parameters()),

        critic2=critic2,

        critic2_optim=torch.optim.Adam(critic2.parameters()),

        action_space=vis_env.action_space

    )



    # 3. Load Weights

    if args.resume_path and os.path.exists(args.resume_path):

        print(f"Loading policy from: {args.resume_path}")

        checkpoint = torch.load(args.resume_path, map_location=args.device, weights_only=False)

        policy.load_state_dict(checkpoint)

    else:

        print("No resume path provided. Using random policy.")

    

    # 4. Set policy to eval mode

    policy.eval()

    

    # 5. Run episodes and record video manually

    video_folder = os.path.join("video", args.task)

    os.makedirs(video_folder, exist_ok=True)

    

    print(f"Recording {args.test_num} episodes to {video_folder}...")

    

    for ep in range(args.test_num):

        frames = []

        obs, _ = vis_env.reset()

        done = False

        total_reward = 0

        step_count = 0

        

        while not done:

            # Get frame before action (access the unwrapped robosuite env)

            try:
                # Try to get the base robosuite environment
                base_env = vis_env.unwrapped if hasattr(vis_env, 'unwrapped') else vis_env.env.env
                # Try common camera names in order of preference
                for camera in ['agentview', 'frontview', 'birdview']:
                    try:
                        frame = base_env.sim.render(width=640, height=480, camera_name=camera)
                        if frame is not None:
                            break
                    except:
                        continue
                else:
                    frame = None
            except Exception as e:
                frame = None

            if frame is not None:

                frames.append(frame[::-1])  # Flip vertically for OpenGL rendering

            

            # Policy action

            with torch.no_grad():

                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(args.device)

                (mu, sigma), _ = policy.actor(obs_tensor)

                action = mu.cpu().numpy()[0]  # Use mean for deterministic evaluation

            

            obs, reward, terminated, truncated, _ = vis_env.step(action)

            done = terminated or truncated

            total_reward += reward

            step_count += 1

        

        # Save video

        video_path = os.path.join(video_folder, f"episode_{ep}.mp4")

        if len(frames) > 0:

            imageio.mimsave(video_path, frames, fps=20)

            print(f"Episode {ep}: Reward={total_reward:.2f}, Steps={step_count}, Video saved to {video_path}")

        else:

            print(f"Episode {ep}: Reward={total_reward:.2f}, Steps={step_count}, No frames recorded")

    

    vis_env.close()

    print(f"All videos saved to {video_folder}")



# =============================================================================

# 5. Main Entry Point

# =============================================================================

if __name__ == "__main__":

    args = get_args()

    

    if args.watch:

        watch_sac(args)

    else:

        train_sac(args)
