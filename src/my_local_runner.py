import os
import gymnasium as gym
import time
from mani_skill.utils.wrappers.record import RecordEpisode
from tasks.ShaoyuZeng import cylinder_push_up_env, domino_toppling_env


task_names = ["CylinderPushUp-v1", "DominoToppling-v1"]
task_idx = 1
task_name = task_names[task_idx]
is_save = False

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
task_videos_dir = os.path.abspath(os.path.join(script_dir, "../task_videos/",task_name))

def generate_videos(n_episodes=10, max_steps_per_episode=100, video_dir=task_videos_dir):
    """
    Generate and save videos of random agent interactions.
    """
    if is_save:
        env = gym.make(task_name, obs_mode="state", render_mode="rgb_array")
        video_dir = os.path.join(video_dir, time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(video_dir, exist_ok=True)
        env = RecordEpisode(env, output_dir=video_dir, save_video=True, 
                        trajectory_name="random_actions", max_steps_per_video=max_steps_per_episode)
    else:
        env = gym.make(task_name, obs_mode="state", render_mode="human")

    for _ in range(n_episodes):
        obs, info = env.reset()
        for _ in range(max_steps_per_episode):
            action = env.action_space.sample()  # Take random action
            obs, reward, terminated, truncated, info = env.step(action)
            if not is_save:
                env.render()
                time.sleep(0.01)
            if terminated or truncated:
                break
    env.close()

if __name__ == "__main__":
    generate_videos(n_episodes=10)