import os
import gymnasium as gym
import time
from mani_skill.utils.wrappers.record import RecordEpisode
import sys
from tasks.ShaoyuZeng.cylinder_push_up_env import CylinderPushUpEnv
import time

# render_notsave = False
render_notsave = True
def generate_videos(n_episodes=10, max_steps_per_episode=100, video_dir="task_videos/CylinderPushUp-v1"):
    """
    Generate and save videos of random agent interactions in the CylinderPushUp environment.
    """
    if render_notsave:
        env = gym.make("CylinderPushUp-v1", obs_mode="state", render_mode="human")
    else:
        env = gym.make("CylinderPushUp-v1", obs_mode="state", render_mode="rgb_array")
        video_dir = os.path.join(video_dir, time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(video_dir, exist_ok=True)

    if not render_notsave:
        env = RecordEpisode(env, output_dir=video_dir, save_video=True, 
                            trajectory_name="random_actions", max_steps_per_video=max_steps_per_episode)
    for _ in range(n_episodes):
        obs, info = env.reset()
        for _ in range(max_steps_per_episode):
            action = env.action_space.sample()  # Take random action
            obs, reward, terminated, truncated, info = env.step(action)
            # print(action, reward)
            if terminated or truncated:
                break
            if render_notsave:
                time.sleep(0.1)
                env.render()
    env.close()

if __name__ == "__main__":
    generate_videos(n_episodes=10)