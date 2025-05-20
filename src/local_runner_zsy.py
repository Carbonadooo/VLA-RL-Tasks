import os
import gymnasium as gym
import time
from mani_skill.utils.wrappers.record import RecordEpisode
import sys
from tasks.ShaoyuZeng.cylinder_push_up_env import CylinderPushUpEnv
from tasks.JunhaoLi.table_scene_base import TableSceneEnv
import time

# render_notsave = False
render_notsave = True
task = ["CylinderPushUp-v1", "TableScene-v1", "DominoToppling-v1", "Jenga-v1"]
i = 0
def generate_videos(n_episodes=1, max_steps_per_episode=100, video_dir="task_videos/"+task[i]):
    """
    Generate and save videos of random agent interactions in the CylinderPushUp environment.
    """
    if render_notsave:
        env = gym.make(task[i], obs_mode="state", render_mode="human")
    else:
        env = gym.make(task[i], obs_mode="state", render_mode="rgb_array")
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