import pickle
import time
import os
from datetime import datetime

import gymnasium
import numpy as np

from ss.gamepad.controllers import PS4
from ss.utils.gamepad_utils import get_gamepad_action, connect_gamepad

# Create the environment with rendering in human mode
env = gymnasium.make("ss/FrankaLiftEnv-v0", render_mode="human")
env_name = env.spec.id.split("/")[1]
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Create a gamepad instance
gamepad = connect_gamepad()

# Directory to save episodes
PATH = f"demonstrations/{env_name}/{timestamp}"
os.makedirs(PATH, exist_ok=True)

episode_counter = 0

try:
    while True:
        # Reset the environment with a seed for reproducibility
        observation, info = env.reset(seed=42)

        # Initialize episode data
        episode = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "terminated": [],
            "truncated": [],
            "infos": [],
        }

        done = False
        while not done:
            # Get action from gamepad
            action, active, _ = get_gamepad_action(gamepad)

            if active:
                # Split action into continuous and discrete parts
                grip = action[-1]  # Discrete action (0 or 1)
                continuous_action = action[:-1]  # 6-element array
                action_tuple = (continuous_action, grip)

                # Take a step in the environment
                observation, reward, terminated, truncated, info = env.step(
                    action_tuple
                )

                # Record the step data
                episode["observations"].append(observation)
                episode["actions"].append(action_tuple)
                episode["rewards"].append(reward)
                episode["terminated"].append(terminated)
                episode["truncated"].append(truncated)
                episode["infos"].append(info)

                # Check if episode is done
                done = terminated or truncated
            else:
                # Wait briefly if gamepad is not active to avoid busy waiting
                time.sleep(0.01)

        # Save the episode to a pickle file
        episode_filename = f"{PATH}/episode_{episode_counter}.pkl"
        with open(episode_filename, "wb") as f:
            pickle.dump(episode, f)
        print(f"Saved episode {episode_counter}")

        episode_counter += 1

except KeyboardInterrupt:
    print("Stopped by user. All collected episodes have been saved.")

finally:
    # Ensure the environment is closed properly
    env.close()
