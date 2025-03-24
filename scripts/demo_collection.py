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

# spaces
print("Observation Space : ", env.observation_space)
print("Action Space : ", env.action_space)

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

        # Initialize episode data with RLDS structure
        episode = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "discounts": [],
            "is_first": [],
            "is_last": [],
            "is_terminal": [],
        }

        done = False
        step_idx = 0

        while not done:
            # Get action from gamepad
            action, active, *_ = get_gamepad_action(gamepad)

            if active:
                # Split action into continuous and discrete parts
                grip = action[-1]  # Discrete action (0 or 1)
                continuous_action = action[:-1]  # 6-element array
                action_tuple = (continuous_action, grip)

                # Take a step in the environment
                next_observation, reward, terminated, truncated, info = env.step(
                    action_tuple
                )

                # Record the step data in RLDS-aligned format
                episode["observations"].append(observation)
                episode["actions"].append(action_tuple)
                episode["rewards"].append(reward)
                episode["discounts"].append(0.0 if (terminated or truncated) else 1.0)
                episode["is_first"].append(step_idx == 0)
                episode["is_last"].append(terminated or truncated)
                episode["is_terminal"].append(terminated)

                # Update for next iteration
                observation = next_observation
                done = terminated or truncated
                step_idx += 1
            else:
                # Wait briefly if gamepad is not active to avoid busy waiting
                time.sleep(0.01)

        # Save the episode to a pickle file with RLDS-aligned structure
        episode_filename = f"{PATH}/episode_{episode_counter}.pkl"
        with open(episode_filename, "wb") as f:
            pickle.dump(episode, f)

        print(f"Saved episode {episode_counter} with {step_idx} steps")
        episode_counter += 1

except KeyboardInterrupt:
    print("Stopped by user. All collected episodes have been saved.")
finally:
    # Ensure the environment is closed properly
    env.close()
