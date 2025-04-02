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

        # Initialize episode data as a list
        episode = []

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

                # Extract state components (all non-image data)
                state_components = []

                # Add all non-image observations to state
                for key, value in observation.items():
                    if "view" not in key:  # Skip image data
                        state_components.append(
                            value.flatten()
                        )  # Flatten in case any are multi-dimensional

                # Combine all state components into a single vector
                state = np.concatenate(state_components).astype(np.float32)

                # Convert action_tuple to a single numpy array
                # Combine continuous action and gripper action
                action_array = np.zeros(
                    7, dtype=np.float32
                )  # 6 for continuous + 1 for gripper
                action_array[:6] = continuous_action
                action_array[6] = float(grip)

                # Record the step data
                step_data = {
                    "front_image": observation["front_view"],
                    "left_image": observation["left_view"],
                    "right_image": observation["right_view"],
                    "top_image": observation["top_view"],
                    "state": state,
                    "action": action_array,
                    "reward": float(reward),
                    "discount": 0.0 if (terminated or truncated) else 1.0,
                    "is_first": step_idx == 0,
                    "is_last": terminated or truncated,
                    "is_terminal": terminated,
                    "language_instruction": "pick up object",  # Add appropriate instruction
                }

                # Add step to episode
                episode.append(step_data)

                # Update for next iteration
                observation = next_observation
                done = terminated or truncated
                step_idx += 1
            else:
                # Wait briefly if gamepad is not active to avoid busy waiting
                time.sleep(0.01)

        # Save the episode to a numpy file
        episode_filename = f"{PATH}/episode_{episode_counter}.npy"
        np.save(episode_filename, episode)

        print(f"Saved episode {episode_counter} with {step_idx} steps")
        episode_counter += 1

except KeyboardInterrupt:
    print("Stopped by user. All collected episodes have been saved.")
finally:
    # Ensure the environment is closed properly
    env.close()
