# SS: Simple Suite for Manipulation Experiments

The "Simple Suite" (SS) provides a clean, modular framework for robotics manipulation experiments using MuJoCo and Gymnasium.

## Features

- Franka Emika Panda robot arm environments
- PlayStation controller integration for teleoperation
- Operational Space Control (OSC) for smooth robot movements
- Data collection pipeline with RLDS-compatible format
- Multiple camera views for rich visual observations

## Installation

```bash
# Clone the repository
git clone https://github.com/Dhanushvarma/ss
cd ss

# Install the package
pip install -e .
```

## Requirements

- Python 3.8+
- MuJoCo
- Gymnasium
- NumPy
- PlayStation 4 controller (for teleoperation)

## Usage

### Running a basic environment

```python
import gymnasium
import time

# Create the environment
env = gymnasium.make("ss/FrankaLiftEnv-v0", render_mode="human")

# Reset the environment
observation, info = env.reset(seed=42)

# Run a simple loop (no control)
for _ in range(100):
    action = (np.zeros(6), 0)  # (position control, gripper control)
    observation, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.01)

# Close the environment
env.close()
```

### Teleoperation with a gamepad

```python
import gymnasium
import numpy as np
import time

from ss.gamepad.controllers import PS4
from ss.utils.gamepad_utils import get_gamepad_action, connect_gamepad

# Create the environment
env = gymnasium.make("ss/FrankaLiftEnv-v0", render_mode="human")
observation, info = env.reset(seed=42)

# Connect gamepad
gamepad = connect_gamepad()

# Control loop
while True:
    # Get action from gamepad
    action, active, terminate = get_gamepad_action(gamepad)
    
    if active:
        # Split action into continuous and discrete parts
        grip = action[-1]  # Discrete action (0 or 1)
        continuous_action = action[:-1]  # 6-element array
        action_tuple = (continuous_action, grip)
        
        # Take a step in the environment
        observation, reward, terminated, truncated, info = env.step(action_tuple)
        
        # Check if episode is done
        if terminated or truncated:
            observation, info = env.reset()
            
    else:
        # Wait briefly if gamepad is not active
        time.sleep(0.01)
        
    # Check for terminate signal from gamepad
    if terminate:
        break

env.close()
```

### Collecting demonstrations

Use the `scripts/demo_collection.py` script to collect demonstrations:

```bash
python scripts/demo_collection.py
```

This will save demonstrations in pickle format with an RLDS-compatible structure.

## Available Environments

### FrankaEnv-v0

A basic environment with a Franka Emika Panda robot arm.

- **Observation Space**: Joint positions, end-effector position and orientation, gripper position, camera views
- **Action Space**: 6-DOF end-effector control + gripper control

### FrankaLiftEnv-v0

A lifting task environment with a Franka Emika Panda robot arm.

- **Observation Space**: Same as FrankaEnv-v0 + block poses
- **Action Space**: Same as FrankaEnv-v0
- **Task**: Lift one of the blocks above a specified height

## System Architecture

The package is organized as follows:

```
ss/
├── assets/               # MuJoCo model files
├── envs/                 # Gymnasium environments
│   ├── franka_env.py     # Base Franka environment
│   └── franka_lift.py    # Lifting task environment  
├── gamepad/              # Gamepad controller interface
├── utils/                # Utility functions
└── scripts/              # Example scripts and demos
```

## Controller Mapping

When using a PlayStation 4 controller:

- **Left Joystick**: XY movement in the workspace
- **Right Joystick**: Z movement in the workspace
- **L2 Button**: Close gripper
- **R2 Button**: Enable control (must be held)
- **Circle Button**: Terminate episode

## Operational Space Control (OSC)

The environments use Operational Space Control for smooth end-effector control. The controller runs at a high frequency internally while the Gymnasium interface runs at a lower frequency.

## Visualization

To visualize collected demonstrations:

```python
import pickle

# Load a demonstration file
with open("demonstrations/episode_0.pkl", "rb") as f:
    demo = pickle.load(f)
    
# Print demonstration metadata
print(f"Number of steps: {len(demo['observations'])}")
print(f"Total reward: {sum(demo['rewards'])}")
```

## License

MIT License