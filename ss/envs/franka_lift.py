import numpy as np
import time
import os

import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer

from ss.envs.franka_env import FrankaEnv


class FrankaLiftEnv(FrankaEnv):

    pwd = os.path.dirname(os.path.abspath(__file__))
    LIFT_XML_PATH = os.path.join(pwd, "../../assets/franka_emika_panda/lift.xml")

    def __init__(self, model_path=LIFT_XML_PATH, render_mode=None):
        super().__init__(model_path=model_path, render_mode=render_mode)
