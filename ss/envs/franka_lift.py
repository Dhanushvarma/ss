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

    def __init__(self, render_mode=None, xml_path=LIFT_XML_PATH):
        super().__init__(render_mode, xml_path)
