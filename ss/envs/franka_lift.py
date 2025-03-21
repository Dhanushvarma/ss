import numpy as np
import time
import os
import random

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

        # TODO: figure out from model directly
        self.num_blocks = 2
        self.block_names = [f"block{i}" for i in range(1, self.num_blocks + 1)]
        self.block_ids = [self.model.body(name).id for name in self.block_names]

        self.observation_space["blocks_poses"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7 * self.num_blocks,), dtype=np.float64
        )

        # how much the block has to be lifted.
        self.lift_height = 0.15

        # NOTE : assumption is that all the blocks are resting at the same height
        self.block_resting_height = self.model.body(
            random.choice(self.block_names)
        ).pos[2]

        # reset params
        self._reset_noise_scale_blocks = 0.05

    def reset_model(self):

        super().reset_model()

        # add noise to x,y position of the blocks
        low = -self._reset_noise_scale_blocks
        high = self._reset_noise_scale_blocks

        for _block_name in self.block_names:

            _dof_adr = self.model.body(_block_name).dofadr[0]
            _dof_size = self.model.body(_block_name).dofnum[0]
            assert _dof_size == 6  # has to be a free joint

            self.data.qpos[_dof_adr : _dof_adr + 2] += np.random.uniform(
                low, high, size=2
            )

        # TODO : why do i see rotation sometimes, in the block's reset pose?

        # call mj_forward to update the data
        mujoco.mj_forward(self.model, self.data)

        observation = self._get_obs()

        return observation

    def _task_done(self):

        done = False

        # is any block lifted above the lift height
        for _block_id in self.block_ids:
            if self.data.xpos[_block_id][2] > (
                self.lift_height + self.block_resting_height
            ):
                done = True
                break

        return done

    def _get_obs(self):

        _obs_dict = super()._get_obs()

        # block pos and quat part of the observation
        _block_obs = []

        for _block_id in self.block_ids:
            _block_obs.extend(self.data.xpos[_block_id])
            _block_obs.extend(self.data.xquat[_block_id])

        _obs_dict["blocks_poses"] = np.array(_block_obs)

        return _obs_dict
