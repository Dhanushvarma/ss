from gymnasium.envs.registration import register

register(
    id="ss/FrankaEnv-v0",
    entry_point="ss.envs:FrankaEnv",
    # Optionally, you can set a maximum number of steps per episode
    # max_episode_steps=300,
    # TODO: Uncomment the above line if you want to set a maximum episode step limit
)

register(
    id="ss/FrankaLiftEnv-v0",
    entry_point="ss.envs:FrankaLiftEnv",
    # Optionally, you can set a maximum number of steps per episode
    # max_episode_steps=300,
    # TODO: Uncomment the above line if you want to set a maximum episode step limit
)