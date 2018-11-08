from gym.envs.registration import register

register(
    id='multiple_control-v0',
    entry_point='multiple.envs:simulationEnv'
)


