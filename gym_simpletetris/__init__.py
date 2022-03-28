from gym.envs.registration import register

register(
    id='SimpleTetris-v0',
    entry_point='gym_simpletetris.envs:TetrisEnv',
)
