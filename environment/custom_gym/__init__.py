from gymnasium.envs.registration import register

register(
        id='ToyExample-v0',
        entry_point='environment.custom_gym.envs.ToyExample.toyexample:ToyEnv',

)


register(
        id = 'Vehicle-v0',
        entry_point = 'environment.custom_gym.envs.DBicycle.dynamic_bicycle:DBicycle_Env'
        )


register(
        id = 'QuadRotor-v0',
        entry_point = 'environment.custom_gym.envs.QuadRotor.quadrotor:QuadRotorEnv'
)


register(
        id = 'OpenCat-v0',
        entry_point = 'environment.custom_gym.envs.OpenCat.opencat:OpenCatGymEnv'
)


register(
        id='Acrobot-v0',
        entry_point='environment.custom_gym.envs.AcrobotCont:AcrobotContEnv',
        max_episode_steps=400
)



register(
        id = 'Obstacle2D-v0',
        entry_point = 'environment.custom_gym.envs.Obstacle2D.obstacle2d:Obstacle2D_Env'
)

