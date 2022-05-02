from gym.envs.registration import register

register(
    id='HideAndSeek15x15-v0',
    entry_point='multiagent_survival.gym_hidenseek.envs:HideAndSeek15x15Env',
    order_enforce=False
)

register(
    id='RandomizedHideAndSeek15x15-v0',
    entry_point='multiagent_survival.gym_hidenseek.envs:RandomizedHideAndSeek15x15Env',
)

register(
    id='JsonHideAndSeek15x15-v0',
    entry_point='multiagent_survival.gym_hidenseek.envs:JsonHideAndSeek15x15Env',
)

register(
    id='LockAndReturn15x15-v0',
    entry_point='multiagent_survival.gym_hidenseek.envs:LockAndReturn15x15Env',
)

register(
    id='SequentialLock15x15-v0',
    entry_point='multiagent_survival.gym_hidenseek.envs:SequentialLock15x15Env',
)