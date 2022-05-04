import gym

from mas.envs.mas_env import MasEnv

def main():
    env = MasEnv()
    print(env.action_space)
    print(env.observation_space)
    env.reset()
    env.render(mode='human')
    done = False
    while not done:
        action = env.action_space.sample()
        #print(f'action = {action}')
        observation, reward, done, info = env.step(action)
        env.render(mode='human')
        #print(f'observation = {observation}')
    print(f'done')
    env.close()

if __name__ == '__main__':
    main()



