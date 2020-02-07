from skimage.color import rgb2gray
from skimage.transform import resize
import gym
import numpy as np
from actor_critic import actor_critic

global episode
episode = 0
EPISODES = 8000000
# 환경 생성
env_name = "BreakoutDeterministic-v4"
class break_out_agent:
    def __init__(self, ):
        global episode
        self.ac = actor_critic(84 * 84 , 3)
        self.env = gym.make(env_name)
        self.act_mapper = {0:1, 1:2, 2:3}
        self.action_size = 3
        episode = 0
    
    def get_action(self, history):
        history = np.float32(history / 255.)
        tmp_history = history.reshape(4, 84 * 84)
        tmp_history = np.swapaxes(tmp_history, 0, 1)
        policy, value = self.ac.actor_critic.predict_on_batch([[tmp_history]])
        # policy, value = self.ac.actor.predict_on_batch([[history]]), self.ac.critic.predict_on_batch([[history]])
        policy = policy[0]
        # print(policy)
        action_index = np.random.choice(self.action_size, 1, p=policy)[0]
        return action_index, policy

    def run(self):
        global episode
        while episode < EPISODES:
            done = False
            dead = True

            score, start_life = 0, 5
            step = 0
            observe = self.env.reset()
            
            # states, actions, rewards
            states, actions, rewards = [], [], []
            state = self.pre_processing(observe)
            history = np.stack((state, state, state, state), axis=2)
            history = np.reshape([history], (84 * 84 * 4))
            states.append(history.copy())
            actions.append([1, 0, 0])
            rewards.append(0)

            action_history = []
            policy_history = []
            while not done:
                step += 1

                action, policy = self.get_action(history)
                real_action = self.act_mapper[action]

                action_history.append(action)
                policy_history.append(policy)

                if dead:
                    action = 0
                    real_action = 1
                    dead = False

                next_observe, reward, done, info = self.env.step(real_action)
                state = self.pre_processing(next_observe)
                state = np.reshape([state], (84 * 84))
                history = np.concatenate([history[84 * 84:], state])

                states.append(history.copy())
                actions.append([1 if action == i else 0 for i in range(self.action_size)])
                # print(actions[-1])
                rewards.append(reward)

                if start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']

                score += reward
                reward = np.clip(reward, -1., 1.)

                if done:
                    states = np.array(states) / 256
                    actor_loss, critic_loss = self.ac.train_models(states, actions, rewards)
                    if episode % 100 == 0:
                      self.ac.actor_critic.save_weights('actor_critic_breakout_conv.h5')
                    episode += 1
                    print("episode:", episode, "  score:", score, " actor_loss:", np.mean(actor_loss),\
                            " critic_loss:", np.mean(critic_loss), "  step:", step, " max policy:", np.max(policy_history), " min policy:", np.min(policy_history))
                    # print(action_history)
                    # print(actor_loss)
                    step = 0

    def pre_processing(self, observe):
        # processed_observe = np.maximum(observe)
        processed_observe = np.uint8(
            resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
        return processed_observe

if __name__ == "__main__":
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다            
            print(e)
    agent = break_out_agent()
    agent.run()