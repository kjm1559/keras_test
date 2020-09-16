from skimage.color import rgb2gray
from skimage.transform import resize
import gym
import numpy as np
from actor_critic import actor_critic

from PIL import Image
import cv2

global episode
episode = 0
EPISODES = 8000000
# Create environment
env_name = "BreakoutDeterministic-v0"
class break_out_agent:
    def __init__(self, ):
        global episode
        self.ac = actor_critic((84, 84, 4) , 3)
        self.env = gym.make(env_name)
        self.act_mapper = {0:1, 1:2, 2:3}
        self.action_size = 3
        episode = 0
        print(self.env.get_action_meanings())
    
    def get_action(self, history):
        # history = np.float32(history / 255.)
        # tmp_history = history.reshape(84, 84, 4)
        # tmp_history = np.swapaxes(tmp_history, 0, 1)
        policy, value = self.ac.actor_critic.predict_on_batch([[history / 255.]])
        # policy = self.ac.actor.predict_on_batch([[tmp_history / 255.]])
        policy = policy[0]
        # print(policy)        
        action_index = np.random.choice(self.action_size, 1, p=policy)[0]
        return action_index, policy
    def get_test_action(self, history):
        # history = np.float32(history / 255.)
        # tmp_history = history.reshape(84, 84, 4)        
        policy, value = self.ac.actor_critic.predict_on_batch([[history / 255.]])
        # policy = self.ac.actor.predict_on_batch([[tmp_history / 255.]])
        policy = policy[0]
        print(np.max(history), np.max(history / 255.))
        action_index = np.argmax(policy)
        return action_index, policy

    def make_video(self):
        observe = self.env.reset()
        # self.env.step(1)
        next_observe, reward, done, info = self.env.step(1)
        state = self.pre_processing(next_observe)
        tmp_history = np.stack((state, state, state, state), axis=2)
        start_life = info['ale.lives']
        dead = False
        
        video = cv2.VideoWriter(str(episode) + '_episode.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (84, 84))
        while not done:                            
            real_action, p = self.get_test_action(tmp_history.copy())
            real_action = self.act_mapper[real_action]
            print(real_action)
            next_observe, _, done, info = self.env.step(real_action)
            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']                
                continue
            
            if dead:
                next_observe, _, done, info = self.env.step(1)
                state = self.pre_processing(next_observe)
                tmp_history = np.stack((state, state, state, state), axis=2)
                tmp_history = np.reshape([tmp_history], (84, 84, 4))
                print(np.sum(tmp_history[:, :, -1] - state), p, _)
                dead = False

            
            state = self.pre_processing(next_observe)
            state = np.reshape([state], (84, 84, 1)) 
            # print(tmp_history[:, :, -1].shape, state.shape)
            # print(np.sum(tmp_history[:, :, -1]), np.sum(state), (tmp_history[:, :, -1] - state.reshape(84, 84)).shape)
            print(np.sum(tmp_history[:, :, -1] - state.reshape(84, 84)), p, _)
            tmp_history = np.concatenate((tmp_history[:, :, 1:], state.copy()), axis=2)  

            gray = cv2.normalize(tmp_history[:, :, -1].reshape((84, 84)) / 255., None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            gray_3c = cv2.merge([gray, gray, gray])                            
            video.write(gray_3c)

        video.release()
        cv2.destroyAllWindows()


    def run(self):
        global episode
        while episode < EPISODES:
            done = False
            dead = True

            score, start_life = 0, 5
            step = 0
            observe = self.env.reset()
            
            for _ in range(np.random.randint(1, 30)):
                observe, _, _, _ = self.env.step(1)

            # states, actions, rewards
            states, actions, rewards = [], [], []
            state = self.pre_processing(observe)
            history = np.stack((state, state, state, state), axis=2)
            history = np.reshape([history], (84, 84, 4))
            states.append(history.copy())
            actions.append([1, 0, 0])
            rewards.append(0)

            action_history = []
            policy_history = []
            one_shot = np.random.randint(200)
            while not done:                
                action, policy = self.get_action(history)
                real_action = self.act_mapper[action]
                # print('hello', step, score, real_action)

                action_history.append(action)
                policy_history.append(policy)

                if dead:
                    action = 0
                    real_action = 1
                    dead = False
                    next_observe, reward, done, info = self.env.step(real_action)

                next_observe, reward, done, info = self.env.step(real_action)
                # if real_action == 1:
                #     real_action = -1
                #     # print('dead and fire')
                #     continue
                step += 1

                if real_action != 1:
                    states.append(history.copy())
                    # print(np.sum(np.array(states[-1]) - np.array(states[-2])))
                    actions.append([1 if action == i else 0 for i in range(self.action_size)])
                    reward = np.clip(reward, -1, 1)
                    rewards.append(reward.copy())

                state = self.pre_processing(next_observe)
                # if step == one_shot:
                #     if episode%10 == 0:
                #         # target_img = Image.fromarray(state)
                #         # target_img.save(str(episode) + '_' + str(step) + '.gif')
                #         for kk in range(4):
                #             target_img = Image.fromarray(history[:,:,kk].reshape(84, 84))
                #             target_img.save(str(episode) + '_' + str(step) + '_' + str(kk) + '.gif')

                state = np.reshape([state], (84, 84, 1))                
                history = np.concatenate((history[:, :, 1:], state), axis=2)                

                if start_life > info['ale.lives']:                    
                    dead = True
                    start_life = info['ale.lives']

                score += reward
                reward = np.clip(reward, -1., 1.)

                if dead:
                    # for _ in range(np.random.randint(1, 5)):
                    #     observe, _, _, _ = self.env.step(1)
                    # state = self.pre_processing(observe)
                    history = np.stack((state, state, state, state), axis=2)
                    history = np.reshape([history], (84, 84, 4))

                if done:
                    states = np.array(states)
                    # if episode % 100 == 0:
                    #     video = cv2.VideoWriter(str(episode) + '_episode.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (84, 84))
                    #     for i in range(step):
                    #         gray = cv2.normalize(states[i, :, :, 3].reshape((84, 84)), None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    #         gray_3c = cv2.merge([gray, gray, gray])                            
                    #         video.write(gray_3c)
                    #     video.release()
                    #     cv2.destroyAllWindows()
                    states = states / 255.                    
                    actor_loss, critic_loss = self.ac.train_models(states, actions, rewards)
                    # loss = self.ac.train_models(states, actions, rewards)
                    if episode % 100 == 0:
                        self.ac.actor_critic.save_weights('actor_critic_breakout_conv2.h5')
                        # self.ac.actor.save_weights('actor_breakout_conv2.h5')
                        # self.ac.critic.save_weights('critic_breakout_conv2.h5')
                    if episode % 100 == 0:
                        self.make_video()                        
                    
                    episode += 1                    
                    print("episode:", episode, "  score:", score, " actor_loss:", actor_loss,\
                            " critic_loss:", critic_loss, "  step:", step, " max policy:", np.max(policy_history), " action count:", [len(np.array(action_history)[np.array(action_history) == jj]) for jj in range(3)])
                    # print("episode:", episode, "  score:", score, " actor_loss:", np.mean(loss),\
                    #         " critic_loss:", loss, "  step:", step, " max policy:", np.max(policy_history), " action count:", [len(np.array(action_history)[np.array(action_history) == jj]) for jj in range(3)])
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
            print(e)
    agent = break_out_agent()
    agent.run()