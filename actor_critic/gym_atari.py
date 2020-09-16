import gym
import numpy as np
import tensorflow as tf
from skimage.color import rgb2gray
from skimage.transform import resize

from actor_critic_model import agent

from PIL import Image


env_name = "BreakoutDeterministic-v4"
episode = 0
EPISODES = 800000
action_mapper = {0:0, 1:2, 2:3}

class worker:
    def pre_processing(self, observe):
        # resize and to grey        
        processed_observe = np.uint8(
            resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
        return processed_observe
    
    def get_action(self, env, ac, history, clone_state, action_len):        
        target_data = []
        target_reward = []
        for i in range(action_len):
            env.restore_state(clone_state)
            new_observe, reward, _, _ = env.step(action_mapper[i])
            new_observe = self.pre_processing(new_observe)
            target_data.append(np.concatenate((history[84 * 84:], new_observe), axis=None).tolist())
            target_reward.append(reward)

        # best action calculate        
        policy_value = ac.model.predict_on_batch(np.array(target_data) / 255)
        # print(np.array(policy_value).shape)
        # print(policy_value)
        # real_action = np.argmax(policy_value[0].numpy().reshape(3))
        policy = policy_value[0].numpy()[:,1].reshape(action_len)
        # print(policy)
        real_action = np.random.choice(action_len, p=(policy / np.sum(policy)))
        actions = [[0]] * action_len
        actions[real_action][0] = 1
        return real_action, policy_value[1].numpy()[real_action][0], policy_value[0].numpy()[:,1].reshape(action_len).tolist()
        # transition = []
        # for i in range(action_len):
        #     transition.append({'state': target, 'action': 1, 'reward':reward, 'value': value})
        # return transition

    def get_action_softmax(self, env, ac, history, clone_state, action_len):        
        policy_value = ac.model.predict_on_batch((np.array(history) / 255).reshape(1, len(history)))        
        policy = policy_value[0].numpy()[0]
        # print(policy)
        real_action = np.random.choice(action_len, p=(policy / np.sum(policy)))
        # real_action = np.argmax(policy)        
        return real_action, policy_value[1].numpy()[0], policy.tolist()

    def __init__(self): 
        global episode  
        global action_mapper
             
        env = gym.make(env_name)
        step = 0
        action_len = len(env.get_action_meanings()) - 1
        input_shape = 84 * 84 * 4

        ac = agent(input_shape, 'test_atari')        

        writer = tf.summary.create_file_writer('graph_worker_softmax')

        while episode < EPISODES:
            done = False
            dead = False
            # tmp_Transition = {'state': [], 'action': [], 'reward': []}
            
            transition = []

            score, start_life = 0, 5
            observe = env.reset()
            # fire ball
            next_observe, _, _, _ = env.step(1)

            state = self.pre_processing(next_observe)
            history = np.stack((state, state, state, state), axis=2)
            history = np.reshape([history], (84 * 84 * 4))
            transition.append({'state': history.copy(), 'action': [1, 0, 0], 'reward': 0, 'value': 0})

            action_history = []
            policy_history = []
            one_shot = np.random.randint(200)
            while not done:
                step += 1
                # current state clone
                clone_state = env.clone_state()
                
                # action, value, policy = self.get_action(env, ac, history, clone_state, action_len)
                action, value, policy = self.get_action_softmax(env, ac, history, clone_state, action_len)                                

                # state restore
                # env.restore_state(clone_state)

                if dead:  
                    action = 0                  
                    real_action = 1
                    dead = False
                else:
                    real_action = action_mapper[action]
                action_history.append(real_action)
                policy_history.append(policy)

                next_observe, reward, done, info = env.step(real_action)
                next_observe = self.pre_processing(next_observe)
                if step == one_shot:
                    target_img = Image.fromarray(next_observe)
                    target_img.save(str(episode) + '_' + str(step) + '.gif')

                history = np.concatenate((history[84 * 84:], next_observe), axis=None)
                score += reward
                reward = np.clip(reward, -1., 1.)
                action_vector = [0] * action_len
                action_vector[action] = 1
                transition.append({'state': (history / 255).copy(), 'action': action_vector.copy(), 'reward':reward.copy(), 'value': value.copy()})

                if start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']

                if done:
                    episode += 1
                    actor_loss, critic_loss = ac.update_model(transition)
                    print('episode:', episode, ' score:', score, ' step:', step, ' actor_loss:', np.mean(actor_loss), ' critic_loss:', critic_loss, ' last_policy:', policy_history[-1])
                    print('history', action_history)
                    ac.model.save_weights('atari_test.h5')                    
                    with writer.as_default():
                        tf.summary.scalar('score', score, step=episode)
                        tf.summary.scalar('step', step, step=episode)
                        tf.summary.scalar('actor_loss', np.mean(actor_loss), step=episode)
                        tf.summary.scalar('critic_loss', np.mean(critic_loss), step=episode)
                    step = 0

if __name__ == '__main__':
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)
    worker()