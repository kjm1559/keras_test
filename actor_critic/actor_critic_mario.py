from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from PIL import Image
import numpy as np

# keras lib
from tensorflow.keras.layers import Dense, Conv2D, Input, Flatten, Concatenate, MaxPooling2D, GRUCell
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

import cv2
import time

from itertools import count

env = gym_super_mario_bros.make('SuperMarioBros-v3')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

env_save = gym_super_mario_bros.make('SuperMarioBros-v0')
env_save = JoypadSpace(env_save, SIMPLE_MOVEMENT)

# state size : (240, 256, 3)
# info : {
#     'coins': 0, 
#     'flag_get': False, 
#     'life': 2, 'score': 0, 
#     'stage': 1, 
#     'status': 'small', 
#     'time': 400, 
#     'world': 1, 
#     'x_pos': 40, 
#     'y_pos': 79
#     }
# [['NOOP'], ['right'], ['right', 'A'], ['right', 'B'], ['right', 'A', 'B'], ['A'], ['left']]
action_mapping = {
    0: [0, 0, 0, 0, 0, 0], # Noop
    1: [1, 0, 0, 0, 0, 0], # Up
    2: [0, 0, 1, 0, 0, 0], # Down
    3: [0, 1, 0, 0, 0, 0], # Left
    4: [0, 1, 0, 0, 1, 0], # Left + A
    5: [0, 1, 0, 0, 0, 1], # Left + B
    6: [0, 1, 0, 0, 1, 1], # Left + A + B
    7: [0, 0, 0, 1, 0, 0], # Right
    8: [0, 0, 0, 1, 1, 0], # Right + A
    9: [0, 0, 0, 1, 0, 1], # Right + B
    10: [0, 0, 0, 0, 1, 1], # Right + A + B
    11: [0, 0, 0, 0, 1, 0], # A
    12: [0, 0, 0, 0, 0, 1], # B
    13: [0, 0, 0, 0, 1, 1], # A + B
}

from skimage.color import rgb2gray
from skimage.transform import resize
def pre_processing(self, observe):
        # processed_observe = np.maximum(observe)
        processed_observe = np.uint8(resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
        return processed_observe

image_size = 84
hidden_size = 512

def resize_img(img):
    # im = Image.fromarray(img).convert('L')
    # return np.asarray(im.resize((28, 28))) / 255.
    processed_observe = np.uint8(resize(rgb2gray(img), (image_size, image_size), mode='constant') * 255)
    return processed_observe / 255

def learning_step(X, hstate, y, model, optimizer, reward):
    with tf.GradientTape() as tape:
        action, value, h_state = model([X, hstate], training=True)
        # calulate advantage
        adv = reward - value
        # entropy = K.categorical_crossentropy(action, y)
        weighted_actions = K.sum(action * y, axis=1)
        eligibility = K.log(weighted_actions) * K.stop_gradient(adv)
        entropy = - K.mean(action * K.log(action))
        actor_loss = 1.0e-3 * entropy - K.mean(eligibility)
        #actor_loss = K.sum(adv * K.categorical_crossentropy(tf.cast(action, 'float'), tf.cast(y, 'float')))
        value_loss = K.mean(tf.keras.losses.MSE(reward, value))

        loss = [actor_loss, value_loss]
    grads=tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # print(model.trainable_variables)
    return model, loss, actor_loss, value_loss

def discount(r, gamma):
        """ Compute the gamma-discounted rewards over an episode
        """
        discounted_r, cumul_r = np.zeros(len(r)), 0.
        for t in reversed(range(0, len(r))):
            cumul_r = r[t] + cumul_r * gamma
            discounted_r[t] = cumul_r
        return discounted_r

def loss_function(y_true, y_pred):
    action_loss = K.mean(K.exp(-y_true[:, 7]) * K.sum((-y_true[:, :7] * K.log(y_pred) - ((1 - y_true[:, :7]) * K.log(1 - y_pred))), axis=1))
    return action_loss

def attention_mechanism(input):
    x = Dense(16, activation='softmax')(input)
    return x

def make_network():
    inputs = Input(shape=(image_size, image_size, 1), name='input_data')
    state = Input(shape=(hidden_size,), name='state')
    x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
    
    # print(x.shape)
    x = Flatten()(x)
    x, new_state = GRUCell(hidden_size)(x, state)

    action_out = Dense(7, activation='softmax', name='action')(x)
    # x = Concatenate()([x, action_out])
    value_out = Dense(1, activation='linear', name='reward')(x)

    model = Model(inputs=[inputs, state], outputs=[action_out, value_out, new_state])
    # model.compile(optimizer='adam', loss=[loss_function, 'mse'])
    # model.summary()
    return model


def train(episode=10000, step=4):
    # load model
    from tensorflow.keras.models import load_model
    model = load_model('mario_test')
    # create model
    # model = make_network()
    optimizer = tf.keras.optimizers.Adam(lr=1e-4)

    # start
    state = env.reset()
    model_input = np.transpose(np.repeat([resize_img(state)], step, axis=0), (1, 2, 0)).reshape(1, image_size, image_size ,step)
    start = time.time()
    h_state = tf.convert_to_tensor(np.zeros(hidden_size).reshape(1, hidden_size), np.float32)

    # for e in range(episode):
    ep = 1
    p_state = 0
    stay_count = 0

    for e in count():
        done = True
        # train step
        history_state = []
        history_value = []
        history_reward = []
        history_action = []
        history_hstate = []
        done_flag = True

        
        # state, reward, done, info = env.step(1)
        # history_reward.append(reward)
        # model_input = resize_img(state)
        
                
        for i in range(50):
        # for step in range(10):
            current_state = resize_img(state).reshape(1, image_size, image_size, 1)
            # model_input = np.concatenate([model_input[:, :, :, 1:], current_state], axis=3)

            # action_prob, value = model.predict_on_batch(model_input)
            # try:
            #     history_state.append(h_state.tolist())
            # except:
            # print(h_state.numpy().tolist())
            history_hstate.append(h_state.numpy().tolist()[0])
            action_prob, value, h_state = model([current_state, h_state], training=False)
            action_prob = action_prob.numpy()
            value = value.numpy()

            # print(action_prob)
            action = np.random.choice(7, p=action_prob[0])
            action_onehot = np.zeros(7)
            action_onehot[action] = 1

            # print(np.mean(current_state), np.max(current_state), action)
            
            state, reward, done, info = env.step(action)
            # if done != True: 
            #     state, reward, done, info = env.step(action)

            # if p_state == info['x_pos']:
            #     stay_count += 1
            #     reward -= 5
            # else:
            #     stay_count = 0
            # p_state = info['x_pos'].copy()

            # history_state.append(model_input[0].tolist())
            history_state.append(current_state[0].tolist().copy())
            history_reward.append(reward)
            history_action.append(action_onehot.tolist())
            # print(state.shape, info)
            # print(resize_img(state).shape)
            # print(reward)
            # print(info, action)
            
            if (done):# | (stay_count > 1000):
                state = env.reset()
                done_flag = False
                print(state.shape, info, stay_count)
                h_state = tf.convert_to_tensor(np.zeros(hidden_size).reshape(1, hidden_size), np.float32)
                break
                # if stay_count > 1000:
                    # history_reward = history_reward[:-1]
                    # history_reward[-1] = -30
                
            # env.render()
        # print(str(e) + ' episode event len :' + str(len(history_reward)), ', time :', time.time() - start, ', x_pos :', info['x_pos'])
        # history_reward = [history_reward[i] + history_reward[i + 1] * 0.9 for i in range(len(history_reward) - 2, -1, -1)][::-1]\
        #                  + [history_reward[-1]]

        if p_state == info['x_pos']:
            stay_count += 1
        else:
            stay_count = 0
        p_state = info['x_pos']
        if stay_count >= 2:
            state = env.reset()
            print(state.shape, info, stay_count)
            h_state = tf.convert_to_tensor(np.zeros(hidden_size).reshape(1, hidden_size), np.float32)
            done = True
            history_reward[-1] = -30

        history_reward = discount(np.array(history_reward), 0.9)        
        # print(history_reward.tolist())
        history_reward = np.clip(np.array(history_reward), -50, 50) #/ 20
        # print(history_reward)
        # print([np.argmax(dd) for dd in history_action])
        
        model, loss, actor_loss, value_loss = learning_step(np.array(history_state), np.array(history_hstate), np.array(history_action), model, optimizer, np.array(history_reward).reshape(len(history_reward), 1))
        if e % 100:
            model.save('mario_test4')
        loss = [loss[0].numpy(), loss[1].numpy()]

        # cal action rate
        actions = np.array([np.argmax(a) for a in history_action])
        u_actions = list(set(actions))
        action_rate = {a: format(len(actions[actions==a])/len(actions), '.2f') for a in u_actions}

        # print(loss)

        print(str(e) + ' step ->', #'loss :', format(loss[0] + loss[1], '.2f'), \
            'actor_loss :', format(loss[0], '.2f'), \
            'value_loss :', format(loss[1], '.2f'), \
            'reward_mean :', format(np.mean(history_reward), '.2f'), \
            ', pos : (' + str(info['x_pos']) + ', ' + str(info['y_pos']) + ')', \
            ', time :', info['time'], \
            ', unique action :', action_rate)

        # validation step
        # if e % 1 == 0:
        if done:
            print('save start ', ep)
            # video = cv2.VideoWriter(str(e) + '_episode.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 60, (256, 240))
            # video_ori = cv2.VideoWriter('episode_out2/' + str(ep) + '_episode_ori.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 60, (256, 240))
            video_ori = cv2.VideoWriter('episode_out2/' + str(ep) + '_episode_ori.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 60, (image_size, image_size))
            ep += 1
            # video = cv2.VideoWriter(str(e) + '_episode.avi', cv2.VideoWriter_fourcc(*'mp4v'), 60, (256, 240))
            
            # _ = env_save.reset()
            state = env.reset()
            
            # state, reward, done, info = env.step(1)
            # state_save, _, _, _ = env_save.step(1)
            # state_save = env_save.state
            # data = cv2.normalize(state_save / 255., None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # print(state_save.shape, data.shape)
            # gray_3c = cv2.merge([state_save[:,:,0], state_save[:,:,1], state_save[:,:,2]])   
            # video.write(state_save)
            # video_ori.write(state)
            # model_input = resize_img(state)
            # model_input = np.transpose(np.repeat([resize_img(state)], step, axis=0), (1, 2, 0)).reshape(1, image_size, image_size ,step)
            done_flag = True
            h_state = tf.convert_to_tensor(np.zeros(hidden_size).reshape(1, hidden_size), np.float32)

            stay_count = 0
            p_state = 0
            actions = []

            while done_flag:
                current_state = resize_img(state).reshape(1, image_size, image_size, 1)
                # model_input = np.concatenate([model_input[:, :, :, 1:], current_state], axis=3)
                action_prob, value, h_state = model([current_state, h_state], training=False)
                action = np.argmax(action_prob)
                actions.append(action.copy())  
                # print(action_prob, current_state)
                # print(action_prob, value)         

                state, reward, done, info = env.step(action)
                # try:
                # state_save = env_save.state
                # except:
                #     _ = env_save.reset()
                #     state_save, _, _, _ = env_save.step(action)
                # data = cv2.normalize(state_save / 255., None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                gray = cv2.normalize(current_state[0], None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                gray_3c = cv2.merge([gray, gray, gray])
                # video.write(state_save)
                video_ori.write(gray_3c)
                # if done != True: 
                #     state, reward, done, info = env.step(action)
                #     video_ori.write(state)
                # print(info, stay_count)
                if p_state == info['x_pos']:
                    stay_count += 1
                else:
                    stay_count = 0
                p_state = info['x_pos'].copy()

                if (done) | (stay_count > 100):
                    state = env.reset()
                    # _ = env_save.reset()
                    print('reset')
                    done_flag = False

            cv2.destroyAllWindows()
            # video.release()
            video_ori.release()
            print('save_vidio')
            print(actions, done)
            h_state = tf.convert_to_tensor(np.zeros(hidden_size).reshape(1, hidden_size), np.float32)
            

    env.close()


# loss = reward_d * cross_entropy()

if __name__ == '__main__':
    print('[moving]', SIMPLE_MOVEMENT)
    train()