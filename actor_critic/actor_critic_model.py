from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras import models as models
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import tensorflow as tf

import numpy as np
import pandas as pd


class agent:
    def actor_critic_optimizer(self):
        """
        setting actor critic optimizer 

        Returns
        optimizer : K.function
            It is the optimization function for to update parameters using gradient

        """
        action = K.placeholder(shape=[None, 1], name='action')
        advantage = K.placeholder(shape=[None, 1], name='advantage')

        # get output actor, critic
        policy, critic = self.model.output

        # action_prob = K.sum(action * policy, axis=1)
        action_prob = action * policy
        loss = K.log(action_prob + 1e-10) * K.stop_gradient(advantage)
        loss = -K.sum(loss, axis=1)
        # loss = -K.sum(loss)

        # entropy for degrees of freedom
        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)

        # actor loss result
        actor_loss = loss + 0.001 * (entropy)

        target = K.placeholder(shape=[None, 1], name='reward')
        # critic_loss = K.mean(K.sum(K.square(target - critic), axis=1))
        critic_loss = K.mean((K.square(target - critic)))
        print(self.model.output[1].shape)

        # get updates
        updates = self.optimizer.get_updates(params=self.model.trainable_weights, loss=[actor_loss, critic_loss])
        return K.function(inputs=[self.model.input, action, advantage, target], outputs=[actor_loss, critic_loss],
                          updates=updates, name='ac_optimizer')

    def __init__(self, input_shape, modelName, discount_factor=0.99):
        """
        initialization 

        Parameters
        ----------
        input_shape : tuple or int
            dimension of input vector
        modelName : string
            file name of saving model
        discount_factor : float
            discount factor
        """

        self.discount_factor = discount_factor
        input_ = Input(shape=input_shape, name='input_data')
        x = Dense(1024, kernel_initializer='glorot_uniform', activation='relu', name='dense1')(input_)
        x = BatchNormalization(name='bn1')(x)
        x = Dropout(0.25, name='drop1')(x)

        x = Dense(512, kernel_initializer='glorot_uniform', activation='relu', name='dense2')(x)
        x = BatchNormalization(name='bn2')(x)
        x = Dropout(0.25, name='drop2')(x)

        x = Dense(256, kernel_initializer='glorot_uniform', activation='relu', name='dense3')(x)
        x = BatchNormalization(name='bn3')(x)
        x = Dropout(0.25, name='drop3')(x)

        x = Dense(128, kernel_initializer='glorot_uniform', activation='relu', name='dense4')(x)
        x = BatchNormalization(name='bn4')(x)
        x = Dropout(0.25, name='drop4')(x)

        x = Dense(64, kernel_initializer='glorot_uniform', activation='relu', name='dense5')(x)
        x = BatchNormalization(name='bn5')(x)
        self.featureData = Dropout(0.25, name='drop5')(x) # feature output

        self.policyNet = Dense(3, kernel_initializer='glorot_uniform', activation='softmax', name='policyNet')(self.featureData) # pitcher change score
        self.valueNet = Dense(1, kernel_initializer='glorot_uniform', activation='linear', name='valueNet')(self.featureData) # p(pitch count, loss score)

        self.model = models.Model(inputs=input_, outputs=[self.policyNet, self.valueNet])

        import tensorflow
        self.tb_hist = tensorflow.keras.callbacks.TensorBoard(log_dir='./graph_' + modelName, histogram_freq=1, write_graph=True, write_images=True)

        self.optimizer = Adam(lr=2.5e-4, name='adam')#RMSprop(lr=0.001, rho=0.99, epsilon=0.01, momentum=0.0, centered=False, name='rms') #Adam(lr=0.01, name='adam')

        self.model._make_predict_function()

        self.actor_updater = self.actor_critic_optimizer()
        print(self.model.summary(90))


        # for initial train
        losses = {
            'policyNet': 'mse',
            'valueNet': 'mse',
        }
        loss_weights = {'policyNet': 1.0, 'valueNet': 1.0}        
        self.model.compile(loss=losses, loss_weights=loss_weights, optimizer='adam', metrics=['acc'])
        
    def gen(self, data):
        for i in data:
            data = i[0]
            label = K.transpose(i[1])
            yield ({'input_data': data}, {'policyNet': label[0], 'valueNet': label[1]})

    def initial_train(self, train, validation, model):
        from tensorflow.keras.callbacks import ModelCheckpoint
        tb_hist = tf.keras.callbacks.TensorBoard(log_dir='./graph_' + model, histogram_freq=1, write_graph=True,
                                                         write_images=True)
        model_path = './models/' + model + '_{epoch:04d}' + '.h5'  # '{epoch:02d}-{val_loss:.4f}.h5'
        cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, period=1)
        history = self.model.fit_generator(self.gen(train), validation_data=self.gen(validation), epochs=10, steps_per_epoch=30376,
                                           validation_steps=7594, verbose=1, callbacks=[tb_hist, cb_checkpoint])        
        return history

    def update(self, transitions):
        '''
        network update function

        Parameters
        ----------
        transitions : list
            list of learning data

        Returns
        -------
        a_loss : numpy.array
            loss of actor
        c_loss : numpy.array
            loss of critic
        value_targets : list
            reward of appling discount factor
        '''

        reward = 0.0

        states = []
        policy_targets = []
        value_targets = []
        actions = [] # policy action 1 or 0

        # value bulk cal
        tmp_states = []
        # for transition in transitions:
        # print(transitions)
        for j in range(len(transitions)):
            try:
                for i in range(len(transitions[j]['state'])):
                    tmp_tmp_states = []
                    for jj in range(3 - j):
                        tmp_tmp_states += [0] * 33
                    tmp_len = int(len(tmp_tmp_states) / 33)
                    for jj in range(3 - tmp_len, -1, -1):
                        tmp_tmp_states += transitions[j - jj]['state'][i]
                    tmp_states.append(tmp_tmp_states.copy())
                    del tmp_tmp_states
                    # tmp_states.append(transitions[j]['state'][i])
            except:
                log.info(transitions)
                exit(-1)
        tmp_states_np = np.array(tmp_states)
        states_predic = self.model.predict_on_batch([tmp_states_np])
        j = 0
        for transition in transitions:
            transition['value'] = []
            for i in range(len(transition['state'])):
                transition['value'].append(states_predic[1][j][0])
                j += 1
        del states_predic
        del tmp_states
        del tmp_states_np

        # for transition in  transitions[::-1]:
        for j in range(len(transitions) - 1, -1, -1):
            reward = max([-1, min([1, transitions[j]['reward'] + self.discount_factor * reward])])
            for i in range(len(transitions[j]['state'])):
                # policy_target = (reward - self.model.predict([transition['state'][i]])[1][0][0]) # advantage
                policy_target = ([reward - (transitions[j]['value'][i])])
                tmp_states = []
                for jj in range(3 - j):
                    tmp_states += [0] * 33
                tmp_len = int(len(tmp_states) / 33)
                for jj in range(3 - tmp_len, -1, -1):
                    tmp_states += transitions[j - jj]['state'][i]
                # states.append(tmp_states.copy() + transitions[j]['state'][i])
                states.append(tmp_states.copy())
                policy_targets.append(policy_target) # PolicyLoss : log(p) * advantage
                actions.append(transitions[j]['action'][i]) # 선택 되지 않은 state에 대해서는 0
                value_targets.append([reward])
                del policy_target
                del tmp_states

        states_np, actions_np, policy_targets_np, value_targets_np = np.array(states), np.array(actions).reshape([len(actions), 1]), np.array(policy_targets), np.array(value_targets)

        # a_loss, c_loss = self.actor_updater([np.array(states), np.array(actions).reshape([len(actions), 1]), np.array(policy_targets), np.array(value_targets)])
        a_loss, c_loss = self.actor_updater([states_np, actions_np, policy_targets_np, value_targets_np])


        del states_np
        del actions_np
        del policy_targets_np
        del value_targets_np

        del policy_targets
        del states
        del actions
        # del value_targets
        return a_loss, c_loss, value_targets

    def update_model(self, transitions):
        reward = 0.0

        states = []
        policy_targets = []
        value_targets = []
        actions = [] # policy action 1 or 0

        # value bulk cal
        tmp_states = []

        # for transition in  transitions[::-1]:
        for j in range(len(transitions) - 1, -1, -1):
            reward = max([-1, min([1, transitions[j]['reward'] + self.discount_factor * reward])])
            
            # policy_target = (reward - self.model.predict([transition['state'][i]])[1][0][0]) # advantage
            policy_target = ([reward - (transitions[j]['value'])])            
            states.append(transitions[j]['state'])
            policy_targets.append(policy_target) # PolicyLoss : log(p) * advantage
            actions.append(transitions[j]['action']) # 선택 되지 않은 state에 대해서는 0
            value_targets.append(reward)
            del policy_target
            # del tmp_states

        # states_np, actions_np, policy_targets_np, value_targets_np = np.array(states), np.array(actions).reshape([len(actions), 1]), np.array(policy_targets), np.array(value_targets)
        states_np, actions_np, policy_targets_np, value_targets_np = np.array(states), np.array(actions), np.array(policy_targets), np.array(value_targets)

        # a_loss, c_loss = self.actor_updater([np.array(states), np.array(actions).reshape([len(actions), 1]), np.array(policy_targets), np.array(value_targets)])
        a_loss, c_loss = self.actor_updater([states_np, actions_np, policy_targets_np, value_targets_np])


        del states_np
        del actions_np
        del policy_targets_np
        del value_targets_np

        del policy_targets
        del states
        del actions
        # del value_targets
        return a_loss, c_loss

# observation, reward, done, info = env.step(action)

