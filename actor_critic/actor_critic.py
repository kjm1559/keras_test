import tensorflow.keras
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Conv2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam, RMSprop
import numpy as np

def shared_model(input_dim, output_dim):
    inputs = Input(shape=(input_dim,), name='encoder_input')    
    x = Dense(1024, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Dense(256, activation='relu')(x)    
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Dense(64, activation='relu')(x)
    actor = Dense(output_dim, activation='softmax', name='actor')(x)
    critic = Dense(1, activation='linear', name='critic')(x)
    return Model(inputs, [actor, critic], name='actor_critic')

def shared_model_conv(input_dim, output_dim):
    inputs = Input(shape=input_dim, name='encoder_input')    
    x = Conv2D(32, 8, strides=4, activation='relu')(inputs)
    x = Conv2D(64, 4, strides=3, activation='relu')(x)
    x = Conv2D(64, 3, strides=1, activation='relu')(x)

    # x = Conv1D(128, 8, strides=2, activation='relu')(x)
    # x = Conv1D(256, 4, strides=2, activation='relu')(x)
    # x = Conv1D(512, 4, strides=2, activation='relu')(x)
    # x = Conv1D(1024, 2, strides=1, activation='relu')(x)
    
    x = Flatten()(x)
    # x = Dense(1024, activation='relu')(inputs)
    # x = BatchNormalization()(x)
    # x = Dropout(0.25)(x)
    # x = Dense(256, activation='relu')(x)    
    # x = BatchNormalization()(x)
    # x = Dropout(0.25)(x)
    # x = Dense(128, activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.25)(x)
    x = Dense(512, activation='relu')(x)
    actor = Dense(output_dim, activation='softmax', name='actor')(x)
    critic = Dense(1, activation='linear', name='critic')(x)
    return Model(inputs, [actor, critic], name='actor_critic')

def each_model(input_dim, output_dim):
    inputs = Input(shape=(input_dim,), name='encoder_input')
    # x = Dense(2048, activation='relu')(inputs)
    x = Dense(1024, activation='relu')(inputs)
    # x = BatchNormalization()(x)
    # x = Dropout(0.3)(x)
    x = Dense(512, activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)

    x = Dense(64, activation='relu')(x)

    actor = Dense(output_dim, activation='softmax')(x)
    critic = Dense(1, activation='linear')(x)

    return Model(inputs, actor, name='actor'), Model(inputs, critic, name='critic') 


class actor_critic:
    def __init__(self, input_dim, output_dim):      
        self.input_dim = input_dim
        self.out_dim = output_dim
        self.gamma = 0.99
        
        # self.actor_critic = shared_model(self.input_dim, self.out_dim)
        self.actor_critic = shared_model_conv(self.input_dim, self.out_dim)
        # for l in self.actor_critic.layers:
        #     print(l.name, l.trainable)

        # self.actor, self.critic = each_model(self.input_dim, self.out_dim)

        # self.adam_optimizer = RMSprop(lr=0.0001, rho=0.99, epsilon=0.01)      
        self.adam_optimizer = Adam(lr=7e-4)
        self.opt = self.optimizer()
        # self.actor_opt = self.actor_optimizer()
        # self.critic_opt = self.critic_optimizer()

      
    def optimizer(self):
        """ Actor Optimization: Advantages + Entropy term to encourage exploration
        (Cf. https://arxiv.org/abs/1602.01783)
        """
        actor, critic = self.actor_critic(self.actor_critic.input)

        action = K.placeholder(shape=(None, self.out_dim))
        advantages = K.placeholder(shape=(None,))
        weighted_actions = K.sum(action * actor, axis=1)
        eligibility = K.log(weighted_actions + 1e-10) * K.stop_gradient(advantages)        
        entropy = K.sum(actor * K.log(actor + 1e-10), axis=1)
        entropy = K.sum(entropy)        
        actor_loss = 1.0e-4 * entropy - K.sum(eligibility)
        # actor_loss = 1.0e-4 * entropy - K.cast(K.sum(eligibility), 'float32')

        discounted_reward = K.placeholder(shape=(None,))
        critic_loss = 0.5 * K.mean(K.square(discounted_reward - critic))
        # loss = actor_loss + critic_loss
        # updates = self.adam_optimizer.get_updates(loss=loss, params=self.actor_critic.trainable_weights)
        # return K.function(inputs=[self.actor_critic.input, action, advantages, discounted_reward], \
        #                     outputs=loss, updates=updates)
        updates = self.adam_optimizer.get_updates(loss=[actor_loss, critic_loss], params=self.actor_critic.trainable_weights)
        return K.function(inputs=[self.actor_critic.input, action, advantages, discounted_reward], \
                            outputs=[actor_loss, critic_loss], updates=updates)

    def actor_optimizer(self):
        # actor, _ = self.actor_critic(self.actor_critic.input)
        # self.actor_critic.layers[-1].trainable = False
        # self.actor_critic.layers[-2].trainable = True
        actor = self.actor(self.actor.input)

        action = K.placeholder(shape=(None, self.out_dim))
        advantages = K.placeholder(shape=(None,))
        weighted_actions = K.sum(action * actor, axis=1)
        eligibility = K.log(weighted_actions + 1e-10) * K.stop_gradient(advantages)
        entropy = K.sum(actor * K.log(actor + 1e-10), axis=1)
        # entropy = K.sum(entropy)
        actor_loss = (- eligibility) + 1.0e-3 * entropy      

        # updates = self.adam_optimizer.get_updates(loss=[actor_loss], params=self.actor_critic.trainable_weights)
        # return K.function([self.actor_critic.input, action, advantages], [actor_loss], updates=updates)
        updates = self.adam_optimizer.get_updates(loss=[actor_loss], params=self.actor.trainable_weights)
        return K.function([self.actor.input, action, advantages], [actor_loss], updates=updates)

    def critic_optimizer(self):
        # _, critic = self.actor_critic(self.actor_critic.input)
        # self.actor_critic.layers[-1].trainable = True
        # self.actor_critic.layers[-2].trainable = False
        critic = self.critic(self.critic.input)
        
        discounted_reward = K.placeholder(shape=(None,))
        critic_loss = K.mean(K.square(discounted_reward - critic))

        # updates = self.adam_optimizer.get_updates(loss=[critic_loss], params=self.actor_critic.trainable_weights)
        # return K.function([self.actor_critic.input, discounted_reward], [critic_loss], updates=updates)
        updates = self.adam_optimizer.get_updates(loss=[critic_loss], params=self.critic.trainable_weights)
        return K.function([self.critic.input, discounted_reward], [critic_loss], updates=updates)


    def train_models(self, states, actions, rewards):
        """ Update actor and critic networks from experience
        """
        # Compute discounted rewards and Advantage (TD. Error)
        # reshape channel
        states = np.array(states)
        # states = states.reshape(states.shape[0], 4, 84*84)
        # states = np.swapaxes(states, 1, 2)
        
        discounted_rewards = self.discount(rewards)
        policy, values = self.actor_critic.predict_on_batch(np.array(states))
        # policy, values = self.actor.predict_on_batch(np.array(states)), self.critic.predict_on_batch(np.array(states))
        advantages = np.array(discounted_rewards) - np.reshape(values, len(values)) 
        discounted_rewards = discounted_rewards 
        # print(np.max(states), np.max(actions), np.max(advantages), np.max(discounted_rewards))
        # print(values.tolist())
        # weighted_actions = K.sum(actions * policy, axis=1)
        # eligibility = K.log(weighted_actions + 1e-10) * advantages
        # entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        # entropy = K.sum(entropy)
        # actor_loss = 1.0e-4 * entropy - K.cast(K.sum(eligibility), 'float32')
        # print(eligibility.numpy().tolist())
        # print(actor_loss, 1.0e-4 * entropy)
        # discounted_rewards = np.clip(discounted_rewards, -1. ,1.)
        # print(max(advantages))
        # Networks optimization
        #   actor_loss = self.actor_opt([states, actions, advantages])
        #   critic_loss = self.critic_opt([states, discounted_rewards])
        #   return actor_loss, critic_loss
        return self.opt([states, np.array(actions), advantages, discounted_rewards]) 

    def discount(self, r):
        """ Compute the gamma-discounted rewards over an episode
        """
        discounted_r, cumul_r = np.zeros_like(r), 0
        for t in reversed(range(0, len(r))):
            cumul_r = r[t] + cumul_r * self.gamma
            discounted_r[t] = cumul_r
        return discounted_r     