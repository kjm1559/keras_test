from __future__ import print_function

import tensorflow.keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Lambda, Input
from tensorflow.keras.losses import mse, binary_crossentropy


# from tensorflow.keras.layers import CuDNNLSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import model_from_json
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K

from sklearn.utils import class_weight
import numpy as np

# from imblearn.keras import balanced_batch_generator
# from imblearn.under_sampling import NearMiss
# from imblearn.under_sampling import TomekLinks

def arithmetic_cal(string):
    if type(string) == np.float64:
        return string
    for i in range(len(string)):
        if i == len(string) - 1:
            return float(string)
        if string[i] == ' ':
            return float(string[:i]) + arithmetic_cal(string[i + 1:])
        elif string[i] == '/':
            return float(string[:i]) / arithmetic_cal(string[i + 1:])

def pitcher_inning_string2float(innings):
    float_inning = []
    for inn in innings:
        float_inning.append(arithmetic_cal(inn))
    return float_inning

def get_pitcher_days(game_id_list):
    from datetime import datetime
    pitcher_days = []
    for i in range(len(game_id_list)):
        if i == 0:
            pitcher_days.append(1)
        else:
            date1 = datetime(int(game_id_list[i][:4]), int(game_id_list[i][4:6]), int(game_id_list[i][6:8]))
            date2 = datetime(int(game_id_list[i - 1][:4]), int(game_id_list[i - 1][4:6]),
                             int(game_id_list[i - 1][6:8]))
            pitcher_days.append(min([1, (date1 - date2).days / 5]))
    return pitcher_days

def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args 
    batch = K.shape(z_mean)[0]
    # batch = z_mean.shape[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

class nn_():
    '''
    간한한 neural networks class, binary_crossentropy를 loss로 사용하며, adam을 optimization function으로 사용
    '''
    def __init__(self,input_dim, batch_size, epochs):
        '''
        nn_을 초기화 하기 위한 함수

        Parameters
        ----------
        input_dim : int
            입력 vector의 차원
        batch_size : int
            Batch의 크기
        epochs : int
            epoch 반복 횟수

        Example
        -------
        >>> network = nn_((None, 32), 16, 100)

        '''
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.epochs = epochs

        inputs = Input(shape=(self.input_dim, ), name='encoder_input')
        x = Dense(128, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        z_mean = Dense(8, name='z_mean')(x)
        z_log_var = Dense(8, name='z_log_var')(x)

        z = Lambda(sampling, output_shape=(8, ), name='z')([z_mean, z_log_var])

        self.encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        print(self.encoder.summary())
        # from_keras.utils import plot_model
        # plot_model(self.encoder, to_file='vae_mlp_encoder.jpg', show_shapes=True)

        latent_inputs = Input(shape=(8, ), name='z_sampling')
        x = Dense(32, activation='relu')(latent_inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(self.input_dim, activation='sigmoid')(x)

        self.decoder = Model(latent_inputs, outputs, name='decoder')
        print(self.decoder.summary())

        outputs = self.decoder(self.encoder(inputs)[2])
        self.vae = Model(inputs, outputs, name='vae_mlp')       

        reconstruction_loss = mse(inputs, outputs)
        reconstruction_loss *= self.input_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        # kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss = K.mean(kl_loss, axis=-1)
        # kl_loss *= -0.5
        kl_loss *= -5e-4
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer='adam')
        print(self.vae.summary())



    def make_train_data(self, df_data, seq_len, player_flag):
        """
        train data를 만들어내는 함수

        Parameters
        ----------
        df_data : pandas.DataFrame
            KBO_Batter_List.pkl 또는 KBO_Pithcer_List.pkl
        seq_len : int
            Player2Vec을 계산하기 위한 sequence 길이
        player_flag : string
            'p' or 'b'

        Returns
        -------
        index_data : list
            [game_id, player_id]의 list
        data : numpy.array
            학습에 사용할 encoded data
        input_shape : tuple
            학습 데이터의 vector dimension
        feature_list : list
            학습에 사용하는 feature
        feature_max : list
            하습에 사용하는 feature의 max값

        """
        from tqdm import tqdm
        player_list = df_data.PlayerID.unique()
        # |삼진|타점|타석|1B|2B|3B|HR|HI|KW|GR|HB|BB|IB|SH|SF|SB|WP|HP|
        # | 5 | 9 | 8 | 6| 4| 3| 4| 3| 1| 5| 2| 5| 3| 3| 3| 4| 6| 3|
        if player_flag == 'b':
            feature_list = ['삼진', '타점', '타석', '1B', '2B', '3B', 'HR', 'HI', 'KW', 'GR', 'HB', 'BB', 'IB', 'SH',
                            'SF', 'SB', 'WP', 'HP']
            df_data['타점'].astype(int)
        elif player_flag == 'p':
            feature_list = ['승', '패', '세', '이닝', '타수', '피안타', '홈런', '4사구', '삼진', '실점', '자책']  # , '평균자책점']
            df_data['이닝'] = pitcher_inning_string2float(df_data['이닝'].values)
        #         df_data['평균자책점'].astype(float)

        feature_max = df_data[feature_list].max()
        max_dict = {feature: feature_max[feature] for feature in feature_list}

        index_data = []
        data = []
        if player_flag == 'p':
            feature_list.append('휴식기')
            max_dict['휴식기'] = 5

        for player in tqdm(player_list, position=0):
            tmp_df_data = df_data[df_data.PlayerID == player].sort_values(by='GameID')
            tmp_seq_encoding_data = [[0] * len(feature_list)] * seq_len
            if player_flag == 'p':
                tmp_df_data['휴식기'] = get_pitcher_days(tmp_df_data.GameID.values)
            for index, idata in tmp_df_data.iterrows():
                tmp_encoding_data = []
                for key in feature_list:
                    tmp_encoding_data.append(idata[key] / max_dict[key])
                tmp_seq_encoding_data.append(tmp_encoding_data)
                del tmp_seq_encoding_data[0]
                data.append(sum(tmp_seq_encoding_data, []))
                #             print(idata)
                index_data.append([idata['GameID'], idata['PlayerID']])
        data = np.array(data)
        input_shape = data.shape[1:]

        return index_data, data, input_shape, feature_list, feature_max

    def train_(self, df_data, seq_len, model, player_flag):
        """
        데이터를 생성하여 학습을 진행하는 function
        model + '_feature_max.pkl'
        './models/' + model + '.h5'
        model + '_train_result.pkl'

        Parameters
        ----------
        df_data : pandas.DataFrame
            학습을 하기 위한 KBO_Batter_List.pkl or KBO_Pitcher_List.pkl
        seq_len : int
            sequence length for learning sequence
        model : string
            save model name
        player_flag : string
            'p' or 'b'
        """
        index_data, train_data, input_shape, feature_list, feature_max = self.make_train_data(df_data, seq_len, player_flag)
        with gzip.open(model + '_feature_max.pkl', 'wb') as f:
            pickle.dump([feature_max], f)
        self.train(train_data, [], model)

        self.vae.load_weights('./models/' + model + '.h5')
        intermediate_model = Model(inputs=[test_nn.encoder.input], outputs=[test_nn.encoder.output])
        y_predict = self.vae.predict(train_data)
        y_latent = intermediate_model.predict(train_data)
        with gzip.open(model + '_train_result.pkl', 'wb') as f:
            pickle.dump([index_data, train_data, y_latent, y_predict], f)

    def train(self, X_train, y_train, model):
        '''
        nn_ 모델의 학습 함수

        Parameters
        ----------
        X_train : numpy.array
            학습에 들어갈 input vector
        y_train : numpy.array
            학습에 사용할 label
        model : String
            저장될 모델의 이름

        Examples
        --------
        >>> network.train(X_train, y_train)


        '''
        tb_hist = tensorflow.keras.callbacks.TensorBoard(log_dir='./graph_' + model, histogram_freq=1, write_graph=True, write_images=True)

        # training_generator = training_generator = BalancedBatchGenerator(X_train, y_train,
        #                                         batch_size=64,
        #                                         random_state=42)
        # training_generator, steps_per_epoch = balanced_batch_generator(
        #     X_train, y_train, sampler=NearMiss(), batch_size=self.batch_size
        # )
        # training_generator, steps_per_epoch = balanced_batch_generator(
        #     X_train, y_train, sampler=TomekLinks(), batch_size=self.batch_size
        # )

        # print(class_weights)
        # self.model.fit_generator(generator=training_generator, epochs=20, verbose=1)
        # histroy = self.model.fit_generator(generator=training_generator, steps_per_epoch=steps_per_epoch, epochs=self.epochs, verbose=1, callbacks=[tb_hist])
        
        model_path = './models/' + model + '.h5'#'{epoch:02d}-{val_loss:.4f}.h5'
        cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
        # early_stopping = EarlyStopping(patience=10)

        history = self.vae.fit(X_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1, shuffle=True, validation_split=0.2, callbacks=[tb_hist, cb_checkpoint])#, class_weight=class_weights)
        self.save_model_weight(model)
        return history

    def save_model_weight(self, model_name):
        '''
        모델과 모델 wieght를 저장하는 함수

        Parameters
        ----------
        model_name : String
            모델 저장시 사용할 이름

        Examples
        --------
        >>> network.save_model_weight('test_model')

        '''
        model_json = self.vae.to_json()
        with open('./models/' + model_name + '.json', 'w') as json_file:
            json_file.write(model_json)
        self.vae.save_weights('./models/' + model_name + '_last' + '.h5')

    def load_model_weight(self, model_name):
        '''
        모델과 모델 wieght를 불러오는 함수

        Parameters
        ----------
        model_name : String
            모델 불러올 때 사용할 이름

        Examples
        --------
        >>> network.load_model_weight('test_model')

        '''
        json_file = open('./models/' + model_name + '.json', 'r')
        loaded_model_json = json_file.read()
        self.vae = model_from_json(loaded_model_json)
        self.vae.load_weights('./models/' + model_name + '.h5')

    def test(self, X_test, y_test):
        '''
        validation을 하기 위한 함수 사실은 prediction 함수를 이용하여 결과를 얻어야함

        Parameters
        ----------
        X_test : numpy.array
            테스트를 위한 input data
        y_test : numpy.array
            테스트를 위한 label

        Examples
        --------
        >>> network.test(X_test, y_test)

        '''
        score = self.vae.evaluate(X_test, y_test, batch_size=self.batch_size)
        print('\nTest loss:', score[0])
        print('Test accuracy:', score[1])


if __name__ == '__main__':
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras import backend as K
    from tensorflow.keras.models import Model
    import tensorflow as tf
    # tf.config.gpu.set_per_process_memory_fraction(0.10)
    # tf.config.gpu.set_per_process_memory_growth(True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            print('ggg')
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
            print('hello')
            print(e)
    batch_size = 32
    num_classes = 2
    epochs = 200
    k = 128#61
    l = 256#183
    seq_len = 7

    # (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # X_train = X_train.reshape(60000, 784)
    # X_test = X_test.reshape(10000, 784)
    # X_train = X_train.astype('float32')
    # X_test = X_test.astype('float32')
    # X_train /= 255
    # X_test /= 255
    # print(X_train.shape[0], 'train samples')
    # print(X_test.shape[0], 'test samples')

    # X_train = X_train[(y_train == 1) | (y_train == 0)]
    # y_train = y_train[(y_train == 1) | (y_train == 0)]

    # X_test = X_test[(y_test == 1) | (y_test == 0)]
    # y_test = y_test[(y_test == 1) | (y_test == 0)]

    # time_X_train
    # time_X_train = []
    # for i in range(seq_len, len(X_train)):
    #     time_X_train.append(X_train[i - seq_len: i])
    # time_X_train = np.array(time_X_train)
    # y_train = y_train[10:]
    # print(time_X_train.shape, X_train.shape)

    # y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
    # y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

    # index = np.arange(len(X_train))
    # np.random.shuffle(index)
    # X_train = X_train[index]
    # y_train = y_train[index]
    # print(y_train)

    
    import gzip
    import pickle
    import sys
    if sys.argv[1] == 'p':
        print('pitcher')
        with gzip.open('no_id_pitcher_autoencoder_train_data.pkl', 'rb') as f:
            X_train = pickle.load(f)        

        test_nn = nn_(65, batch_size, epochs)

        test_nn.train(X_train, [], 'no_id_pitcher_vae')

        # test_nn.test(X_test, y_test, 'testtest')
        # test_nn.encoder.load_weights('./models/pitcher_autoencoder.h5')
        test_nn.vae.load_weights('./models/no_id_pitcher_vae.h5')

        intermediate_model = Model(inputs=[test_nn.encoder.input], outputs=[test_nn.encoder.output])    
        y_predict = test_nn.vae.predict(X_train)
        y_latent = intermediate_model.predict(X_train)
        print(y_predict[0] - X_train[0], y_latent[0])
        with gzip.open('no_id_pitcher_vae_train_latent.pkl', 'wb') as f:
            pickle.dump([y_latent, y_predict], f)
    elif sys.argv[1] == 'b':
        import pandas as pd
        bl = pd.read_pickle('~/KBOProject/KBO_Batter_List.pkl')
        bl = bl[bl.GameID.str.contains('2019')]
        test_nn = nn_(54, batch_size, epochs)
        test_nn.train_(bl, 3, 'new_batter_model', 'b')

    elif sys.argv[1] == 'pp':
        import pandas as pd
        pl = pd.read_pickle('~/KBOProject/KBO_Pitcher_List.pkl')
        pl = pl[pl.GameID.str.contains('2019')]
        test_nn = nn_(36, batch_size, epochs)
        test_nn.train_(pl, 3, 'new_pitcher_model', 'p')

    else:
        print('batter')
        with gzip.open('event_no_order_no_id_batter_autoencoder_train_data.pkl', 'rb') as f:
            X_train = pickle.load(f)        

        test_nn = nn_(21, batch_size, epochs)

        test_nn.train(X_train, [], 'no_id_no_order_batter_vae')

        # test_nn.encoder.load_weights('./models/batter_autoencoder.h5')
        test_nn.vae.load_weights('./models/no_id_no_order_batter_vae.h5')

        intermediate_model = Model(inputs=[test_nn.encoder.input], outputs=[test_nn.encoder.output])    
        y_predict = test_nn.vae.predict(X_train)
        y_latent = intermediate_model.predict(X_train)
        print(y_predict[0] - X_train[0], y_latent[0])
        with gzip.open('batter_vae_train_latent.pkl', 'wb') as f:
            pickle.dump([y_latent, y_predict], f)

    # print(len(y_flatten), len(X_test))


    
# from keras.layers import Input, Dense, concatenate
# from keras.models import Model

# inputs = Input(shape=(100,))
# dnn = Dense(1024, activation='relu')(inputs)
# dnn = Dense(128, activation='relu', name="layer_x")(dnn)
# dnn = Dense(1024, activation='relu')(dnn)
# output = Dense(10, activation='softmax')(dnn)

# model_a = Model(inputs=inputs, outputs=output)

# # You don't need to recreate an input for the model_a, 
# # it already has one and you can reuse it
# input_b = Input(shape=(200,))

# # Here you get the layer that interests you from model_a, 
# # it is still linked to its input layer, you just need to remember it for later
# intermediate_from_a = model_a.get_layer("layer_x").output

# # Since intermediate_from_a is a layer, you can concatenate it with the other input
# merge_layer = concatenate([input_b, intermediate_from_a])
# dnn_layer = Dense(512, activation="relu")(merge_layer)
# output_b = Dense(5, activation="sigmoid")(dnn_layer)
# # Here you remember that one input is input_b and the other one is from model_a
# model_b = Model(inputs=[input_b, model_a.input], outputs=output_b)
