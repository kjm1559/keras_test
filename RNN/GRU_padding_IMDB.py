from __future__ import print_function

import tensorflow.keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Lambda, Input, GRU, Masking, TimeDistributed, Embedding, Lambda, Permute
from tensorflow.keras.losses import mse, binary_crossentropy


# from tensorflow.keras.layers import CuDNNLSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import model_from_json
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
from tensorflow.keras.datasets import imdb

from sklearn.utils import class_weight
import numpy as np

from gensim.models import Word2Vec
import gensim
from tqdm import tqdm

MAX_SEQUENCE = 400

def attention_mechanism(input):
    x = Dense(MAX_SEQUENCE, activation='softmax')(input)
    return x

class nn_():
    '''
    간한한 neural networks class, binary_crossentropy를 loss로 사용하며, adam을 optimization function으로 사용
    '''

    def __init__(self, input_dim, batch_size, epochs):
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

        inputs = Input(shape=self.input_dim, name='encoder_input')
        # x = K.one_hot(inputs, 100)
        # x = Embedding(100, 64, input_length=250, mask_zero=True)(inputs)
        # x = OneHot(input_dim=100, input_length=250)(inputs)
        # print(x.shape)
        x = Masking(mask_value=0, input_shape=(None, 100))(inputs)
        x = GRU(256, name='GRU_layer1', return_sequences=True)(x)
        # x = GRU(256, name='GRU_layer1')(x)
        attention_x = attention_mechanism(x)
        gru_out = Permute((2, 1))(x)
        attention_mul = K.batch_dot(gru_out, attention_x)
        attention_mul = Permute((2, 1))(attention_mul)
        # x = GRU(256, name='GRU_layer2', return_sequences=True)(x)
        x = GRU(256, name='GRU_layer2')(attention_mul)
        outputs = Dense(1, activation='sigmoid')(x)
        # x = Masking(mask_value=0, input_shape=input_dim)(x)
        # outputs = TimeDistributed(Dense(100, activation='tanh'))(x)
        print(outputs.shape)

        self.model = Model(inputs, outputs, name='vae_mlp')
        self.model.compile(optimizer='adam', loss='mse', metrics=['acc'])
        print(self.model.summary())

    def make_train_data_full_sequence(self, train_data, model):
        """
        train data를 만들어내는 함수       

        """

        seq_data = []
        for i in range(len(train_data)):
            tmp_data = []
            for j in range(len(train_data[i])):
                try:
                    tmp_data.append(model.wv[train_data[i][j]].tolist())
                except:
                    print('not exist :', train_data[i][j])
                    
            seq_data.append(tmp_data)

        seq_data = tensorflow.keras.preprocessing.sequence.pad_sequences(seq_data, maxlen=MAX_SEQUENCE, dtype='float32', padding='post')#.astype(float)
        seq_data = seq_data[:, :, :]

        return seq_data

        # return index_data, data, input_shape, feature_list, feature_max

    def train_(self, model):
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
        import pickle
        import gzip

        (x_train, y_train), (x_test, y_test) = imdb.load_data()#num_words=100)

        for i in range(len(x_train)):
            for j in range(len(x_train[i])):
                x_train[i][j] = str(x_train[i][j])

        try:
            models = Word2Vec.load("imdb_word2vec.model")
        except:
            models = Word2Vec(x_train, size=100, window=5, min_count=1)
            models.save("imdb_word2vec.model")
        print('load word2vec complete')

        train_data = self.make_train_data_full_sequence(x_train, models)
        
        train_data = tensorflow.keras.preprocessing.sequence.pad_sequences(train_data, maxlen=MAX_SEQUENCE, dtype='float32', padding='post')#.astype(float)        
        
        self.train(train_data, y_train, model)
        self.model.load_weights('./' + model + '.h5')

        for i in range(len(x_test)):
            for j in range(len(x_test[i])):
                x_test[i][j] = str(x_test[i][j])

        test_data = self.make_train_data_full_sequence(x_test, models)
        predict = self.model.predict(test_data)

        correct = 0
        for i in range(len(y_test)):
            if (y_test[i] == 0) & (predict[i] < 0.5):
                correct += 1
            if (y_test[i] == 1) & (predict[i] >= 0.5):
                correct += 1
        print('acc :', correct / len(y_test))

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
        tb_hist = tensorflow.keras.callbacks.TensorBoard(log_dir='./graph_' + model, histogram_freq=1, write_graph=True,
                                                         write_images=True)

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

        model_path = './' + model + '.h5'  # '{epoch:02d}-{val_loss:.4f}.h5'
        cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
        early_stopping = EarlyStopping(patience=10)

        history = self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1, shuffle=True,
                               validation_split=0.2,
                               callbacks=[tb_hist, cb_checkpoint, early_stopping])  # , class_weight=class_weights)
        # self.model.save_weights('./models/' + model + '.h5')
        return history


if __name__ == '__main__':
    import pandas as pd
    import tensorflow as tf

    # tf.config.gpu.set_per_process_memory_fraction(0.10)
    # tf.config.gpu.set_per_process_memory_growth(True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
            print(e)
    
    test = nn_((None, 100), 32, 200)
    test.train_('gru_IMDB_sequence_attention')