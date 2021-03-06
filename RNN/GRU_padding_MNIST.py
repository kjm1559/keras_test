from __future__ import print_function

import tensorflow.keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Lambda, Input, GRU, Masking, TimeDistributed
from tensorflow.keras.losses import mse, binary_crossentropy


# from tensorflow.keras.layers import CuDNNLSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import model_from_json
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
from tensorflow.keras.datasets import mnist

from sklearn.utils import class_weight
import numpy as np

def attention_mechanism(input):
    x = Dense(6, activation='softmax')(input)
    return x


class nn_():
    '''
    simple neural networks class, loss is binary_crossentropy, adam optimization function
    '''

    def __init__(self, input_dim, batch_size, epochs):
        '''
        Init function

        Parameters
        ----------
        input_dim : int
            Dimension of input vector
        batch_size : int
            Batch size
        epochs : int
            epoch repeat

        Example
        -------
        >>> network = nn_((None, 32), 16, 100)

        '''
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.epochs = epochs

        inputs = Input(shape=self.input_dim, name='encoder_input')
        x = Masking(mask_value=0, input_shape=input_dim)(inputs)
        x = GRU(256, name='GRU_layer1', return_sequences=True)(x)
        # x = attention_mechanism(x)
        # x = GRU(256, name='GRU_layer2', return_sequences=True)(x)
        # outputs = Dense(8, activation='sigmoid')(x)
        # x = Masking(mask_value=0, input_shape=input_dim)(x)
        outputs = TimeDistributed(Dense(28, activation='sigmoid'))(x)

        self.model = Model(inputs, outputs, name='vae_mlp')
        self.model.compile(optimizer='adam', loss='mae')
        print(self.model.summary())

    def make_train_data_full_sequence(self, train_data):
        """
        Create train data   

        """
        

        seq_data = train_data[:, :-1, :]
        seq_label = train_data[:, 1:, :]

        return seq_data, seq_label

        # return index_data, data, input_shape, feature_list, feature_max

    def train_(self, model):
        """
        Create data and train model 
        model + '_feature_max.pkl'
        './models/' + model + '.h5'
        model + '_train_result.pkl'

        Parameters
        ----------
        df_data : pandas.DataFrame
            KBO_Batter_List.pkl or KBO_Pitcher_List.pkl for training            
        seq_len : int
            sequence length for learning sequence
        model : string
            save model name
        player_flag : string
            'p' or 'b'
        """
        import pickle
        import gzip

        # mnist train 60000 x 28 x 28
        # mnist test 10000 x 28 x 28
        (x_train, _), (x_test, _) = mnist.load_data()
        x_train = x_train / 255
        x_test = x_test / 255
        


        train_data, label_data = self.make_train_data_full_sequence(x_train)

        import tensorflow.keras
        max_ = 0
        for i in range(len(train_data)):
            if max_ < len(train_data):
                max_ = len(train_data)
        print(max_)

        print(train_data[0][0], train_data[1][0])
        train_data = tensorflow.keras.preprocessing.sequence.pad_sequences(train_data, maxlen=28, dtype='float32', padding='post')#.astype(float)
        label_data = tensorflow.keras.preprocessing.sequence.pad_sequences(label_data, maxlen=28, dtype='float32', padding='post')#.astype(float)

        print(train_data[0][0], train_data[1][0])

        
        self.train(train_data, label_data, model)
        self.model.load_weights('./' + model + '.h5')

        sample_index = np.random.randint(10000)
        test = x_test[sample_index]
        test_30 = test[:int(28 * 0.3)]
        test_50 = test[:int(28 * 0.5)]
        test_30 = tensorflow.keras.preprocessing.sequence.pad_sequences([test_30], maxlen=28, dtype='float32', padding='post')
        test_50 = tensorflow.keras.preprocessing.sequence.pad_sequences([test_50], maxlen=28, dtype='float32', padding='post')
        test_30_ori = test_30.copy()
        test_50_ori = test_50.copy()

        # 30% data
        for i in range(int(28 * 0.3), 28):            
            tmp_data = self.model.predict_on_batch(test_30)            
            test_30[0][i] = tmp_data[0][i]

        # 50% data
        for i in range(int(28 * 0.5), 28):
            tmp_data = self.model.predict(test_50)            
            test_50[0][i] = tmp_data[0][i]

        # save image 
        from PIL import Image
        original = np.concatenate([test * 255, test], axis=1)
        con_30 = np.concatenate([test_30_ori[0] * 255, test_30[0] * 255], axis=1)
        con_50 = np.concatenate([test_50_ori[0] * 255, test_50[0] * 255], axis=1)
        target_img = Image.fromarray(np.concatenate([original, con_30, con_50], axis=0))
        target_img.save('result.gif')

        # target_img = Image.fromarray(np.concatenate([test * 255, test], axis=1))
        # target_img.save('original.gif')
        # target_img = Image.fromarray(np.concatenate([test_30_ori[0] * 255, test_30[0] * 255], axis=1))
        # target_img.save('predict_30.gif')
        # target_img = Image.fromarray(np.concatenate([test_50_ori[0] * 255, test_50[0] * 255], axis=1))
        # target_img.save('predict_50.gif')

        # y_predict = self.model.predict(train_data)

        # with gzip.open(model + '_train_result.pkl', 'wb') as f:
        #     pickle.dump([train_data, label_data, y_predict], f)

    def train(self, X_train, y_train, model):
        '''
        Training function of nn_ 

        Parameters
        ----------
        X_train : numpy.array
            input vector
        y_train : numpy.array
            label
        model : String
            model name

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

    def test(self, X_test, y_test):
        '''
        validation function

        Parameters
        ----------
        X_test : numpy.array
            test input data
        y_test : numpy.array
            test label

        Examples
        --------
        >>> network.test(X_test, y_test)

        '''
        score = self.vae.evaluate(X_test, y_test, batch_size=self.batch_size)
        print('\nTest loss:', score[0])
        print('Test accuracy:', score[1])


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
            print(e)
    
    test = nn_((None, 28), 32, 200)
    test.train_('gru_MNIST_sequence')