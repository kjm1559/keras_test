import numpy as np
from datetime import datetime
from datetime import timedelta
from tensorflow.keras.layers import GRU, Input, Permute, Reshape, Dense, Multiply, dot, Lambda
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

import tensorflow as tf
from datetime import datetime
import sys

path = 'backup_' + datetime.now().strftime('%Y%m%d') + '/'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

string_index = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', \
                'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',\
                'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', \
                'w', 'x', 'y', 'z', '-', ' ', ',']

TIME_STEPS = 30

def encoding_string(string, string_index):
    if len(string) < TIME_STEPS:
        for i in range(TIME_STEPS - len(string)):
            string += ' '
    encoded_data = np.zeros((len(string), len(string_index)))
    for i in range(len(string)):
        encoded_data[i][string_index.index(string[i])] = 1
    return encoded_data

def decoding_string(encoded_data, string_index):
    string = ''
    for idata in encoded_data:
        string += string_index[np.argmax(idata)]
    return string

def attention_mechanism(input):
    x = Dense(TIME_STEPS, activation='softmax')(input)
    return x

def attention_model(input_shape):
    input_ = Input(shape=(TIME_STEPS, input_shape,), name='input_data')
    gru_out = GRU(256, return_sequences=True, name='encode_gru')(input_)
    print('gru', gru_out.shape)
    attention_x = attention_mechanism(gru_out)
    gru_out = Permute((2, 1))(gru_out)
    attention_mul = K.batch_dot(gru_out, attention_x)
    attention_mul = Permute((2, 1))(attention_mul)
    output = GRU(input_shape, return_sequences=True, name='decode_gru')(K.reverse(attention_mul, axes=1))
    if sys.argv[1] == 'train':
        model = Model(inputs=input_, outputs=output)
    elif sys.argv[1] == 'test':
        model = Model(inputs=input_, outputs=[output, attention_x])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def make_train_data():
    train_X = []
    train_y = []
    td = timedelta(days=1)
    date = datetime(1970, 1, 1)
    now = datetime(2020, 3, 10)
    while True:
        train_X.append(encoding_string(date.strftime("%Y-%m-%d") + ', ' + str(date.weekday()), string_index))
        train_y.append(encoding_string(date.strftime("%A, %d %B %Y").lower(), string_index))
        if date == now:
            break
        date = date + td
    return train_X, train_y

a, b = make_train_data()
for bb in b:
    if len(bb) != TIME_STEPS:
        print(len(bb), decoding_string(bb, string_index))

model = attention_model(len(string_index))
if sys.argv[1] == 'test':
    model.load_weights('attention_test.h5')
if sys.argv[1] == 'test':
    index = np.random.randint(len(a))
    test_data = encoding_string('1854-05-08, 0', string_index)
    test = model.predict_on_batch([[test_data]])
    print('input :', decoding_string(test_data, string_index))
    print('prdict :', decoding_string(test[0][0], string_index))
    print('answer :', decoding_string(b[index], string_index))
    print(test[1][0].numpy().tolist())
elif sys.argv[1] == 'train':
    for i in range(300):
        model.fit(np.array(a), np.array(b), batch_size=32, epochs=1, verbose=1, validation_split=0.2)
        model.save_weights('attention_test.h5')

        model.load_weights('attention_test.h5')
        # print(model.predict_on_batch([[a[-1]]]).shape)
        index = np.random.randint(len(a))
        print('epoch :', i, 'input :', decoding_string(a[index], string_index))
        print('prdict :', decoding_string(model.predict_on_batch([[a[index]]])[0], string_index))
        print('answer :', decoding_string(b[index], string_index))


