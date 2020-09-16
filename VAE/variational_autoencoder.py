# import 
import tensorflow.keras
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K

# def make_image(tensor):
#     """
#     Convert an numpy representation image to Image protobuf.
#     Copied from https://github.com/lanpa/tensorboard-pytorch/
#     """
#     from PIL import Image
#     height, width, channel = tensor.shape
#     image = Image.fromarray(tensor)
#     import io
#     output = io.BytesIO()
#     image.save(output, format='PNG')
#     image_string = output.getvalue()
#     output.close()
#     return tf.Summary.Image(height=height,
#                          width=width,
#                          colorspace=channel,
#                          encoded_image_string=image_string)

# class TensorBoardImage(keras.callbacks.Callback):
#     def __init__(self, tag):
#         super().__init__() 
#         self.tag = tag

#     def on_epoch_end(self, epoch, logs={}):
#         # Load image
#         img = data.astronaut()
#         # Do something to the image
#         img = (255 * skimage.util.random_noise(img)).astype('uint8')

#         image = make_image(img)
#         summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag, image=image)])
#         writer = tf.summary.FileWriter('./logs')
#         writer.add_summary(summary, epoch)
#         writer.close()

#         return

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

class valiationalAutoencoder:
    def __init__(self, input_dim, output_dim, batch_size, epochs):
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.epochs = epochs

        inputs = Input(shape=(self.input_dim,), name='encoder_input')
        x = Dense(1024, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        z_mean = Dense(128, name='z_mean')(x)
        z_log_var = Dense(128, name='z_log_var')(x)

        z = Lambda(sampling, output_shape=(128,), name='z')([z_mean, z_log_var])

        self.encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        # print(self.encoder.summary())

        latent_inputs = Input(shape=(128,), name='z_sampling')
        x = Dense(256, activation='relu')(latent_inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(1024, activation='relu')(x)
        outputs = Dense(self.input_dim, activation='sigmoid')(x)

        self.decoder = Model(latent_inputs, outputs, name='decoder')
        # print(self.decoder.summary())

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
        # print(self.vae.summary())

    def train(self, X_train, model):
        tb_hist = tensorflow.keras.callbacks.TensorBoard(log_dir='./graph_' + model, histogram_freq=1, write_graph=True, write_images=True)
        model_path = './' + model + '.h5'#'{epoch:02d}-{val_loss:.4f}.h5'
        cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True, save_format='tf')
        history = self.vae.fit(X_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1, shuffle=True, validation_split=0.2, callbacks=[tb_hist, cb_checkpoint])
        return history

if __name__ == '__main__':
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)
    from tensorflow.keras.datasets.mnist import load_data
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image    
    (x_train, y_train), (x_test, y_test) = load_data()
    vae = valiationalAutoencoder(784, 784, 32, 10)
    # train
    vae.train(x_train.reshape(x_train.shape[0], 784) / 256, 'mnist_test')
    # load model
    # vae.vae.load_weights('mnist_test.h5')
    random_index = np.random.randint(len(x_test))
    target_img = Image.fromarray(x_test[random_index])
    
    # reconstruction test
    test_result = vae.vae.predict([[x_test[random_index].reshape(784) / 255]])[0]
    result_img = Image.fromarray(test_result.reshape(28, 28) * 255)    

    target_img.save('target_img.gif')
    result_img.save('result_img.gif')
