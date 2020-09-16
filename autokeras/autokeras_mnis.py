import numpy as np
import tensorflow as tf
import json
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow import keras
import autokeras as ak
from trains import Task
task = None
task = Task.init(project_name='test_project', task_name='mnist autokeras test', output_uri="http://10.102.20.220:8081/")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
task_number = 0
# print('[log log]', task.get_logger().get_default_upload_destination())
# print('[log log]', task.get_output_log_web_page())
# Initialize the image classifier
clf = ak.ImageClassifier(
        project_name='image_classifier',
#        tuner='bayesian',
        overwrite=True,
        max_trials=2)
class custom_callback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        global task_number
        global task
        print(task_number, 'train start')
        #tmp_model = clf.export_model()
        #Task.set_artifacts(tmp_model.to_json())
        #print('111111111111111111')
        #print('callbackback', clf.tuner.results_summary())#clf.tuner.oracle.get_space())
        #print('111111111111111111')
        #print(clf.tuner.hypermodel.build(clf.tuner.oracle.get_space()).to_json())
        #task.connect(json.loads(clf.tuner.hypermodel.build(clf.tuner.oracle.get_space()).to_json()), 'mnist autokeras' + str(task_number))
        task.set_name('mnist autokeras' + str(task_number))
        #task.set_parameters_as_dict(json.loads(clf.tuner.hypermodel.build(clf.tuner.oracle.get_space()).to_json()))
        #task.upload_artifact('mnist autokeras artifact' + str(task_number), json.loads(clf.tuner.hypermodel.build(clf.tuner.oracle.get_space()).to_json()))
        task_number += 1
    def on_train_end(self, logs=None):
        # task.set_model_label_enumeration(clf.tuner.oracle.get_space().get_config())
        global task
        task.connect_configuration(json.loads(clf.tuner.hypermodel.build(clf.tuner.oracle.get_space()).to_json()))
        # task.flush()
        # task.close()

tensorboard_callback_train = keras.callbacks.TensorBoard(log_dir='log', write_graph=True, histogram_freq=0)
cc = custom_callback()
# Feed the image classifier with training data.
clf.fit(x_train, y_train, epochs=10, validation_split=0.15, callbacks=[tensorboard_callback_train, cc], verbose=0)

# Predict with the best model.
predicted_y = clf.predict(x_test)
print(predicted_y)

# Evalute the best model with testing data.
print(clf.evaluate(x_test, y_test))

model = clf.export_model()

try:
    model.save("model_autokeras", save_format='tf')
except:
    model.save('model_sutokeras.h5')

from tensorflow.keras.models import load_model
loaded_model = load_model('model_autokeras', custom_objects=ak.CUSTOM_OBJECTS)


