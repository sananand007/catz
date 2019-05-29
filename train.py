import wandb 
from wandb.keras import WandbCallback 

wandb.init(project="nightking")

from keras.layers import Conv2D, UpSampling2D, MaxPooling2D
from keras.models import Sequential
from keras.callbacks import Callback
from keras.layers import Dropout
from keras.layers import Dense
from keras.optimizers import SGD
import random
import glob
import wandb
from wandb.keras import WandbCallback
import subprocess
import os
from PIL import Image
import numpy as np
from keras import backend as K
from keras.layers import GRU, LSTM, ConvLSTM2DCell

# Added for CNN model
#from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten, Dense, Reshape
import tensorflow as tf
from tensorflow.python.client import device_lib
K.tensorflow_backend._get_available_gpus()

configuration = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} )
sess = tf.Session(config=configuration) 
K.set_session(sess)

# For using a DC GAN [Reference Datacamp - https://www.datacamp.com/community/tutorials/generative-adversarial-networks]

# Using GPU - https://www.quora.com/How-can-I-run-Keras-on-GPU

run = wandb.init(project='catz')
config = run.config

config.num_epochs = 20
config.batch_size = 40
config.img_dir = "images"
config.height = 96
config.width = 96

val_dir = 'catz/test'
train_dir = 'catz/train'

# automatically get the data if it doesn't exist
if not os.path.exists("catz"):
    print("Downloading catz dataset...")
    subprocess.check_output(
        "curl https://storage.googleapis.com/wandb/catz.tar.gz | tar xz", shell=True)


class ImageCallback(Callback):
    def on_epoch_end(self, epoch, logs):
        validation_X, validation_y = next(
            my_generator(15, val_dir))
        output = self.model.predict(validation_X)
        wandb.log({
            "input": [wandb.Image(np.concatenate(np.split(c, 5, axis=2), axis=1)) for c in validation_X],
            "output": [wandb.Image(np.concatenate([validation_y[i], o], axis=1)) for i, o in enumerate(output)]
        }, commit=False)


def my_generator(batch_size, img_dir):
    """A generator that returns 5 images plus a result image"""
    cat_dirs = glob.glob(img_dir + "/*")
    counter = 0
    while True:
        input_images = np.zeros(
            (batch_size, config.width, config.height, 3 * 5))
        output_images = np.zeros((batch_size, config.width, config.height, 3))
        random.shuffle(cat_dirs)
        if ((counter+1)*batch_size >= len(cat_dirs)):
            counter = 0
        for i in range(batch_size):
            input_imgs = glob.glob(cat_dirs[counter + i] + "/cat_[0-5]*")
            #print(input_imgs)
            imgs = [Image.open(img) for img in sorted(input_imgs)]
            input_images[i] = np.concatenate(imgs, axis=2)
            output_images[i] = np.array(Image.open(
                cat_dirs[counter + i] + "/cat_result.jpg"))
        # print("Input Image \n")
        # print(input_images.shape)
        yield (input_images, output_images)
        counter += batch_size

"""
Base Model - val_perceptual_distance = 28.6 [Best score][No Image aug]

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same',
                 input_shape=(config.height, config.width, 5 * 3)))
model.add(MaxPooling2D(2, 2))
#model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))
"""


"""
Base Model v2 - val_perceptual_distance =  28.04 [Best score][No Image aug]

model = Sequential()
model.add(Conv2D(128, (3, 3), activation='relu', padding='same',
                 input_shape=(config.height, config.width, 5 * 3)))
model.add(MaxPooling2D(2, 2))
#model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))
"""

"""
Base Model v3[Trying GANs] - val_perceptual_distance =  28.04 [Best score][No Image aug]

model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', padding='same',
                 input_shape=(config.height, config.width, 5 * 3)))
#model.add(MaxPooling2D(2, 2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
#model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))
"""

""" 
Base Model v3[Trying Conv+LSTM] - val_perceptual_distance =  28.04 [Best score][No Image aug]
"""
model = Sequential()
model.add(ConvLSTM2D(filters=32, kernel_size=(3,3), padding='same', input_shape=(5, config.height, config.width, 3), return_sequences=True, stateful=False))
model.add(BatchNormalization())
model.add(ConvLSTM2D(filters=32, kernel_size=(3,3), padding='same', return_sequences=True))
model.add(BatchNormalization())
model.add(ConvLSTM2D(filters=32, kernel_size=(3,3), padding='same', return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(ConvLSTM2D(filters=32, kernel_size=(3,3), padding='same', return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.3))
LSTM_to_conv_dims = (96, 96, 3)
model.add(Reshape(LSTM_to_conv_dims))
model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))
print(f'Shape of model {model.summary()}')

#Modified Model
"""
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same',
                 input_shape=(config.height, config.width, 5 * 3)))

model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.3))

model.add(UpSampling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.3))

model.add(UpSampling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.3))

model.add(UpSampling2D((2, 2)))
model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))
"""


def perceptual_distance(y_true, y_pred):
    rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
    r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]

    return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))


sgd = SGD(lr=0.001, decay=0.03, momentum=0.9, nesterov=True)
model.compile(optimizer='adam', loss='mse', metrics=[perceptual_distance])

model.fit_generator(my_generator(config.batch_size, train_dir),
                    steps_per_epoch=len(
                        glob.glob(train_dir + "/*")) // config.batch_size,
                    epochs=config.num_epochs, callbacks=[
    ImageCallback(), WandbCallback()],
    validation_steps=len(glob.glob(val_dir + "/*")) // config.batch_size,
    validation_data=my_generator(config.batch_size, val_dir))

#model.fit(X_train, y_train, validationData=(X_test, y_test) , epochs=config.epochs, callbacks=[WandbCallback()])

model.save(os.path.join(wandb.run.dir, "model_tcool.h5"))
