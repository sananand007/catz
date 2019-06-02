import wandb 
from wandb.keras import WandbCallback 

wandb.init(project="nightking")

from keras.layers import Conv2D, UpSampling2D, MaxPooling2D, AveragePooling3D, Conv3D, MaxPooling3D
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

config.num_epochs = 10
config.batch_size = 64
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
        # wandb.log({
        #     "input": [wandb.Image(np.concatenate(np.split(c, 5, axis=2), axis=1)) for c in validation_X],
        #     "output": [wandb.Image(np.concatenate([validation_y[i], o], axis=1)) for i, o in enumerate(output)]
        # }, commit=False)



def my_generator(batch_size, img_dir):
    """A generator that returns 5 images plus a result image"""
    cat_dirs = glob.glob(img_dir + "/*")
    counter = 0
    while True:
        input_images = np.zeros(
            (batch_size, 5, config.width, config.height, 3))
        output_images = np.zeros((batch_size, 1, config.width, config.height, 3))
        random.shuffle(cat_dirs)
        if ((counter+1)*batch_size >= len(cat_dirs)):
            counter = 0
        for i in range(batch_size):
            input_imgs = glob.glob(cat_dirs[counter + i] + "/cat_[0-5]*")
            imgs = [Image.open(img) for img in sorted(input_imgs)]
            for j, img in enumerate(imgs):
                input_images[i][j] = img
#             input_images[i] = np.concatenate(imgs, axis=2)
            output_images[i] = np.array(Image.open(
                cat_dirs[counter + i] + "/cat_result.jpg"))
        print(f'the input size {input_images.shape}, the output size {output_images.shape}')
        yield (input_images, output_images)
        counter += batch_size



model = Sequential()
model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=(5, config.height, config.width, 3), padding='same'))
# output_size = (1, config.height, config.width, 3)
# model.add(Reshape(output_size))
print(f'Shape of model {model.summary()}')



def perceptual_distance(y_true, y_pred):
    rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
    r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]

    return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))


# sgd = SGD(lr=0.001, decay=0.03, momentum=0.9, nesterov=True)
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
