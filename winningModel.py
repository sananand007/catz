from keras.layers import Conv2D, UpSampling2D, MaxPooling2D
from keras.models import Sequential
from keras.callbacks import Callback
import random
import glob
import wandb
from wandb.keras import WandbCallback
import subprocess
import os
from PIL import Image
import numpy as np
from keras import backend as K

run = wandb.init(project='catz')
config = run.config

config.num_epochs = 2
config.batch_size = 32
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
            imgs = [Image.open(img) for img in sorted(input_imgs)]
            input_images[i] = np.concatenate(imgs, axis=2)
            output_images[i] = np.array(Image.open(
                cat_dirs[counter + i] + "/cat_result.jpg"))
        yield (input_images, output_images)
        counter += batch_size


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same',
                 input_shape=(config.height, config.width, 5 * 3)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))


def perceptual_distance(y_true, y_pred):
    rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
    r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]

    return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))


model.compile(optimizer='adam', loss='mse', metrics=[perceptual_distance])

model.fit_generator(my_generator(config.batch_size, train_dir),
                    steps_per_epoch=len(
                        glob.glob(train_dir + "/*")) // config.batch_size,
                    epochs=config.num_epochs, callbacks=[
    ImageCallback(), WandbCallback()],
    validation_steps=len(glob.glob(val_dir + "/*")) // config.batch_size,
    validation_data=my_generator(config.batch_size, val_dir))



from keras.callbacks import *

class CyclicLR(Callback):    
    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())


# Define our custom metric
def PSNRLoss(y_true, y_pred):
    MAXp = 255.0
    #20 * np.log10(MAXp) - (10. * np.log10(K.mean(K.square(y_pred - y_true))))
    #return 48.13 - (4.3422 * K.log(K.mean(K.square(y_pred - y_true))))
    #return 10.0 * K.log(1.0 / (K.mean(K.square(y_pred - y_true)))) / K.log(10.0)
    return -10.0  * K.log((MAXp ** 2) / (K.mean(K.square(y_pred - y_true)))) / K.log(10.0)


import tensorflow as tf
from keras.losses import mean_absolute_error

def SSIMLoss(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=255.0))
    return 0.84 * (1.0 - (ssim / 2.0)) + 0.16 * mae


# Data Augmentation Stratergies

from skimage.measure import compare_ssim as ssim

def get_verydiff_samples(train_path):
    difflist = []
    #print(os.listdir(os.getcwd()))
    currentpath = os.getcwd()
    os.chdir(train_path)
    for dirs in os.listdir(os.getcwd()):
        file1 = os.path.join(os.path.join(os.getcwd(), dirs), 'cat_4.jpg')
        file2 = os.path.join(os.path.join(os.getcwd(), dirs), 'cat_result.jpg')
        image1 = img_to_array(load_img(file1))
        image2 = img_to_array(load_img(file2))
        similarity = ssim(image1, image2, multichannel=True)
        if similarity < 0.85:
            difflist.append(dirs)
    os.chdir(currentpath)
    return difflist


def perform_noisy_data_augmentation_ssim(img_dir):
    #print(os.getcwd())
    row , col, ch = 96, 96, 3
    mean = 0
    var = 0.1
    sigma = var**0.5
    train_path = os.path.join(os.getcwd(), img_dir)
    difflist = get_verydiff_samples(train_path)
    cat_dirs = [os.path.join(train_path, item) for item in difflist]
    batch_size = 5
    for cat_dir in cat_dirs:
        input_imgs = glob.glob(cat_dir + "/cat_[0-5]*")
        output_imgs = glob.glob(cat_dir + "/cat_r*")
        imgs = sorted(input_imgs)
        #imgs.extend(output_imgs)
        for batch_num in range(batch_size):
            dir_name = os.path.dirname(input_imgs[0])
            dir_name = dir_name + '_' + str(batch_num)
            os.mkdir(dir_name)
            #print(dir_name)
            for index, img in enumerate(imgs):
                #print(img)
                x = img_to_array(load_img(img))
                #print(x.shape)
                img_name = os.path.basename(img).split('.')[0]
                #print(img_name)
                noise = np.random.normal(mean,sigma,(row,col,ch))
                noise = noise.reshape(row,col,ch)
                x = x + noise
                #print(new_image)
                new_img = array_to_img(x)
                #print("Saving image")
                new_img.save(os.path.join(dir_name, (img_name +".jpg")))

            for img in os.listdir(dir_name):
                file_name = os.path.basename(img)
                #print(file_name)
                if 'result' not in file_name:
                    name = file_name.split('_')[:2]
                    new_name = "_".join(s for s in name)
                    new_name = new_name + ".jpg"
                    #print(new_name)
                else:
                    new_name = 'cat_result.jpg'
                os.rename(os.path.join(dir_name, file_name),
                              os.path.join(dir_name, new_name))
            shutil.copy(output_imgs[0], os.path.join(dir_name, 
                                                     'cat_result.jpg'))

import shutil

def perform_data_augmentation(img_dir):
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        fill_mode='nearest')

    #print(os.getcwd())
    cat_dirs = glob.glob(os.getcwd() + img_dir + '/*')
    #print(cat_dirs)
    batch_size = 2
    for cat_dir in cat_dirs:
        input_imgs = glob.glob(cat_dir + "/cat_[0-5]*")
        output_imgs = glob.glob(cat_dir + "/cat_r*")
        imgs = sorted(input_imgs)
        #imgs.extend(output_imgs)
        for batch_num in range(batch_size):
            dir_name = os.path.dirname(input_imgs[0])
            dir_name = dir_name + '_' + str(batch_num)
            os.mkdir(dir_name)
            #print(dir_name)
            for index, img in enumerate(imgs):
                #print(img)
                x = img_to_array(load_img(img))
                x = x.reshape((1,) + x.shape)
                img_name = os.path.basename(img).split('.')[0]
                #print(img_name)
                for output in datagen.flow(x, batch_size=1, 
                                           save_to_dir=dir_name, 
                                           save_prefix=img_name, 
                                           save_format='jpg'):
                    break
            for img in os.listdir(dir_name):
                file_name = os.path.basename(img)
                #print(file_name)
                if 'result' not in file_name:
                    name = file_name.split('_')[:2]
                    new_name = "_".join(s for s in name)
                    new_name = new_name + ".jpg"
                    #print(new_name)
                else:
                    new_name = 'cat_result.jpg'
                os.rename(os.path.join(dir_name, file_name),
                              os.path.join(dir_name, new_name))
            shutil.copy(output_imgs[0], os.path.join(dir_name, 
                                                     'cat_result.jpg'))


def perform_noisy_data_augmentation(img_dir):
    #print(os.getcwd())
    row , col, ch = 96, 96, 3
    mean = 0
    var = 0.01
    sigma = var**0.5
    cat_dirs = glob.glob(os.getcwd() + img_dir + '/*')
    #print(cat_dirs)
    batch_size = 3
    for cat_dir in cat_dirs:
        input_imgs = glob.glob(cat_dir + "/cat_[0-5]*")
        output_imgs = glob.glob(cat_dir + "/cat_r*")
        imgs = sorted(input_imgs)
        #imgs.extend(output_imgs)
        for batch_num in range(batch_size):
            dir_name = os.path.dirname(input_imgs[0])
            dir_name = dir_name + '_' + str(batch_num)
            os.mkdir(dir_name)
            #print(dir_name)
            for index, img in enumerate(imgs):
                #print(img)
                x = img_to_array(load_img(img))
                #print(x.shape)
                img_name = os.path.basename(img).split('.')[0]
                #print(img_name)
                noise = np.random.normal(mean,sigma,(row,col,ch))
                noise = noise.reshape(row,col,ch)
                x = x + noise
                #print(new_image)
                new_img = array_to_img(x)
                #print("Saving image")
                new_img.save(os.path.join(dir_name, (img_name +".jpg")))

            for img in os.listdir(dir_name):
                file_name = os.path.basename(img)
                #print(file_name)
                if 'result' not in file_name:
                    name = file_name.split('_')[:2]
                    new_name = "_".join(s for s in name)
                    new_name = new_name + ".jpg"
                    #print(new_name)
                else:
                    new_name = 'cat_result.jpg'
                os.rename(os.path.join(dir_name, file_name),
                              os.path.join(dir_name, new_name))
            shutil.copy(output_imgs[0], os.path.join(dir_name, 
                                                     'cat_result.jpg'))


train_dirs = r'catz/train'
perform_noisy_data_augmentation_ssim(train_dirs)

# Modified to include pairs of data points
def my_modifiedgenerator(batch_size, img_dir):
    """A generator that returns 5 images plus a result image"""
    cat_dirs = glob.glob(img_dir + "/*")
    concat_img = None
    counter = 0
    while True:
        input_images = np.zeros(
            (batch_size, 5, config.width, config.height, 3))
        combined_images = np.zeros(
            (batch_size, 4, config.width, config.height, 3*2))
        combined_frames = np.zeros((4, config.width, config.height, 3*2))
        all_frames = np.zeros((batch_size, config.width, config.height, 3))
        #all_frames = np.zeros((batch_size, config.width, config.height, 3))
        output_images = np.zeros((batch_size, config.width, config.height, 3))
        random.shuffle(cat_dirs)
        if ((counter+1)*batch_size >= len(cat_dirs)):
            counter = 0
        for i in range(batch_size):
            input_imgs = glob.glob(cat_dirs[counter + i] + "/cat_[0-5]*")
            imgs = [np.array(Image.open(img)) for img in sorted(input_imgs)]
            first = True
            concat_img = imgs[0]
            for img in imgs[1:]:
                if(first):
                    concat_img = np.concatenate((concat_img[None, ...], 
                                           img[None, ...]), axis=0)
                    first = False
                else:
                    concat_img = np.concatenate((concat_img, img[None, ...]), 
                                          axis=0)

            for index in range(len(imgs)-1):
                combined_frames[index] = np.concatenate((imgs[index], 
                                                  imgs[index+1]), axis=2)
            #combined_frames[4] = np.concatenate((imgs[0], 
            #                                      imgs[4]), axis=2)
            input_images[i] = concat_img
            combined_images[i] = combined_frames
            output_images[i] = np.array(Image.open(
                cat_dirs[counter + i] + "/cat_result.jpg"))
            #all_frames[i] = np.concatenate(imgs, axis=2)
            #all_frames[i] = np.mean(imgs, axis=0)
            all_frames[i] = imgs[-1]
            #all_frames[i] = np.concatenate([imgs[0], imgs[4]], axis=2)
        #input_images = input_images/255.0
        #output_images = output_images/255.0
        #combined_images = combined_images/255.0
        yield ([input_images], output_images)
        counter += batch_size

class modifiedImageCallback(Callback):
    def on_epoch_end(self, epoch, logs):
        validation_X, validation_y = next(
            my_modifiedgenerator(15, val_dir))
        output = self.model.predict(validation_X)
        wandb.log({
            "input": [wandb.Image(np.concatenate(np.split(c, 5, 
                                                          axis=0), 
                                       axis=1)) for c in validation_X[0]],
            "output": [wandb.Image(np.concatenate([validation_y[i], o], 
                                       axis=1)) for i, o in enumerate(output)]
        }, commit=False)
        #K.set_value(optimizer.lr, 0.5 * K.get_value(optimizer.lr))


import math
import matplotlib.pyplot as plt

class LRCallback(Callback):
    def __init__(self, model):
        self.model = model
        self.losses = []
        self.lrs = []
        self.betas = []
        self.best_loss = 1e9
        self.start_lr=0.00001
        self.end_lr=100.0
        self.start_beta=0.99
        self.end_beta=0.01
        self.original_lr = K.get_value(self.model.optimizer.lr)
        self.beta_1 = K.get_value(self.model.optimizer.beta_1)
        self.num_batches =  3340.0       #epochs * x_train.shape[0] / 40
        self.lr_mult = ((self.end_lr / self.start_lr) ** (1 / self.num_batches))
        self.beta_mult = ((self.end_beta / self.start_beta) ** (1 / self.num_batches))
        self.batch_count = 0

    def on_batch_end(self, batch, logs):
        self.batch_count = self.batch_count + 1
        # Log the learning rate
        lr = K.get_value(self.model.optimizer.lr)
        beta = K.get_value(self.model.optimizer.beta_1)
        self.lrs.append(lr)
        self.betas.append(beta)
        # Log the loss
        loss = logs['loss']
        self.losses.append(loss)

        # Check whether the loss got too large or NaN
        #if math.isnan(loss) or loss > self.best_loss * 4:
            #self.model.stop_training = True
        #    return

        if loss < self.best_loss:
            self.best_loss = loss

        if self.batch_count == self.num_batches-1:
            self.plot_loss_lr()
        # Increase the learning rate for the next batch
        lr *= self.lr_mult
        beta /= self.beta_mult
        #lr += self.lr_mult
        #print("new lr : ", lr)
        K.set_value(self.model.optimizer.lr, lr)
        K.set_value(self.model.optimizer.beta_1, beta)
    
    def on_epoch_end(self, epoch, logs):
        print("Lr : ", self.model.optimizer.lr)
        print("Lr : ", self.model.optimizer.beta_1)

    def plot_loss_lr(self, n_skip_beginning=10, n_skip_end=5):
        """
        Plots the loss.
        Parameters:
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
        """
        plt.ylabel("loss")
        plt.xlabel("learning rate (log scale)")
        plt.plot(self.lrs[n_skip_beginning:-n_skip_end], 
                 self.losses[n_skip_beginning:-n_skip_end])
        plt.xscale('log')

    def plot_loss_momentum(self, n_skip_beginning=10, n_skip_end=5):
        plt.ylabel("loss")
        plt.xlabel("Momentum (log scale)")
        plt.plot(self.betas[n_skip_beginning:-n_skip_end], 
                 self.losses[n_skip_beginning:-n_skip_end])
        plt.xscale('log')

class EarlyStoppingByDistance(Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True
earlystop = EarlyStoppingByDistance(monitor='val_perceptual_distance', 
                                    value=10.0, verbose=1)

# LR range test

lrfinder = LRCallback(model)
K.set_value(optimizer.lr, lrfinder.start_lr)
K.set_value(optimizer.beta_1, lrfinder.start_beta)

history = model.fit_generator(my_modifiedgenerator(config.batch_size, 
                                                   train_dir), 
      steps_per_epoch=len(glob.glob(train_dir + "/*")) // config.batch_size,
                              epochs=20, 
      callbacks=[lrfinder, modifiedImageCallback(), WandbCallback()],
validation_steps=len(glob.glob(val_dir + "/*")) // config.batch_size,
validation_data=my_modifiedgenerator(config.batch_size, val_dir))


lrfinder.plot_loss_lr()

fig = plt.figure()
plt.ylabel("loss")
plt.xlabel("learning rate (log scale)")
plt.plot(lrfinder.lrs[1539:1545], 
         lrfinder.losses[1539:1545])
plt.xscale('log')
plt.show()




# New Architecture

lstm_filters = 5
input_img = Input(shape=(5, config.width, config.height, 3))

combined_frames = Input(shape=(config.width, config.height, 3*5))
#input1 = Lambda(lambda x: x[:,:4,:,:,:])(input_img)

x = ConvLSTM2D(filters=lstm_filters, kernel_size=(3, 3), padding='same', 
               return_sequences=True)(input_img)

x = ConvLSTM2D(filters=lstm_filters, kernel_size=(3, 3), padding='same', 
               return_sequences=True)(x)

c1 = ConvLSTM2D(filters=lstm_filters, kernel_size=(3, 3), padding='same', 
               return_sequences=True)(x)


x = TimeDistributed(MaxPooling2D((2,2), (2,2)))(c1)



x = ConvLSTM2D(filters=2*lstm_filters, kernel_size=(3, 3), padding='same', 
               return_sequences=True)(x)

x = ConvLSTM2D(filters=2*lstm_filters, kernel_size=(3, 3), padding='same', 
               return_sequences=True)(x)

c2 = ConvLSTM2D(filters=2*lstm_filters, kernel_size=(3, 3), padding='same', 
               return_sequences=True)(x)


x = TimeDistributed(MaxPooling2D((2,2), (2,2)))(c2)




x = ConvLSTM2D(filters=2*lstm_filters, kernel_size=(3, 3), padding='same', 
               return_sequences=True)(x)

x = ConvLSTM2D(filters=2*lstm_filters, kernel_size=(3, 3), padding='same', 
               return_sequences=True)(x)

c3 = ConvLSTM2D(filters=2*lstm_filters, kernel_size=(3, 3), padding='same', 
               return_sequences=True)(x)


x = TimeDistributed(UpSampling2D((2,2)))(c3)
x = concatenate([c2, x])
x = TimeDistributed(Conv2D(lstm_filters, (3, 3), padding='same'))(x)


x = TimeDistributed(UpSampling2D((2,2)))(x)
x = concatenate([c1, x])


x = TimeDistributed(Conv2D(3, (3, 3), padding='same', activation='relu'))(x)
x = TimeDistributed(UpSampling2D((2,2)))(x)

x = TimeDistributed(Conv2D(3, (3, 3), padding='same', activation='relu'))(x)
x = TimeDistributed(MaxPooling2D((2,2)))(x)

output = Conv3D(3, (3, 3, 3), padding='same', activation='relu')(x)
output = Lambda(lambda x: x[:, 4:, :, :, :])(output)
output = Reshape((96,96,3))(output)

final = Lambda(lambda x: x[:, 4:, :, :, :])(input_img)

final = Reshape((96,96,3))(final)


#final = Add()([output, final])
final = concatenate([output, final])

result = Conv2D(3, (3, 3), padding='same', activation='relu')(final)

model = Model(inputs=[combined_frames, input_img], outputs=result)
#model.compile(optimizer=optimizer, loss=[SSIMLoss], 
#              metrics=[perceptual_distance])
model.summary()


optimizer = Adam(lr=0.0017, beta_1=0.99)
#optimizer = Adam(lr=0.0017)
#optimizer = tf.contrib.opt.AdamWOptimizer(learning_rate=0.0017, 
#                                          weight_decay=0.3)
#from keras.optimizers import SGD

#sgd = SGD(lr=0.0017, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=optimizer, loss='mse', 
              metrics=[perceptual_distance])

# later convergence at 0.000004 and 0.99

#K.set_value(optimizer.lr, 0.000001)
#K.set_value(optimizer.beta_1, 0.99)
filepath="weights.best3.hdf5"
#model.load_weights(filepath)
checkpoint = ModelCheckpoint(filepath, monitor='val_perceptual_distance', 
                             verbose=0, 
                             save_best_only=True, mode='min')
# new_lr = lr*factor
# this factor is kept same as we would have in learning rate range
# max_lr = 0.00173, min_lr = 0.0008
# factor = ((self.end_lr / self.start_lr) ** (1 / self.num_batches))
# (0.00173/0.0008)**(1/5566)

#factor = (0.0008/0.00173)**(1/835)
reducelr = ReduceLROnPlateau(monitor='val_perceptual_distance', factor=0.3, 
                             patience=3, min_lr=0.0000000001)

history = model.fit_generator(my_modifiedgenerator(config.batch_size, 
                                                   train_dir), 
     steps_per_epoch=len(glob.glob(train_dir + "/*")) // config.batch_size,
                              epochs=200, 
     callbacks=[earlystop, reducelr, checkpoint, ImageCallback(), WandbCallback()],
     validation_steps=len(glob.glob(val_dir + "/*")) // config.batch_size,
     validation_data=my_modifiedgenerator(config.batch_size, val_dir))