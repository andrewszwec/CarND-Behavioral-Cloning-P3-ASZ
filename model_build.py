from __future__ import print_function
import cv2
import csv
import numpy as np
from PIL import Image
import pickle

import keras

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Recommended Flipping
from keras.layers import Lambda
from keras.layers import Cropping2D

import os
import csv

import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def prepImage(path, img_dir):
    filename = path.split('/')[-1]
    current_path = img_dir + filename
    return cv2.imread(current_path, 1) # 0 = grayscale, 1 = Colour


# driving_log = './data/driving_log.csv'
# img_dir = './data/IMG/'
driving_log = './car_training_data/driving_log.csv'
img_dir = './car_training_data/IMG/'

samples = []
with open(driving_log) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


train_samples, validation_samples = train_test_split(samples, test_size=0.2)





# Hyperparameters - leave these in this position
batch_size = 256
epochs = 15
learn_rate = 0.001
drop_rate = 0.3

## Add multi camera angels (left/right) + Augmentation# Flip the Images for more data   

def generator(samples, batch_size=32):
    num_samples = len(samples)
    batch_size = batch_size // 2 # to deal with image augmentation
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                
                # IMAGES
                img_center = prepImage(batch_sample[0], img_dir)
                images.append(img_center)
                
                img_left = prepImage(batch_sample[1], img_dir)
                images.append(img_left)
                
                img_right = prepImage(batch_sample[2], img_dir)
                images.append(img_right)
                
                
                # STEERING
                steering_center = float(batch_sample[3])
                angles.append(steering_center)

                correction = 0.2 # this is a parameter to tune
                steering_left = steering_center + correction
                angles.append(steering_left)
                steering_right = steering_center - correction
                angles.append(steering_right)
        
            # AUGMENTATION (Flip the Images for more data)
            augmented_images = []
            augmented_angles = [] 
            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image, 1))
                augmented_angles.append(angle*-1.0)  

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            
            yield shuffle(X_train, y_train)
            
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)



callbacks = [
    EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=5),
    ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True, verbose=0)
]

img_h, img_w, ch = 160, 320, 3

model = Sequential()

# # Normalise
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(img_h,img_w,ch)))
# Crop
model.add(Cropping2D(cropping=((50,20), (0,0)) )) # , input_shape=(3,160,320)))

# Conv Layer 1
model.add(Conv2D(8, (3, 3), activation='relu', padding='same')) # filters = 8
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(drop_rate))

# Conv Layer 2
model.add(Conv2D(16, (3, 3), activation='relu', padding='same')) # filters = 16
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(drop_rate))

# Conv Layer 3
model.add(Conv2D(32, (3, 3), activation='relu', padding='same')) # filters = 32
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(drop_rate))

# Conv Layer 4
model.add(Conv2D(64, (3, 3), activation='relu', padding='same')) # filters = 64
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(drop_rate))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dropout(drop_rate))
model.add(Dense(1))

model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learn_rate))

history_object = model.fit_generator(train_generator,
                    # num train samples * 3 for centre, left, right camera, *2 for augmentation
                    steps_per_epoch= (len(train_samples)*3*2) / batch_size, 
                    validation_data=validation_generator, 
                    validation_steps= (len(validation_samples)*3*2) / batch_size, 
                    epochs=epochs,
                    callbacks=callbacks)


pickle.dump( history_object, open( "history_object.p", "wb" ) )

print('End Success')