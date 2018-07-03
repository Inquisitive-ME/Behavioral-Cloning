import csv
import cv2
import numpy as np
from matplotlib import pyplot as plt

from keras.models import Model
from keras.layers import Input
from keras.layers import concatenate
from keras.layers import Cropping2D

from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

from keras.layers import Dense, Activation, Flatten, Dropout, Lambda

def getImage(source_path):
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    BGRImage = cv2.imread(current_path)
    return cv2.cvtColor(BGRImage, cv2.COLOR_BGR2RGB)

steering_correction = 0.25

def addRow(line, images, measurements):
    steering_center = float(line[3])
    
    #split between not steering and steering
    if(steering_center ==0):
        #no steering
        pUseImage = 0.75
        pIncludeSide = .8
        pFlipSide = 0.0

    else:
        #steering
        pUseImage = 1.0
        pIncludeSide = 0.6
        pFlipSide = 0.0
        
    #Include only a percentage of non-steering images
    if(np.random.rand()>(1.0-pUseImage)):
        measurements.append(steering_center)    
                
        center_image = getImage(line[0])
        images.append(center_image)
            
        flipped_steer_center = -steering_center
        measurements.append(flipped_steer_center)
            
        flipped_center_im = np.fliplr(center_image)
        images.append(flipped_center_im)
        
    #Percentage of side images to include
    if(np.random.rand()>(1.0-pIncludeSide)):
        left_image = getImage(line[1])
        right_image = getImage(line[2])

        images.append(left_image)
        images.append(right_image)
        
        steering_left = (steering_center + steering_correction)
        steering_right = (steering_center - steering_correction)        

        measurements.append(steering_left)
        measurements.append(steering_right)
            
         #Percentage of time to include flipped side image
        if(np.random.rand()>(1.0-pFlipSide)):
            flipped_left_im = np.fliplr(left_image)
            flipped_right_im = np.fliplr(right_image)

            images.append(flipped_left_im)
            images.append(flipped_right_im)

            #flipped_steer_left = -steering_center + steering_correction
            #flipped_steer_right = -steering_center - steering_correction

            flipped_steer_left = -steering_left
            flipped_steer_right = -steering_right

            measurements.append(flipped_steer_left)
            measurements.append(flipped_steer_right)


#compiling data set without generator
lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

#print(lines[0])
lines=lines[1:]
#print(lines[0])
images = []
measurements = []

for line in lines:
    addRow(line, images, measurements)
        

X_train = np.array(images)
y_train = np.array(measurements)

print("Number of images: ",len(X_train))

input = Input(shape=(160, 320, 3))

preProcess1 = Lambda(lambda x: x/255.0 - 0.5)(input)
#preProcess1 = Lambda(lambda x: x/127.5 -1.)(input)

crop1 = Cropping2D(cropping=((50,25),(0,0)), input_shape = (1,160,320))(preProcess1)

conv1 = Convolution2D(64, (7, 7), strides=2)(crop1)
conv1 = MaxPooling2D(pool_size=(3,3), strides=2)(conv1)

conv2 = Convolution2D(192, (3, 3), activation='linear', strides=1)(conv1)
conv2 = MaxPooling2D(pool_size=(3, 3), strides=2)(conv2)

#Incesption Layer
Inception1Conv1 = Convolution2D(64, (1, 1), padding='same', activation='linear')(conv2)
Inception1Conv2 = Convolution2D(128, (3, 3), padding='same', activation='linear')(conv2)
Inception1Conv3 = Convolution2D(32, (5, 5), padding='same', activation='linear')(conv2)
Inception1MP = MaxPooling2D(pool_size=(3,3), strides=1, padding='same')(conv2)

Inception1 = concatenate([Inception1Conv1, Inception1Conv2, Inception1Conv3, Inception1MP], axis=3)

drop1 = Dropout(0.4)(Inception1)
act1 = Activation('linear')(drop1)

Flat1=Flatten()(act1)
FCL2 = Dense(10, activation='elu')(Flat1)

output=Dense(1)(FCL2)

model = Model(inputs = input, outputs = output)
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle=True, epochs = 5)
#model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/3,
#          validation_data=validation_generator, validation_steps=len(validation_samples)/3, epochs=2)

#model.summary()

model.save('model.h5')
print("SAVED MODEL")




