from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation
from keras import optimizers
import matplotlib.image as img
import numpy as np
import cv2
#from PIL import Image
import os



path = "../../Desktop/Data_set_cnn/"
# all image would have a input size of 4608, 4608, 3
M = 4608
N = 4608

training_input = []
training_output = []

for i in os.listdir(path + "Doors"):
    #print(i)
    if(not i.endswith('db')):
        image = img.imread(path + "Doors/" + i)
        shape = image.shape
        image = cv2.resize(image, dsize = (M, N))
        training_input.append(image)
        training_output.append([1, 0, 0])
        '''
        if(M > shape[0]):
            M = shape[0]
        if(N > shape[1]):
            N = shape[1]
        '''


#print("Max M for Doors is", M)
#print("Max N for Doors is", N)

for i in os.listdir(path + "Sign"):
    if(not i.endswith('db')):
        image = img.imread(path + "Sign/" + i)
        shape = image.shape
        image = cv2.resize(image, dsize = (M, N))
        training_input.append(image)
        training_output.append([0, 1, 0])
        '''
        if(M < shape[0]):
            M = shape[0]
        if(N < shape[1]):
            N = shape[1]
        '''

#print("Max M for Doors + Sign is", M)
#print("Max N for Doors + Sign is", N)

for i in os.listdir(path + "Stairs"):
    if(not i.endswith('db')):
        image = img.imread(path + "Stairs/" + i)
        shape = image.shape
        image = cv2.resize(image, dsize = (M, N))
        training_input.append(image)
        training_output.append([0, 0, 1])
        '''
        if(M < shape[0]):
            M = shape[0]
        if(N < shape[1]):
            N = shape[1]
        '''

#print("Max M for all is", M)
#print("Max N for all is", N)

print(training_output)






model = Sequential()

filters_1 = 32
filters_2 = 32
filters_3 = 64

kernal_size = (3, 3)

input_shape = (150, 150, 3)
pool_size = (2, 2)


model.add(Conv2D(filters_1, kernal_size, input_shape = input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = pool_size))

model.add(Conv2D(filters_2, kernal_size))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = pool_size))

model.add(Conv2D(filters_3, kernal_size))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = pool_size))

model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))
