from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
from keras.datasets import mnist
import cv2
import h5py
#from PIL import Image
import os



path = "../../../Data_set_cnn/"
#path = "../../Desktop/Data_set_cnn/"
# all image would have a input size of 4608, 4608, 3
M = 150
N = 150






training_input = []
training_output = []
'''
print("Start inputing data")

for i in os.listdir(path + "Doors"):
    #print(i)
    if(not i.endswith('db')):
        image = load_img(path + "Doors/" + i, target_size = (150, 150))
        array_image = img_to_array(image)
        #print(array_image.shape)
        training_input.append(array_image)
        training_output.append([1, 0, 0])
        
        if(M > shape[0]):
            M = shape[0]
        if(N > shape[1]):
            N = shape[1]
        
print("Doors finish")

#print("Max M for Doors is", M)
#print("Max N for Doors is", N)

for i in os.listdir(path + "Sign"):
    if(not i.endswith('db')):
        image = load_img(path + "Sign/" + i, target_size = (150, 150))
        array_image = img_to_array(image)
        training_input.append(array_image)
        training_output.append([0, 1, 0])
        
        if(M < shape[0]):
            M = shape[0]
        if(N < shape[1]):
            N = shape[1]
        
print("Sign finish")
#print("Max M for Doors + Sign is", M)
#print("Max N for Doors + Sign is", N)

for i in os.listdir(path + "Stairs"):
    if(not i.endswith('db')):
        image = load_img(path + "Stairs/" + i, target_size = (150, 150))
        array_image = img_to_array(image)
        training_input.append(array_image)
        training_output.append([0, 0, 1])
        
        if(M < shape[0]):
            M = shape[0]
        if(N < shape[1]):
            N = shape[1]
        

#print("Max M for all is", M)
#print("Max N for all is", N)
print("Stairs finish")

print("Data cleaning finish")

training_input = np.array(training_input)
training_output = np.array(training_output)


print(type(training_input))
print(type(training_input[1]))

with h5py.File('cnn_object_input.hdf5', 'w') as f:
    write_in = f.create_dataset('input_data', data = training_input)
    write_out = f.create_dataset('output_data', data = training_output)

with h5py.File('cnn_object_input.hdf5', 'r') as f:
    training_input = f['input_data']
    training_output = f['output_data']

'''



(x_train, y_train), (x_test, y_test) = mnist.load_data()


real_y_train = np.empty(shape = [0, 10])
real_y_test = np.empty(shape = [0, 10])


# build up the training set output
for i in y_train:
    trans = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    trans[i] = 1
    trans = np.array([trans])
    real_y_train = np.vstack((real_y_train, trans))


# build up the testing set output
for i in y_test:
    trans = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    trans[i] = 1
    trans = np.array([trans])
    real_y_test = np.vstack((real_y_test, trans))



model = Sequential()

filters_1 = 32
filters_2 = 32
filters_3 = 64

kernal_size = (3, 3)

input_shape = (150, 150, 3)
pool_size = (2, 2)


model.add(Conv2D(filters_1, kernal_size, input_shape = (28, 28)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = pool_size))

model.add(Conv2D(filters_2, kernal_size))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = pool_size))

model.add(Conv2D(filters_3, kernal_size))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = pool_size))

model.add(Flatten())
model.add(Dense(32, activation = 'relu'))
#model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


model.fit(x_train, real_y_train, epochs = 20)

model.save('object_regonize.h5')

