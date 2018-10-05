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



path_lab = "../../../dog_cat/train/"
#path_own_comp = "../../Desktop/dog_cat/train/"
# all image would have a input size of 4608, 4608, 3
M = 150
N = 150






training_input = []
training_output = []

print("Start inputing data")

counter = 0
x_num = 0
y_num = 0
for i in os.listdir(path_lab):
    #print(i)
    if(i.endswith('.jpg')):
        if(x_num + y_num == 5000):
            break
        image = load_img(path_lab + i, target_size = (150, 150))
        array_image = img_to_array(image)
        #print(array_image.shape)
        if(i.startswith('dog')):
            if(not x_num == 2500):
                training_input.append(array_image)
                training_output.append([1, 0])
                x_num += 1
        if(i.startswith('cat')):
            if(not y_num == 2500):
                training_input.append(array_image)
                training_output.append([0, 1])
                y_num += 1
        counter = counter+1
        precentage = (counter / 25000) * 100
        if(precentage % 10 == 0):
            print(precentage, "%")


'''
print("Doors finish")

#print("Max M for Doors is", M)
#print("Max N for Doors is", N)

for i in os.listdir(path + "Sign"):
    if(not i.endswith('db')):
        image = load_img(path + "Sign/" + i, target_size = (150, 150))
        array_image = img_to_array(image)
        training_input.append(array_image)
        training_output.append([0, 1, 0])
        
print("Sign finish")
#print("Max M for Doors + Sign is", M)
#print("Max N for Doors + Sign is", N)

for i in os.listdir(path + "Stairs"):
    if(not i.endswith('db')):
        image = load_img(path + "Stairs/" + i, target_size = (150, 150))
        array_image = img_to_array(image)
        training_input.append(array_image)
        training_output.append([0, 0, 1])
        

#print("Max M for all is", M)
#print("Max N for all is", N)
print("Stairs finish")
'''
print("Data cleaning finish")

training_input = np.array(training_input)
training_output = np.array(training_output)


print(type(training_input))
print(type(training_input[1]))

with h5py.File('cnn_object_input.hdf5', 'w') as f:
    write_in = f.create_dataset('input_data', data = training_input)
    write_out = f.create_dataset('output_data', data = training_output)

'''
with h5py.File('cnn_object_input.hdf5', 'r') as f:
    training_input = f['input_data']
    training_output = f['output_data']
'''



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
'''


model = Sequential()

filters_1 = 32
filters_2 = 32
filters_3 = 64

kernal_size = (3, 3)

input_shape = (150, 150, 3)
pool_size = (2, 2)


model.add(Conv2D(filters_1, kernal_size, input_shape = (150, 150, 3)))
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
model.add(Dense(2, activation = 'softmax'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


model.fit(training_input, training_output, epochs = 20)

model.save('object_regonize.h5')

test_path_own = "../../Desktop/dog_cat/test1/"
test_path_lab = "../../../dog_cat/test1/"

while(1 == 1):
    x = int(input("Please input a number between 1 - 12500, quit by input -1"))
    if(x == -1):
        break
    else:
        x = str(x) + ".jpg"
        image = load_img(test_path + x, target_size = (150, 150))
        array_image = img_to_array(image)
        print(model.predict(arr_image))
