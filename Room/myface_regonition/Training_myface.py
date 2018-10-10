import os
from sklearn.neural_network import MLPClassifier
import face_recognition as face
from sklearn import metrics
import pickle
from sklearn.externals import joblib


clf = MLPClassifier(hidden_layer_sizes = (500,500))

x = []
y = []

for i in os.listdir("Train_myface"):
    print(i)
    img = face.load_image_file("Train_myface/" + i)
    encode = face.face_encodings(img)[0]
    x.append(encode)
    y.append(1)

for i in os.listdir("Train_male"):
    print(i)
    img = face.load_image_file("Train_male/" + i)
    encode = face.face_encodings(img)[0]
    x.append(encode)
    y.append(0)



for i in os.listdir("Train_female"):
    print(i)
    img = face.load_image_file("Train_female/" + i)
    encode = face.face_encodings(img)[0]
    x.append(encode)
    y.append(0)



    ##print(i)




clf.fit(x,y)
joblib.dump(clf, "my_face.pkl")
'''
y_test = []
for i in range(0, 41):
    y_test.append(0)

img = face.load_image_file("/input_female.jpg")
encode =face.face_encodings(img)
y_out = clf.predict(encode)
print(metrics.accuracy_score(y_test, y_out))
'''
'''
for i in os.listdir("../Desktop/Test_face"):
    img = face.load_image_file("../Desktop/Test_face/" + i)
    encode = face.face_encodings(img)
    output = clf.predict(encode)[0]
    if(output == 0):
        print(i , "is a female picture")
    if(output == 1):
        print(i , "is a male picture")
    ##print(clf.predict(encode))
'''


##print(type(encode))

##print(len(encode))
