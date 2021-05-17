from django.shortcuts import render

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import pandas as pd
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import random
from keras.utils.np_utils import to_categorical
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Flatten,MaxPooling2D,Dropout
import cv2
from sklearn.metrics import confusion_matrix
import pickle


def disease(request):
    return render(request, 'disease.html')


def submit(request):
    b = request.FILES['img']
    fs = FileSystemStorage()
    f = fs.save(b.name,b)
    '''
    DATASET = "train"
    DATASET2 = "valid"

    CATEGORIES = ["Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___healthy", "Tomato___Late_blight",
                  "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
                  "Tomato___Target_spot", "Tomato___Tomato_mosaic_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"]

    train_data = []

    for category in CATEGORIES:
        label = CATEGORIES.index(category)
        path = os.path.join(DATASET, category)
        print(path)
        for img_file in os.listdir(path):
            img = cv.imread(os.path.join(path, img_file), 1)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = cv.resize(img, (64, 64))
            train_data.append([img, label])

    test_data = []

    for category in CATEGORIES:
        label = CATEGORIES.index(category)
        path = os.path.join(DATASET2, category)
        print(path)
        for img_file in os.listdir(path):
            img = cv.imread(os.path.join(path, img_file), 1)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = cv.resize(img, (64, 64))
            test_data.append([img, label])

    print(len(train_data))
    print(len(test_data))

    random.shuffle(train_data)
    random.shuffle(test_data)

    X_train = []
    y_train = []

    for features, label in train_data:
        X_train.append(features)
        y_train.append(label)

    len(X_train), len(y_train)

    X_test = []
    y_test = []

    for features, label in test_data:
        X_test.append(features)
        y_test.append(label)

    len(X_test), len(y_test)

    X_train = np.array(X_train).reshape(-1, 64, 64, 3)
    X_train = X_train / 255.0
    X_train.shape

    X_test = np.array(X_test).reshape(-1, 64, 64, 3)
    X_test = X_test / 255.0
    X_test.shape

    order = ['BACTERIAL SPOT', 'EARLY BLIGHT', 'HEALTHY', 'LATE BLIGHT', 'LEAF MOLD', 'SEPTORIA LEAF SPOT',
             'SPIDER MITE', 'TARGET SPOT', 'MOSAIC VIRUS', 'YELLOW LEAF CURL VIRUS']
    one_hot_train = to_categorical(y_train)
    one_hot_train

    one_hot_test = to_categorical(y_test)
    one_hot_test
    classifier = Sequential()

    classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.2))

    classifier.add(Conv2D(64, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.2))

    classifier.add(Conv2D(128, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.4))

    classifier.add(Flatten())

    classifier.add(Dense(activation='relu', units=64))
    classifier.add(Dense(activation='relu', units=128))
    classifier.add(Dense(activation='relu', units=64))
    classifier.add(Dense(activation='softmax', units=10))

    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    classifier.summary()

    classifier.fit(X_train, one_hot_train, epochs=1, batch_size=20, validation_split=0.2)
    classifier.save('d_model.h5',save_format='h5')

    y_predi = classifier.predict_classes(X_test)

    y_pred = y_predi.tolist()
    print(y_pred[:4])
    print(y_test[:4])

    print(type(y_pred))
    print(type(y_test))
    array = confusion_matrix(y_test, y_pred)
    print(array)
    '''
    classifier = keras.models.load_model('d_model.h5')
    frame = cv2.imread(b.name)
    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (64, 64))
    img = np.array(img).reshape(-1, 64, 64, 3)
    img = img / 255.0
    img.shape
    order = ['BACTERIAL SPOT', 'EARLY BLIGHT', 'HEALTHY', 'LATE BLIGHT', 'LEAF MOLD', 'SEPTORIA LEAF SPOT',
             'SPIDER MITE', 'TARGET SPOT', 'MOSAIC VIRUS', 'YELLOW LEAF CURL VIRUS']
    res = order[classifier.predict_classes(img)[0]]

    return render(request, 'disease.html',{"res":res})
