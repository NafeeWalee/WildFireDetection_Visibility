import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.metrics import Precision

import tensorflow as tf  # supports till python version 3.11.x, does not work on python 3.12 (10th January,2024) 
import random as rn 

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2
from tqdm import tqdm
import os
from random import shuffle
from zipfile import ZipFile
from PIL import Image


DIR = rf"{os.path.dirname(__file__)}"
X=[]
Z=[]
IMG_SIZE = 150

TEST_SIZE = 0.20

batch_size = 16  # 32 #64 #128
epochs = 200

#evaluation results directory
res_DIR = rf"{DIR}\Results" 

#assuming dataset folder is in the same folder as this file
noFire_DIR= rf"{DIR}\wildfire_detection_dataset\noFire" 
Fire_DIR= rf"{DIR}\wildfire_detection_dataset\Fire"

#new directory for fire and noFire
visibility_output_DIR= rf"{DIR}\wildfire_detection_dataset\visibilityOutput"

#directory path will dynamically be added through loop
noFire_New_DIR = ''
Fire_New_DIR = ''

def detect_visibility(imgCls, input_dir, visibility_output_DIR):
    global Fire_New_DIR
    global noFire_New_DIR

    if not os.path.exists(visibility_output_DIR):
            os.makedirs(visibility_output_DIR)

    for img_name in tqdm(os.listdir(input_dir)):
        img_path = os.path.join(input_dir, img_name )
        img = cv2.imread(img_path)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray)

        if mean_intensity >= 81:
            label = "High"
        elif mean_intensity >= 50 and mean_intensity <= 80:
            label = "Moderate"
        else:
            label = "Low"

        output_label_dir = os.path.join(visibility_output_DIR, imgCls, label)
        if not os.path.exists(output_label_dir):
            os.makedirs(output_label_dir)

        output_path = os.path.join(output_label_dir, img_name)
        cv2.imwrite(output_path, img)

    if(imgCls == 'Fire'):
        Fire_New_DIR = rf"{visibility_output_DIR}\{imgCls}"
    else:
        noFire_New_DIR = rf"{visibility_output_DIR}\{imgCls}"

def make_train_data(imgCls, base_dir):
    for folder_name in tqdm(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, folder_name)

        for filename in os.listdir(folder_path):
            if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # Adjust extensions as needed
                image_path = os.path.join(folder_path, filename)

                img = cv2.imread(image_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                X.append(np.array(img))
                Z.append(str(imgCls))

detect_visibility('Fire', Fire_DIR, visibility_output_DIR)

detect_visibility('noFire', noFire_DIR, visibility_output_DIR)

make_train_data('Fire', Fire_New_DIR)
print(len(X))

make_train_data('noFire', noFire_New_DIR)
print(len(X))

lb=LabelBinarizer()
Y=lb.fit_transform(Z)
Y=to_categorical(Y,2)
X=np.array(X)
X=X/255

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=42)
print(len(x_train), len(x_test))

np.random.seed(42)
rn.seed(42)
tf.random.set_seed(42)

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (150, 150, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(2, activation = "softmax"))

from keras.callbacks import ReduceLROnPlateau
red_lr= ReduceLROnPlateau(monitor='val_acc' ,patience=3,verbose=1,factor=0.1)

datagen = ImageDataGenerator(
    featurewise_center=False, # set input mean toe over the dataset
    samplewise_center=False, # set each sample mean toe
    featurewise_std_normalization = False, # divide inputs by std of the dataset
    samplewise_std_normalization=False, # divide each input by its std
    zca_whitening=False, # apply ZCA whitening
    rotation_range = 0.2, # randomly rotate images in the range (degrees, e to 180)
    zoom_range = 0.1, # Randomly zoom image
    width_shift_range=0.2, # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2, # randomly shift images vertically (fraction of total height)
    horizontal_flip=False, # randomly flip images
    vertical_flip=True) # randomly flip images

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['AUC', 'accuracy'])  #Precision(thresholds=0), 

model.summary()
print(model.summary())

History = model.fit(datagen.flow(x_train, y_train, batch_size = batch_size),
    epochs = epochs, validation_data = (x_test,y_test),
    verbose = 1, steps_per_epoch = x_train.shape[0] // batch_size)

if not os.path.exists(res_DIR):
            os.makedirs(res_DIR)

filename = rf"{res_DIR}/test_size_{TEST_SIZE}_epoch_{epochs}_batch_size_{batch_size}_visibility"

plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend([ 'train', 'test'])
plt.savefig(rf"{filename}_Model_Loss.png")
#plt.show()

plt.plot(History.history['accuracy'])
plt.plot(History.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epochs')
plt.legend([ 'train', 'test'])
plt.savefig(rf"{filename}_Model_Accuracy.png")
#plt.show()

plt.plot(History.history['auc'])
plt.plot(History.history['val_auc'])
plt.title('Model AUC')
plt.ylabel('AUC')
plt.xlabel('Epochs')
plt.legend([ 'train', 'test'])
plt.savefig(rf"{filename}_Model_AUC.png")
# plt.show()

# make predictions on the testing set
print("[INFO] evaluating network ... ")
predidxs = model.predict(x_test, batch_size=batch_size)
predidxs = np.argmax(predidxs, axis = 1)

report = classification_report(y_test.argmax(axis = 1), predidxs, target_names = lb.classes_, output_dict=True)
plt.figure(figsize=(6, 6))
sns.heatmap(pd.DataFrame(report).transpose(), annot=True, cmap="YlGnBu", fmt='.2f', linewidths=.5)
plt.title('Classification Report')
plt.xlabel('Metrics')
plt.ylabel('Classes')
plt.savefig(rf"{filename}_Classification_Report.png")
# plt.show()
print(classification_report(y_test.argmax(axis = 1), predidxs, target_names = lb.classes_))

y_test=np.argmax(y_test, axis=1)
#Create confusion matrix and normalizes it over predicted (columns)
cm = confusion_matrix(y_test, predidxs , normalize='pred')
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = lb.classes_)
disp.plot()
plt.savefig(rf"{filename}_Confusion_Matrix.png")
# plt.show()

N = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), History.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), History.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), History.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), History.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(rf"{filename}_visibility_Loss_and_Accuracy.png")
# plt.show()

a=1