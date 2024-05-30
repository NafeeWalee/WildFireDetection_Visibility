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
from keras.callbacks import ReduceLROnPlateau

import tensorflow as tf  # supports till python version 3.11.x, does not work on python 3.12 (10th January,2024) 
import random as rn 

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2
from tqdm import tqdm
import os
from random import shuffle
from zipfile import ZipFile
from PIL import Image

tf.compat.v1.ragged.RaggedTensorValue

DIR = rf"{os.path.dirname(__file__)}"
X=[]
Z=[]
X_original = []  # To store the original images
IMG_SIZE = 150

TEST_SIZE = 0.20

batch_size = 16  # 32 #64 #128
epochs = 20

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
       
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred_image = cv2.GaussianBlur(gray_image, (1, 1), 0)

        # Perform adaptive thresholding
        binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Perform morphology operations to clean up the image
        kernel = np.ones((1,1), np.uint8)
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

        # Calculate the total number of pixels
        total_pixels = np.prod(binary_image.shape)

        # Calculate the number of visible pixels
        visible_pixels = np.count_nonzero(binary_image)

        # Calculate the visibility ratio
        visibility_ratio = visible_pixels  / total_pixels

        if visibility_ratio  > 0.80 and visibility_ratio  <= 1:
            label = "Very High"
        elif visibility_ratio  > 0.60 and visibility_ratio  <= 0.80:
            label = "High"
        elif visibility_ratio  > 0.40 and visibility_ratio  <= 0.60:
            label = "Moderate"
        elif visibility_ratio  > 0.20 and visibility_ratio  <= 0.40:
            label = "Low"
        else:
            label = "Very Low"

        output_label_dir = os.path.join(visibility_output_DIR, imgCls, label)
        if not os.path.exists(output_label_dir):
            os.makedirs(output_label_dir)

        output_path = os.path.join(output_label_dir, img_name)
        cv2.imwrite(output_path, img)

    if(imgCls == 'Fire'):
        Fire_New_DIR = rf"{visibility_output_DIR}\{imgCls}"
    else:
        noFire_New_DIR = rf"{visibility_output_DIR}\{imgCls}"

def calculate_nwdi(image):
    green = image[:, :, 1]
    blue = image[:, :, 0]
    nwdi = (green - blue) / (green + blue + 1e-10)  # Adding a small value to avoid division by zero
    return nwdi

def apply_nwdi_mask(image):
    nwdi = calculate_nwdi(image)
    
    water_surface_mask = np.where((nwdi > 0.2) & (nwdi <= 1), 1, 0)
    flooding_humidity_mask = np.where((nwdi > 0) & (nwdi <= 0.2), 1, 0)
    moderate_drought_mask = np.where((nwdi > -0.3) & (nwdi <= 0), 1, 0)
    drought_mask = np.where((nwdi >= -1) & (nwdi <= -0.3), 1, 0)

    image[water_surface_mask == 1] = [0, 0, 0]  # Color water surfaces in blue
    #image[flooding_humidity_mask == 1] = [0, 255, 0]  # Color flooding/humidity areas in green
    #image[moderate_drought_mask == 1] = [0, 255, 255]  # Color moderate drought areas in yellow
    #image[drought_mask == 1] = [255, 0, 0]  # Color drought areas in red
    
    return image

def make_train_data(imgCls, base_dir):
    for folder_name in tqdm(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, folder_name)

        for filename in os.listdir(folder_path):
            if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(folder_path, filename)

                img = cv2.imread(image_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                X_original.append(np.array(img))  # Save the original image
                img = apply_nwdi_mask(img)  # Apply NWDI mask

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
X_original = np.array(X_original)  # Convert original images list to numpy array
X_original = X_original / 255  # Normalize original images

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=42)
x_train_orig, x_test_orig, _, _ = train_test_split(X_original, Y, test_size=TEST_SIZE, random_state=42)  # Split original images
print(len(x_train), len(x_test))

np.random.seed(42)
rn.seed(42)
tf.random.set_seed(42)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=96, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=96, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(2, activation="softmax"))

batch_size = 16
epochs = 10

red_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.1)

datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['AUC', 'accuracy'])

model.summary()

History = model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    epochs=epochs, validation_data=(x_test, y_test),
    verbose=1, steps_per_epoch=len(x_train) // batch_size, callbacks=[red_lr]
)

model.save('fire_detection_model.h5')

plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.savefig("Model_Loss.png")
plt.show()

plt.plot(History.history['accuracy'])
plt.plot(History.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.savefig("Model_Accuracy.png")
plt.show()

plt.plot(History.history['auc'])
plt.plot(History.history['val_auc'])
plt.title('Model AUC')
plt.ylabel('AUC')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.savefig("Model_AUC.png")
plt.show()

print("[INFO] evaluating network ... ")
predidxs = model.predict(x_test, batch_size=batch_size)
predidxs = np.argmax(predidxs, axis=1)

report = classification_report(y_test.argmax(axis=1), predidxs, target_names=lb.classes_, output_dict=True)
plt.figure(figsize=(6, 6))
sns.heatmap(pd.DataFrame(report).transpose(), annot=True, cmap="YlGnBu", fmt='.2f', linewidths=.5)
plt.title('Classification Report')
plt.xlabel('Metrics')
plt.ylabel('Classes')
plt.savefig("Classification_Report.png")
plt.show()
print(classification_report(y_test.argmax(axis=1), predidxs, target_names=lb.classes_))

y_test = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_test, predidxs, normalize='pred')
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lb.classes_)
disp.plot()
plt.savefig("Confusion_Matrix.png")
plt.show()

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
plt.savefig("Loss_and_Accuracy.png")
plt.show()

# Visualize some processed images with comparison to the original
fig, axes = plt.subplots(nrows=2, ncols=min(len(x_test), 5), figsize=(15, 6))
for i in range(min(len(x_test), 5)):
    axes[0, i].imshow(x_test_orig[i])
    axes[0, i].set_title("Original")
    axes[0, i].axis('off')
    axes[1, i].imshow(x_test[i])
    axes[1, i].set_title("Processed")
    axes[1, i].axis('off')
plt.show()
