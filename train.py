"""
Design and train of neural network based on pre-trained 
VGG16 model.

Generates and saves .h5 file of neural network weights to
be used with 'test.py' script.
"""
import time
start_time = time.time()
import sys
import numpy as np
import cv2
import csv
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.callbacks import History
from keras.layers.core import Dense, Flatten

if (len(sys.argv) == 2):
    path = sys.argv[1]
elif (len(sys.argv) == 1):
    path = '.'
else:
    raise Exception('Execution syntax: "python3 train.py" or "python3 train.py dataset_path"')

# getting train images list from .csv file
with open(path + '/train.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    train_img_names = []
    for row in csv_reader:
        train_img_names.append(row)
# getting validation images list from .csv file
with open(path + '/valid.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    valid_img_names = []
    for row in csv_reader:
        valid_img_names.append(row)

train_img_labels = []
train_img = []
valid_img_labels = []
valid_img = []
# loading train images
for i in range (len(train_img_names)):
    img = cv2.imread(path + '/' + train_img_names[i][0])
    img = cv2.resize(img, (250,250))
    if (img is None):
        raise FileNotFoundError('File {} not found.'.format(train_img_names[i][0]))
    train_img.append(img)
    train_img_labels.append(int(train_img_names[i][1]))
# loading validation images
for i in range (len(valid_img_names)):
    img = cv2.imread(path + '/' + valid_img_names[i][0])
    img = cv2.resize(img, (250,250))
    if (img is None):
        raise FileNotFoundError('File {} not found.'.format(valid_img_names[i][0]))
    valid_img.append(img)
    valid_img_labels.append(int(valid_img_names[i][1]))

valid_img = np.array(valid_img, np.float64)
valid_img_labels = np.array(valid_img_labels)
valid_img_labels = np.transpose(valid_img_labels)

train_img = np.array(train_img, np.float64)
train_img_labels = np.array(train_img_labels)
train_img_labels = np.transpose(train_img_labels)

# load pretrained model
vgg16_model = VGG16(input_shape=(250, 250, 3), include_top=False)

model = Sequential()
for layer in vgg16_model.layers:
    model.add(layer)
    
for layer in model.layers:
    layer.trainable = False
model.add(Flatten())
model.add(Dense(12, activation='relu', kernel_initializer='random_normal'))
model.add(Dense(15, activation='relu', kernel_initializer='random_normal'))
model.add(Dense(8, activation='relu', kernel_initializer='random_normal'))
model.add(Dense(1, activation='hard_sigmoid', kernel_initializer='normal'))
model.compile(optimizer ='adam', loss='binary_crossentropy', metrics =['accuracy'])

# train model
hist = History()
model.fit(train_img,
          train_img_labels,
          validation_data=(valid_img,valid_img_labels),
          epochs=7,
          verbose=1,
          callbacks=[hist])

# export weights
model.save_weights('model.h5')

# printing metrics
val_error = 1 - hist.history['val_acc'][-1]
train_error = 1 - hist.history['acc'][-1]
output_text = 'Validation error: {val_error}%\nTrain error: {train_error}%\nProcessing time: {exec_time} seconds'.format(
    val_error=round(val_error * 100, 2),
    train_error=round(train_error * 100, 2),
    exec_time=round(time.time() - start_time, 2)
    )

print(output_text)
