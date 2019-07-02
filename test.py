"""
Evaluation of the trained network with the test dataset,
based on weights providede in 'model.h5' file.
"""
import time
start_time = time.time()
import sys
import numpy as np
import cv2
import csv
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers.core import Dense, Flatten

if (len(sys.argv) == 2):
    path = sys.argv[1]
elif (len(sys.argv) == 1):
    path = '.'
else:
    raise Exception('Execution syntax: "python3 test.py" or "python3 test.py dataset_path"')

# getting test images list from .csv file
with open(path + '/test.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    test_img_names = []
    for row in csv_reader:
        test_img_names.append(row)

test_img_labels = []
test_img = []

# loading test images
for i in range (len(test_img_names)):
    img = cv2.imread(path + '/' + test_img_names[i][0])
    img = cv2.resize(img, (250,250))
    if (img is None):
        raise FileNotFoundError('File {} not found.'.format(test_img_names[i][0]))
    test_img.append(img)
    test_img_labels.append(int(test_img_names[i][1]))

test_img = np.array(test_img, np.float64)
test_img_labels = np.array(test_img_labels)

# loading model
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
# loading weights generated from 'treino.py' script
model.load_weights('model.h5')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# evaluate the model
score = model.evaluate(test_img, test_img_labels, verbose=1)
test_error = 1 - score[1]

# printing metrics
output_text = 'Erro de teste: {test_error}%'.format(
    test_error=round(test_error * 100, 2))

incorrect = np.nonzero(model.predict_classes(test_img).reshape((-1,)) != test_img_labels)[0]
if (len(incorrect) != 0):
    output_text += '\nIncorrectly classified files:'
for idx in incorrect:
    output_text += ('\n' + test_img_names[idx][0])
output_text += '\n\nProcessing time: {exec_time} seconds'.format(
    exec_time=round(time.time() - start_time, 2)
    )

print(output_text)