import sys
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

if (len(sys.argv) == 2):
    path = sys.argv[1]
elif (len(sys.argv) == 1):
    path = '.'
else:
    raise Exception('Execution syntax: "python3 generate_dataset.py" or "python3 generate_dataset.py pictures_path"')

pictures = glob.glob(path + "/*.jpg")
dataset = pd.DataFrame(columns=['name', 'label'])

for filename in pictures:
    filename = filename.split('/')[-1]
    if filename[:3] == 'cat':
        label = 1
    elif filename[:3] == 'dog':
        label = 0
    dataset = dataset.append(pd.DataFrame({
        'name':[filename],
        'label':[label]
        }
    ), ignore_index=True, sort=False)

print('Dogs: {dogs}\tCats: {cats}'.format(
    dogs=dataset[dataset['label'] == 0].shape[0],
    cats=dataset[dataset['label'] == 1].shape[0]
    ))

train, test = train_test_split(dataset, test_size=.2)
train, valid = train_test_split(train, test_size=.4)

print(
"""
Train:
\tDogs: {dogs_train}\tCats: {cats_train}
Validation:
\tDogs: {dogs_valid}\tCats: {cats_valid}
Test:
\tDogs: {dogs_test}\tCats: {cats_test}
""".format(
    dogs_train=train[train['label'] == 0].shape[0],
    cats_train=train[train['label'] == 1].shape[0],
    dogs_valid=valid[valid['label'] == 0].shape[0],
    cats_valid=valid[valid['label'] == 1].shape[0],
    dogs_test=test[test['label'] == 0].shape[0],
    cats_test=test[test['label'] == 1].shape[0],
))

train.to_csv(path + '/train.csv', sep=';', index=False, header=False)
test.to_csv(path + '/test.csv', sep=';', index=False, header=False)
valid.to_csv(path + '/valid.csv', sep=';', index=False, header=False)
