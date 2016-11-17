import os

import pandas as pd
import tensorflow as tf
import tflearn
from sklearn.model_selection import train_test_split

################################
# Preparing Data
################################

# read data from file
data = pd.read_csv('data/train.csv')

# fill nan values with 0
data = data.fillna(0)
# convert ['male', 'female'] values of Sex to [1, 0]
data['Sex'] = data['Sex'].apply(lambda s: 1 if s == 'male' else 0)
# 'Survived' is the label of one class,
# add 'Deceased' as the other class
data['Deceased'] = data['Survived'].apply(lambda s: 1 - s)

# select features and labels for training
dataset_X = data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]
dataset_X = dataset_X.as_matrix()
dataset_Y = data[['Survived', 'Deceased']]
dataset_Y = dataset_Y.as_matrix()

# split training data and validation set data
X_train, X_val, y_train, y_val = train_test_split(dataset_X, dataset_Y,
                                                  test_size=0.2,
                                                  random_state=42)

# arguments that can be set in command line
tf.app.flags.DEFINE_integer('epochs', 10, 'Training epochs')
FLAGS = tf.app.flags.FLAGS

ckpt_dir = './ckpt_dir'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

# defind model
input = tflearn.input_data([None, 6])
y_pred = tflearn.layers.fully_connected(input, 2, activation='softmax')
net = tflearn.regression(y_pred)
model = tflearn.DNN(net)

# restore model if there is a checkpoint
if os.path.isfile(os.path.join(ckpt_dir, 'model.ckpt')):
    model.load(os.path.join(ckpt_dir, 'model.ckpt'))
# train model
model.fit(X_train, y_train, n_epoch=FLAGS.epochs)
# save the trained model
model.save(os.path.join(ckpt_dir, 'model.ckpt'))
# evaluate on validation set
metric = model.evaluate(X_val, y_val)
print('Accuracy on validation set: %.9f' % metric[0])
