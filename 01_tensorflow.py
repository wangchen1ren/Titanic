import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

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
X_train, X_test, y_train, y_test = train_test_split(dataset_X, dataset_Y,
                                                    test_size=0.2,
                                                    random_state=42)

# arguments that can be set in command line
tf.app.flags.DEFINE_integer('epochs', 100, 'Training epochs')
FLAGS = tf.app.flags.FLAGS

# create symbolic variables
X = tf.placeholder(tf.float32, shape=[None, 6])
y_true = tf.placeholder(tf.float32, shape=[None, 2])

# weights and bias are the variables to be trained
weights = tf.Variable(tf.random_normal([6, 2]))
bias = tf.Variable(tf.zeros([2]))
y_pred = tf.nn.softmax(tf.matmul(X, weights) + bias)

# Minimise cost using cross entropy
# NOTE: add a epsilon(1e-10) when calculate log(y_pred),
# otherwise the result will be -inf
cross_entropy = - tf.reduce_sum(y_true * tf.log(y_pred + 1e-10),
                                reduction_indices=1)
cost = tf.reduce_mean(cross_entropy)

# use gradient descent optimizer to minimize cost
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

# use session to run the calculation
with tf.Session() as sess:
    # variables have to be initialized at the first place
    tf.initialize_all_variables().run()

    # training loop
    for epoch in range(FLAGS.epochs):
        total_loss = 0.
        for i in range(len(X_train)):
            # prepare feed data and run
            feed_dict = {X: [X_train[i]], y_true: [y_train[i]]}
            _, loss = sess.run([train_op, cost], feed_dict=feed_dict)
            total_loss += loss
        # display loss per epoch
        print('Epoch: %04d, loss=%.9f' % (epoch + 1, total_loss))
    print("Training complete!")

    # predict on test set
    pred = sess.run(y_pred, feed_dict={X: X_test})
    pred_class = np.argmax(pred, 1)
    true_class = np.argmax(y_test, 1)
    accuracy = np.mean(np.equal(pred_class, true_class).astype(np.float32))
    print("Predict accuracy: %.9f" % accuracy)
