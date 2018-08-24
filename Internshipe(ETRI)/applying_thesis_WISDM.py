
# coding: utf-8

# In[8]:



import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
from sklearn import metrics
from sklearn.model_selection import train_test_split


PLOT_ACTIVITIES = False
SAVE_MODEL = True
LOAD_MODEL = True
PLOT_RESULT = False
PLOT_RESULT = False
EXPORT_TO_ANDROID = True

# get_ipython().magic('matplotlib inline')
sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 14, 8

RANDOM_SEED = 42

#columns = ['user','activity','timestamp', 'x-axis', 'y-axis', 'z-axis']
df = pd.read_csv('data4/WISDM_final.txt', delimiter = ',')
#df = df.dropna()



N_TIME_STEPS = 200
N_FEATURES = 3
step = 200
segments = []
labels = []
for i in range(0, len(df) - N_TIME_STEPS, step):
    body_acc_x = df['x'].values[i: i + N_TIME_STEPS]
    body_acc_y = df['y'].values[i: i + N_TIME_STEPS]
    body_acc_z = df['z'].values[i: i + N_TIME_STEPS]
    label = stats.mode(df['activity'][i: i + N_TIME_STEPS])[0][0]
    segments.append([body_acc_x, body_acc_y, body_acc_z])
    labels.append(label)
    
#print("Segments size ->")
#print(np.array(segments).shape)
#print("\n")

reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
labels = np.asarray(pd.get_dummies(labels), dtype = np.float32)

X_train, X_test, y_train, y_test = train_test_split(
        reshaped_segments, labels, test_size=0.3, random_state=42)

#print("Reshape segements size ->")

#print(X_train.shape)
#print(y_train.shape)


# In[13]:




# In[14]:

# # Building the model
# 
# Our model contains 2 fully-connected and 2 LSTM layers (stacked on each other) with 64 units each:
N_CLASSES = 6
N_HIDDEN_UNITS = 50

def create_LSTM_model(inputs):
    W = {
        'hidden': tf.Variable(tf.random_normal([N_FEATURES, N_HIDDEN_UNITS])),
        'output': tf.Variable(tf.random_normal([N_HIDDEN_UNITS, N_CLASSES]))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([N_HIDDEN_UNITS], mean=1.0)),
        'output': tf.Variable(tf.random_normal([N_CLASSES]))
    }
    
    X = tf.transpose(inputs, [1, 0, 2])
    X = tf.reshape(X, [-1, N_FEATURES])
    hidden = tf.nn.relu(tf.matmul(X, W['hidden']) + biases['hidden'])
    hidden = tf.split(hidden, N_TIME_STEPS, 0)

    # Stack 2 LSTM layers
    lstm_layers = [tf.contrib.rnn.BasicLSTMCell(N_HIDDEN_UNITS, forget_bias=1.0) for _ in range(2)]
    lstm_layers = tf.contrib.rnn.MultiRNNCell(lstm_layers)

    outputs, _ = tf.contrib.rnn.static_rnn(lstm_layers, hidden, dtype=tf.float32)

    # Get output for the last time step
    lstm_last_output = outputs[-1]

    return tf.matmul(lstm_last_output, W['output']) + biases['output']

# Now, let create placeholders for our model:
tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, N_TIME_STEPS, N_FEATURES], name="input")
Y = tf.placeholder(tf.float32, [None, N_CLASSES])
    
# Note that we named the input tensor, that will be useful when using the model from Android. Creating the model:
pred_Y = create_LSTM_model(X)
pred_softmax = tf.nn.softmax(pred_Y, name="y_")

# Again, we must properly name the tensor from which we will obtain predictions.
# We will use L2 regularization and that must be noted in our loss op:
L2_LOSS = 0.0015

l2 = L2_LOSS *	sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred_Y, labels = Y)) + l2

# Finally, let's define optimizer and accuracy ops:
LEARNING_RATE = 0.00025



optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

correct_pred = tf.equal(tf.argmax(pred_softmax, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

# # Training
# 
# The training part contains a lot of TensorFlow boilerplate.
# We will train our model for 50 epochs and keep track of accuracy and error:

N_EPOCHS = 200
BATCH_SIZE = 1024

saver = tf.train.Saver()

history = dict(train_loss=[], 
                     train_acc=[], 
                     test_loss=[], 
                     test_acc=[])

sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

train_count = len(X_train)

for i in range(1, N_EPOCHS + 1):
    for start, end in zip(range(0, train_count, BATCH_SIZE),
                          range(BATCH_SIZE, train_count + 1,BATCH_SIZE)):
        sess.run(optimizer, feed_dict={X: X_train[start:end],
                                       Y: y_train[start:end]})

    _, acc_train, loss_train = sess.run([pred_softmax, accuracy, loss], feed_dict={
                                            X: X_train, Y: y_train})

    _, acc_test, loss_test = sess.run([pred_softmax, accuracy, loss], feed_dict={
                                            X: X_test, Y: y_test})

    history['train_loss'].append(loss_train)
    history['train_acc'].append(acc_train)
    history['test_loss'].append(loss_test)
    history['test_acc'].append(acc_test)

    if i != 1 and i % 10 != 0:
        continue

    print('epoch: {', i, '} test accuracy: {', acc_test, '} loss: {', loss_test, '}')
    
predictions, acc_final, loss_final = sess.run([pred_softmax, accuracy, loss], feed_dict={X: X_test, Y: y_test})

print('final results: accuracy: {', acc_final, '} loss: {', loss_final, '}')


# Whew, that was a lot of training. Do you feel thirsty?
# Let's store our precious model to disk:

pickle.dump(predictions, open("WISDM_predictions_seg200.p", "wb"))
pickle.dump(y_test, open("WISDM_test_seg200.p", "wb"))
tf.train.write_graph(sess.graph_def, '.', './checkpoint/har.pbtxt')  
saver.save(sess, save_path = "./checkpoint/har.ckpt")
sess.close()

# And loading it back:
predictions = pickle.load(open("WISDM_predictions_seg200.p", "rb"))
test = pickle.load(open("WISDM_test_seg200.p", "rb"))


# # Evaluation

# Our model seems to learn well with accuracy reaching above 97% and loss hovering at around 0.2. Let's have a look at the confusion matrix for the model's predictions:


# In[40]:
LABELS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32']
# In[41]:
max_test = np.argmax(y_test, axis=1)
max_predictions = np.argmax(predictions, axis=1)

print('checkcheckcheckcheckcheckcheckcheck!!!!')
print(y_test.shape)

#confusion_matrix = metrics.confusion_matrix(max_test, max_predictions)
#sns.heatmap(confusion_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");


# Again, it looks like our model performs real good. Some notable exceptions include the misclassification of Upstairs for Downstairs and vice versa. Jogging seems to fail us from time to time as well!
# 
# # Exporting the model
# 
# Now that most of the hard work is done we must export our model in a way that TensorFlow for Android will understand it:

