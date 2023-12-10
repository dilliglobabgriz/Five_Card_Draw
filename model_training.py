import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Dropout
from keras.models import load_model


try:
    final_df = pd.read_csv('fivecarddraw_400k.csv')
except FileNotFoundError:
    print(f'The CSV file fivecarddraw.csv was not found.\n')

#let's get an idea of what the dataframe looks like
print(final_df.info())


# Feature list is updated regularly, but the current version has the following parameters
# rank and suit represent the respective rank and suit of each card in the starting hand
# Class is the hand class where high card is 9 down to royal flush as 0
# S pos is the position of the card being swapped in the hand
# S rank and suit at the rank of suit of the new card in the hand
# S class is the class of the swapped hand
#feature_list = ['c1_rank', 'c2_rank', 'c3_rank', 'c4_rank', 'c5_rank', 'c1_suit', 'c2_suit', 'c3_suit', 'c4_suit', 'c5_suit', 'class', 's_pos', 's_rank', 's_suit', 's_class']
feature_list = ['c1_rank', 'c2_rank', 'c3_rank', 'c4_rank', 'c5_rank', 'c1_suit', 'c2_suit', 'c3_suit', 'c4_suit', 'c5_suit', 'class', 's_pos']

X_df = final_df[feature_list]
#X_df = np.array(final_df[feature_list]).astype(np.float32)

y_df = final_df['outcome']
#y_df = np.array(final_df['outcome']).astype(np.float32).reshape(-1,1)

#next, break up the data into trining data and testing data...20% of the data will be used to evaluate
#the model, and 80% of the data will be used to train the model.  You can change these parameters
#to explore the impact.  we are using the train_test_split method we imported.
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size = 0.2)


'''
we will set up a neural net with 5 layers, each layer will have a different number of nodes
again, play with these parameters to see if there is an impact on the accuracy of the model.
be curious about these parameters!  
  
https://keras.io/guides/sequential_model/

In a neural network, the activation function is responsible for transforming the summed weighted input 
from the node into the activation of the node or output for that input.

The rectified linear activation function or ReLU for short is a piecewise linear function that will output 
the input directly if it is positive, otherwise, it will output zero. It has become the default activation f
unction for many types of neural networks because a model that uses it is easier to train and often achieves better performance.

The sigmoid and hyperbolic tangent activation functions cannot be used in networks with many layers due to the vanishing gradient problem.
The rectified linear activation function overcomes the vanishing gradient problem, allowing models to learn faster and perform better.
The rectified linear activation is the default activation when developing multilayer Perceptron and convolutional neural networks.

play with the different activation functions.

An epoch is a single iteration through the training data.  The more epochs, the more the model is trained.  Be careful not to overfit
the data.  Of course, there are a finite number of dealer card/player hands to consider...so what would it mean to overfit the data?
'''

model = Sequential()
model.add(Dense(32, activation='relu'))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64, activation='softmax'))
model.add(Dense(32))
model.add(Dense(16, activation='relu'))
model.add(Dense(8))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

#train the model
model.fit(X_train, y_train, epochs=20, batch_size=256, verbose=1)

#make some predictions based on the test data that we reserved
pred_Y_test = model.predict(X_test)
#also get the actual results so we can compare
actuals = y_test

#evaluate the model...check out the various metrics used to evaluate a model...you can do your own search
#   https://neptune.ai/blog/performance-metrics-in-machine-learning-complete-guide


fpr, tpr, threshold = metrics.roc_curve(actuals, pred_Y_test)
roc_auc = metrics.auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(10,8))
plt.plot(fpr, tpr, label = ('ROC AUC = %0.3f' % roc_auc))

plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
ax.set_xlabel('False Positive Rate', fontsize=16)
ax.set_ylabel('True Positive Rate', fontsize=16)
plt.setp(ax.get_legend().get_texts(), fontsize=16)
plt.tight_layout()
plt.savefig(fname='roc_curve_blackjack', dpi=150)
plt.show()


print(model.summary())

#we an save the model and then load it to continue where we left off
model.save('basic_model.keras')


#NEXT: use the model to determine cozmo's course of action

# Load the trained model if you've saved it earlier
loaded_model = load_model('basic_model.keras')

# Make predictions on the test data
pred_Y_test = loaded_model.predict(X_test)

# Convert the model's probability predictions to binary (0 or 1) based on a threshold
threshold = 0.7  # You can adjust this threshold
pred_Y_test_binary = (pred_Y_test > threshold).astype(int)

# Calculate the confusion matrix
confusion = metrics.confusion_matrix(y_test, pred_Y_test_binary)

# Visualize the confusion matrix using a heatmap
sns.heatmap(confusion, annot=True, fmt="d", cmap="Greens", xticklabels=["1", "0"], yticklabels=["1", "0"])
plt.title(f'Confusion Matrix')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Calculate and print additional classification metrics
accuracy = metrics.accuracy_score(y_test, pred_Y_test_binary)
precision = metrics.precision_score(y_test, pred_Y_test_binary)
recall = metrics.recall_score(y_test, pred_Y_test_binary)
f1_score = metrics.f1_score(y_test, pred_Y_test_binary)

print("Accuracy:", round(accuracy, 3))
print("Precision:", round(precision, 3))
print("Recall:", round(recall, 3))
print("F1 Score:", round(f1_score, 3))