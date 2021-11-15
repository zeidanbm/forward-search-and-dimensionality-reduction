import time
import pandas as pd
import numpy as np
from random import randrange
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.discriminant_analysis
from sklearn import tree

df = pd.read_csv('/home/batman/Desktop/McMaster/data/online_shoppers_intention.csv')
df.describe()

print(df.info())

# check for null data
print(pd.isnull(df).sum())

# check for na data
print(pd.isna(df).sum())

# Drop all categorical and binary features
df = df.drop(['Weekend', 'VisitorType', 'TrafficType', 'Region', 'Browser', 'OperatingSystems', 'Month'], axis=1)

# Visualize the number of purchase and no purchase
sns.countplot(x=df['Revenue'], label='Count')
plt.show()

# Drop the labeling column from X and store in y
# The Revenue is a True/False labeling so we will change that to 1/0
X = df.drop('Revenue', axis=1)
y = df['Revenue']*1

# convert to numpy arrays
X = pd.DataFrame(X).to_numpy()
y = np.array(y)
TOTAL_FEATURES = np.size(X[0])

# Split a dataset into a train and test set
def data_split(X, y, split=0.75):
  X_train = list()
  y_train = list()
  train_size = split*len(X)
  X_copy = list(X)
  Y_copy = list(y)
  while len(X_train) < train_size:
    index = randrange(len(X_copy))
    y_train.append(Y_copy.pop(index))
    X_train.append(X_copy.pop(index))
  return np.array(X_train), np.array(X_copy), np.array(y_train), np.array(Y_copy)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = data_split(X, y)

print('Training data size %d' % len(y_train))
print('Test data size %d' % len(y_test))

# Scale and Center the data
u = np.mean(X_train, 0)
std = np.std(X_train, 0)
X_train_mc = (X_train - u)/std
X_test_mc = (X_test - u)/std

# Calculate the Covariance Matrix
start_time = time.time()
cov = np.dot(X_train_mc.T, X_train_mc)

# Calculate the Eigenvalues and Eigenvectors of the covariance matrix
D, Vt = np.linalg.eig(cov)

# Get Eigenvalue index sorted low to high
sortIndex = np.argsort(D)

# Sort Eigenvectors corresponding to each eigenvalue
newVt = np.zeros((10, 10))

for i in range(10):
  newVt[:, i] = Vt[:, sortIndex[i]]

# Projection to the Principle components
z = np.dot(X_train_mc, newVt)
end_time = time.time() - start_time
z_test = np.dot(X_test_mc, newVt)
print("Inital Time for PCA calculations: {}".format(end_time))

lda1 = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
recon_errors = []
lda1_errors = []
dt1_errors = []
time1 = [[0] * 5 for i in range(8)]
time1_labels = ["PCA", "LDA Training", "LDA Testing", "TD Training", "TD Testing"]
confusion_lda1 = []
confusion_dt1 = []

# Reduce dimensionality and compute the projection of X into the principle axes
for i in range(0, 8):
    start_time = time.time()
    # reduce dimension
    Zreduced = z[:, i:TOTAL_FEATURES]
    Zreduced_test = z_test[:, i:TOTAL_FEATURES]
    Xrecon = np.dot(Zreduced, newVt[:, i:TOTAL_FEATURES].T)
    recon_errors.append((np.square(np.subtract(Xrecon, X_train_mc))).mean())
    time1[i][0] = time.time() - start_time

    # Attempt to fit the reconstructed data with LDA
    start_time = time.time()
    lda1.fit(Zreduced, y_train)
    time1[i][1] = time.time() - start_time

    # Compute the classification error
    start_time = time.time()
    pred = lda1.predict(Zreduced_test)
    lda1_errors.append(sum(abs(pred - y_test)))
    time1[i][2] = time.time() - start_time

    # Calculating Confusion Matrix - LDA
    cm = np.zeros((2, 2))
    for a, p in zip(y_test, pred):
      cm[a][p] += 1
    confusion_lda1.append(cm)

    # Using DT
    classification_tree = tree.DecisionTreeClassifier(max_depth=4, min_samples_split=6)
    # Train our decision tree
    start_time = time.time()
    classification_tree = classification_tree.fit(Zreduced, y_train)
    time1[i][3] = time.time() - start_time

    # Compute the classification error
    start_time = time.time()
    pred = classification_tree.predict(Zreduced_test)
    dt1_errors.append(sum(abs(pred - y_test)))
    time1[i][4] = time.time() - start_time

    # Calculating Confusion Matrix - DT
    cm = np.zeros((2, 2))
    for a, p in zip(y_test, pred):
      cm[a][p] += 1
    confusion_dt1.append(cm)

def generate_time_table(time, labels):
  row_format ="{:>15}" * (len(labels) + 1)
  print(row_format.format("", *labels))
  for row in time:
    print(row_format.format('', *np.round(row, 4)))

  lines = plt.plot(time)
  plt.legend(lines, labels)

generate_time_table(time1, time1_labels)

def generate_cm(cm):
  fig, axs = plt.subplots(4, 2, figsize=(12, 8))
  for i in range(len(cm)):
    print("Confusion matrix for {} features \n {} \n".format(TOTAL_FEATURES-i, cm[i]))
    sns.heatmap(cm[i]/np.sum(cm[i]), annot=True, fmt='.2%', cmap='Blues', ax=axs.flat[i])

generate_cm(confusion_lda1)

generate_cm(confusion_dt1)

# Plot Reconstruction MSE w.r.t Number of Principle Components
plt.plot([10, 9, 8, 7, 6, 5, 4, 3], recon_errors, c='r', marker='o')
plt.title('PCA')
plt.xlabel('Number of Principle Components')
plt.ylabel('Reconstruction MSE')
plt.show()
# Plot Classification Error w.r.t Retained Features - LDA
plt.plot([10, 9, 8, 7, 6, 5, 4, 3], lda1_errors, c='b', marker='o')
plt.title('LDA Classification')
plt.xlabel('Retained Features')
plt.ylabel('Classification Error')
plt.show()
# Plot Classification Error w.r.t Retained Features - DT
plt.plot([10, 9, 8, 7, 6, 5, 4, 3], dt1_errors, c='b', marker='o')
plt.title('DT Classification')
plt.xlabel('Retained Features')
plt.ylabel('Classification Error')
plt.show()

# Using Forward Search to extract features from the original data
# This class will be used later in applying the forward search
class ForwardSearch:

  def __init__(self, X, y):
    self.X = X
    self.y = y
    self.bestFeatures = []
    self.cols = [str(i) for i in range(X[0].size)]

  def get_best_features(self, X, keep = 1):
    for k in range(keep):
      feature_col = self.get_next_feature_with_min_error()
      self.bestFeatures.append(feature_col)
      self.cols.remove(str(feature_col))
    return self.X[:, self.bestFeatures]

  def get_next_feature_with_min_error(self):
    errors = 1000000 * np.ones(np.size(self.X[0]))
    for (index, feature) in enumerate(self.cols):
      errors[int(feature)] = self.find_error_using_lda(self.X[:, self.bestFeatures + [int(feature)]])
    return np.argmin(errors)

  def find_error_using_lda(self, selected_X):
    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    lda.fit(selected_X, self.y)
    pred = lda.predict(selected_X)
    return sum(abs(pred - self.y))

  def find_error_using_dt(self, selected_X):
    classification_tree = tree.DecisionTreeClassifier(max_depth=6)
    classification_tree.fit(selected_X, self.y)
    pred = classification_tree.predict(selected_X)
    return sum(abs(pred - self.y))

lda2_errors = []
dt2_errors = []
fs = ForwardSearch(X_train_mc, y_train)
lda2 = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
time2 = [[0] * 5 for i in range(8)]
time2_labels = ["Forward Search", "LDA Training", "LDA Testing", "TD Training", "TD Testing"]
confusion_lda2 = []
confusion_dt2 = []
# Perform Forward Search to remove features then use lda to do the fitting and calculate the prediction errors
for i in range(3, 11):
  start_time = time.time()
  X_rd = fs.get_best_features(X_train_mc, 3) if i == 3 else fs.get_best_features(X_rd)
  time2[i-3][0] = time.time() - start_time

  # Train with LDA
  start_time = time.time()
  lda2.fit(X_rd, y_train)
  time2[i-3][1] = time.time() - start_time

  # Compute the classification error
  start_time = time.time()
  X_rd_test = X_test_mc[:, fs.bestFeatures]
  pred = lda2.predict(X_rd_test)
  lda2_errors.append(sum(abs(pred - y_test)))
  time2[i-3][2] = time.time() - start_time

  # Calculate Confusion Matrix
  cm = np.zeros((2, 2))
  for a, p in zip(y_test, pred):
    cm[a][p] += 1
  confusion_lda2.append(cm)

  # Using DT
  classification_tree = tree.DecisionTreeClassifier(max_depth=6)
  # Train our decision tree
  start_time = time.time()
  classification_tree.fit(X_rd, y_train)
  time2[i-3][3] = time.time() - start_time

  # Compute the classification error
  start_time = time.time()
  pred = classification_tree.predict(X_rd_test)
  dt2_errors.append(sum(abs(pred - y_test)))
  time2[i-3][4] = time.time() - start_time

  # Calculating Confusion Matrix - DT
  cm = np.zeros((2, 2))
  for a, p in zip(y_test, pred):
    cm[a][p] += 1
  confusion_dt2.append(cm)

generate_time_table(time2, time2_labels)

generate_cm(confusion_lda2)

generate_cm(confusion_dt2)

# Plot Classification Error w.r.t Retained Features
plt.plot([3, 4, 5, 6, 7, 8, 9, 10], lda2_errors, c='r', marker='o')
plt.title('LDA Classification')
plt.xlabel('Retained Features')
plt.ylim([0, np.max(lda2_errors) + 300])
plt.ylabel('Classification Error')
plt.show()

# Plot Classification Error w.r.t Retained Features - DT
plt.plot([3, 4, 5, 6, 7, 8, 9, 10], dt2_errors, c='b', marker='o')
plt.title('DT Classification')
plt.xlabel('Retained Features')
plt.ylabel('Classification Error')
plt.show()