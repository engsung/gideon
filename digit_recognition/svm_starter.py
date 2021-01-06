import pandas as pd
import matplotlib.pyplot as plt  
from sklearn import svm
from sklearn import metrics
import joblib
import numpy as np
from sklearn.utils import shuffle

# Step 1: Get Data from CSV
dataframe = pd.read_csv('csv/dataset4labels.csv');
dataframe = dataframe.sample(frac=1).reset_index(drop=True)
print(dataframe)

# Step 2: Separate Labels and Features
x = dataframe.drop(['label'], axis=1)
y = dataframe["label"]

x_train, y_train = x[0:130], y[0:130]
x_test, y_test = x[130:], y[130:]

# Step 3: Make sure you have the correct Feature / label combination in training
grid_data = x_train.values[40].reshape(28,28)
plt.imshow(grid_data, interpolation=None, cmap="gray")
plt.title(y_train.values[40])
plt.show()

# Step 4: Build a Model and Save it
model = svm.SVC(kernel="linear")
model.fit(x_train, y_train)
# save model
joblib.dump(model, "model/svm_4label_linear")

# Step5 : Print Accuracy
predictions = model.predict(x_test)
print("Model score/accuracy is", metrics.accuracy_score(y_test, predictions))
