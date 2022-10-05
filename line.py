'''
let's see how an SVM can solve the line problem where the labels alternate
'''

import numpy as np
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions
from matplotlib import pyplot as plt

# Make dataset
p = 100
X = []
Y = []
for i in range(p):
    X.append([i,0 + np.random.normal(0,0.01)])
    Y.append(i%2)

X = np.array(X)
y = np.array(Y)

# Train a SVM
model = SVC(kernel='rbf')
model.fit(X,y)

plot_decision_regions(X, y, clf=model, legend=2)
plt.show()