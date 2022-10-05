'''
let's see how an SVM can solve the circle problem
'''

import numpy as np
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions
from matplotlib import pyplot as plt

# Make dataset
angle = np.linspace(0,180,100)
rad1 = 5
rad2 = 3
X = []
Y = []
for a in angle:
    X.append([np.cos(a) * rad1 + np.random.normal(0,0.1), np.sin(a) * rad1 + np.random.normal(0,0.1)])
    Y.append(1)
    X.append([np.cos(a) * rad2 + np.random.normal(0,0.1), np.sin(a) * rad2 + np.random.normal(0,0.3)])
    Y.append(-1)

X = np.array(X)
y = np.array(Y)


# Train a SVM
linearModel = SVC(kernel='poly')
linearModel.fit(X,y)

plot_decision_regions(X, y, clf=linearModel, legend=2)
plt.show()