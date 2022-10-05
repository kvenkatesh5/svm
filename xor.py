'''
let's see how an SVM can solve the XOR problem
'''

import numpy as np
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions
from matplotlib import pyplot as plt

# Make dataset
X = [[0,1],[1,1],[1,0],[0,0]]
Y = [1,0,1,0]
for i in range(0,4):
    pt = X[i]
    lb = Y[i]
    for t in range(100):
        X.append([pt[0]+np.random.normal(loc=0, scale=0.1), pt[1]+np.random.normal(loc=0, scale=0.1)])
        Y.append(lb)

X = np.array(X)
y = np.array(Y)

# Train a SVM
linearModel = SVC(kernel='poly')
linearModel.fit(X,y)

plot_decision_regions(X, y, clf=linearModel, legend=2)
plt.show()