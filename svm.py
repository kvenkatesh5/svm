'''
SVM Implementation using PyTorch
'''

# Imports
from pickletools import optimize
from termios import TAB2
import torch as T
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Generate data
def get_data():
    '''
    Generate a (somewhat random) simple 2D scatter dataset
    X^(i) in R^2
    Y^(i) in {-1,1}
    '''
    X = np.random.uniform(0,10, size=(1000,2))
    Y = np.apply_along_axis(lambda x: 1 if x[1]>x[0] else -1, axis=1, arr=X)
    Y = Y.reshape(-1,1)

    N,D = X.shape
    X = T.from_numpy(X).float()
    Y = T.from_numpy(Y).float()

    # Plotting
    # X_pos = X[(Y==1).flatten(),:]
    # X_neg = X[(Y==-1).flatten(),:]
    # plt.scatter([v[0] for v in X_pos], [v[1] for v in X_pos])
    # plt.scatter([v[0] for v in X_neg], [v[1] for v in X_neg])
    # plt.show()

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25,random_state=1)
    return X_train, X_test, Y_train, Y_test, N, D

# Main
def train(X_train, X_test, Y_train, Y_test, N, D, C):
    print(X_train)
    model = T.nn.Linear(in_features=D,out_features=1,bias=True)
    n_epochs = 100
    learning_rate = 1e-4
    weight_decay = 1e-4
    optim = T.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_history = []
    for epoch in range(n_epochs):
        Y_train_hat = model(X_train)
        # Soft-margin loss
        loss = T.mean(T.clamp(C - Y_train_hat * Y_train, min=0))
        optim.zero_grad()
        loss.backward()
        optim.step()
        loss_history.append(loss.item())
    print(model.weight)


if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test, N, D = get_data()
    train(X_train, X_test, Y_train, Y_test, N, D, C=1)
    train(X_train, X_test, Y_train, Y_test, N, D, C=10)