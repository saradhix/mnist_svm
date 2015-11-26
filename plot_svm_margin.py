#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
SVM Margins Example
=========================================================
The plots below illustrate the effect the parameter `C` has
on the separation line. A large value of `C` basically tells
our model that we do not have that much faith in our data's
distribution, and will only consider points close to line
of separation.

A small value of `C` includes more/all the observations, allowing
the margins to be calculated using all the data in the area.

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle

# we create 40 separable points
np.random.seed(0)
mnist = fetch_mldata("MNIST Original")
mytargets = list(range(0,10))
mytargets = [0, 9]
XX_train, yy_train = mnist.data / 255., mnist.target
X_train=[]
y_train=[]
for i, label in enumerate(yy_train):
  if label in mytargets:
    X_train.append(XX_train[i])
    y_train.append(yy_train[i])
num_samples_to_plot = 500
X_train, y_train = shuffle(X_train, y_train)
X_train, y_train = X_train[:num_samples_to_plot], y_train[:num_samples_to_plot]  # lets subsample a bit for a first impression

for digit in mytargets:
  instances=[i for i in y_train if i==digit]
  print "Digit",digit,"appears ",len(instances), "times"

Y=y_train

pca = PCA(n_components=2)
X = pca.fit_transform(X_train)

print X.shape


# figure number
fignum = 1

# fit the model
for penalty in [1,0.1,0.01,0.001]:

    clf = svm.SVC(kernel='linear', C=penalty)
    clf.fit(X, Y)

    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    # plot the parallels to the separating hyperplane that pass through the
    # support vectors
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    yy_down = yy + a * margin
    yy_up = yy - a * margin

    # plot the line, the points, and the nearest vectors to the plane
    plt.figure(fignum, figsize=(4, 3))
    plt.clf()
    plt.prism()
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10)
    plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired)

    plt.axis('tight')
    x_min = -4.8
    x_max = 4.2
    y_min = -6
    y_max = 6

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.figure(fignum, figsize=(4, 3))
    plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xticks(())
    plt.yticks(())
    fignum = fignum + 1
    plt.title('C='+str(penalty))

plt.show()
