"""
==================================================
Plot different SVM classifiers in the MNIST dataset
==================================================

Comparison of different linear SVM classifiers on a 2D projection of the MNIST
dataset. We only consider the first 2 features of this dataset:

- Sepal length
- Sepal width

This example shows how to plot the decision surface for four SVM classifiers
with different kernels.

The linear models ``LinearSVC()`` and ``SVC(kernel='linear')`` yield slightly
different decision boundaries. This can be a consequence of the following
differences:

- ``LinearSVC`` minimizes the squared hinge loss while ``SVC`` minimizes the
  regular hinge loss.

- ``LinearSVC`` uses the One-vs-All (also known as One-vs-Rest) multiclass
  reduction while ``SVC`` uses the One-vs-One multiclass reduction.

Both linear models have linear decision boundaries (intersecting hyperplanes)
while the non-linear kernel models (polynomial or Gaussian RBF) have more
flexible non-linear decision boundaries with shapes that depend on the kind of
kernel and its parameters.


"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

from sklearn.decomposition import PCA
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle

np.random.seed(0)
mnist = fetch_mldata("MNIST Original")
mytargets = list(range(0,10))
mytargets = [0, 9, 1]
XX_train, yy_train = mnist.data / 255., mnist.target
X_train=[]
y_train=[]
for i, label in enumerate(yy_train):
  if label in mytargets:
    X_train.append(XX_train[i])
    y_train.append(yy_train[i])
num_samples_to_plot = 1000
X_train, y_train = shuffle(X_train, y_train)
X_train, y_train = X_train[:num_samples_to_plot], y_train[:num_samples_to_plot]  # lets subsample a bit for a first impression

for digit in mytargets:
  instances=[i for i in y_train if i==digit]
  print "Digit",digit,"appears ",len(instances), "times"

  y =[ 2 if i==9 else i for i in y_train] #Map the labels to 0 to n-1


pca = PCA(n_components=2)
X = pca.fit_transform(X_train)

print X.shape

h = .02  # step size in the mesh

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(X, y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
lin_svc = svm.LinearSVC(C=C).fit(X, y)

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
titles = ['SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']


for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    #plt.subplot(3, 2, i + 1)
    #plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.figure(i+1,figsize=(4,3))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z,  alpha=0.8)
    plt.prism()

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y )
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title('MNIST digit classification using '+titles[i]+'(3 classes)')

plt.show()
