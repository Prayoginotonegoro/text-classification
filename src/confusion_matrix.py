print(__doc__)

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# import some data to play with

# Split the data into a training set and a test set

# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = np.array([[4063,13,21,5,30,1,9,6],[6,4453,6,14,3,4,3,10],[22,13,3687,10,18,5,19,4],[7,27,1,3851,11,158,4,163],[31,9,23,11,3941,4,18,1],[0,7,0,179,3,3673,1,59],[23,6,9,10,17,1,4196,21],[2,11,0,103,0,41,21,3932]])

#cnf_matrix = np.array([[401,7,9,2,5,1,6,1],[1,443,1,4,0,1,0,1],[3,3,375,7,6,1,7,3],[2,5,1,349,1,24,1,30]
#,[4,5,5,3,378,2,2,1],[1,1,0,15,1,353,1,13],[9,9,6,3,1,2,395,7],[0,2,2,23,0,9,0,346]])

np.set_printoptions(precision=2)
class_names = ["hockey",'movies','nba','news','nfl','politics','soccer','worldnews']
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                      title='Normalized confusion matrix')
plt.show()
