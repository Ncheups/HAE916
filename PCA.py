import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from numpy.linalg import eig
import seaborn as sb
from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay

############################################# PCA #####################
def PCA(x , dimension):

    # Calculation of the mean
    data_mean = np.mean(data, axis = 0)
    # Calculation of the Shift matrix
    data_shift = data - data_mean
    # Covariance matrix
    data_covariance = np.cov(data_shift, rowvar = False)
    # Calculation of the eigenvalues
    values_eigen, vectors_eigen = np.linalg.eigh(data_covariance)

    # index
    sorted_index = np.argsort(values_eigen)[::-1]
    # eigen_values
    eigenvalue = values_eigen[sorted_index]
    # eigenvectors
    eigenvectors = vectors_eigen[:,sorted_index]

    # Projection of the new dimension model
    eigenvectors_reduce = eigenvectors[:,0:dimension]
    data_newdimension = np.dot(eigenvectors_reduce.transpose(),data_shift.transpose()).transpose()

    return data_newdimension

############################################# SVM #####################

def SVM(x,y):
    clf = svm.SVC(kernel="linear", C=1e10)
    clf.fit(X, y)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

    # plot the decision function
    ax = plt.gca()
    DecisionBoundaryDisplay.from_estimator(clf,X,plot_method="contour",colors="k",levels=[-1, 0, 1],alpha=0.5,linestyles=["-", "-", "-"],ax=ax,)
    # plot support vectors
    ax.scatter(clf.support_vectors_[:, 0],clf.support_vectors_[:, 1],s=50,linewidth=1,facecolors="none",edgecolors="k",)
    plt.show()


############################################# main #####################

x = pd.read_csv("C:/Users/nolha/Desktop/HAE916/flowerTrain.data", names=['sepal length','sepal width','petal length','petal width','species'])
data = x.iloc[:,0:4]
plt.scatter(x.iloc[:,0],x.iloc[:,1],alpha =0.2,s =100*x.iloc[:,2], c=100*x.iloc[:,3])

#x data on 4D

data_newdimension = PCA(x, 2)

#data_newdimension data projected on 2D

dataset = x.iloc[:,4]
data_frame = pd.DataFrame(data_newdimension , columns = ['Component1','Component2'])
data_framee = pd.concat([data_frame , pd.DataFrame(dataset)] , axis = 1)

plt.figure(figsize = (6,6))
sb.scatterplot(data = data_framee , x = 'Component1',y = 'Component2', hue = 'species' , s = 60 , palette= 'icefire')

#plt.figure(figsize = (6,6))

X = data_newdimension
y = [0 for i in range(len(dataset))]

for i in range(len(dataset)):
    if(dataset[i] == 'convoluta'):
        y[i] = 0
    else:
        if(dataset[i] == 'versicolor'):
            y[i] = 0
        else:
            if(dataset[i] == 'interior'):
                y[i] = -1

SVM(X,y)