
#Loading dataset from scikit-learn dataset
from sklearn.datasets import load_iris
iris = load_iris()

iris.keys()

from sklearn import datasets as ds
from matplotlib import pyplot as pl
images = ds.load_sample_images()
pl.imshow(images.images[0])

import sklearn.datasets as ds
data = ds.fetch
data.keys()

Data = [
    {'Price':710000,'Rooms':2,'Neighbourhood':'Cuffe Parade'},
    {'Price':740000,'Rooms':1,'Neighbourhood':'Coloba'},
    {'Price':730000,'Rooms':7,'Neighbourhood':'Versova'},
    {'Price':790000,'Rooms':3,'Neighbourhood':'Andheri'},
]

from sklearn.feature_extraction import DictVectorizer
vec = DicVectorizer(sparse=False,dtype=int)
vec.fit_transform(data)

import numpy as np
x=np.array([[1,-1,2],
    [2,0,0],
    [0,-1,-1]])
    
from sklearn import preprocessing
x_scale = preprocessing.scale(X)
x_scale.mean(axis=0)

from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=0.2)
sel.fit_transform(x)

