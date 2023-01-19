import sklearn
import pandas as pd

column_names = ['diagnosis', 'radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness','concavity','concave points','symmetry','fractal dimension']

wc = pd.read_csv('data/wdbc.data', names = column_names, usecols = [1]+[*range(22,32)]) #We are only interested in the second column (the diagnosis) and the last 10 columns, which are the real valued features 

wc.head()

import numpy as np

X = wc.iloc[:, -10:].values # training data is the last ten elements
y_raw = wc.iloc[:, 0].values
y = [1 if i == "M" else 0 for i in y_raw]
#print(y)

radiusRaw = wc.iloc[:, 1].values
radiusNp = np.array(radiusRaw)