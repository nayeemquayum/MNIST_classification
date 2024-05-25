#import package
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly_express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#read data
sample_df=pd.read_csv('data/train.csv')
#print (sample_df.head())
#split data in train test
X_train,X_test,y_train,y_test = train_test_split(sample_df.drop(columns=['label']),\
                                                 sample_df['label'],test_size=0.2,random_state=40)
#this first step of PCA is to scale the values to standar normal distribution
scaler= StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)
#Visualize data in 3D
#Create PCA for 3
print('Creating 3D plot')
pca = PCA(n_components=3)
X_train_temp=pca.fit_transform(X_train)
X_test_temp=pca.transform(X_test)
fig= px.scatter_3d(data_frame=X_train,\
                   x=X_train_temp[:,0],
                   y=X_train_temp[:,1],
                   z=X_train_temp[:,2],
                   color=y_train)
fig.update_layout(margin=dict(l=20,r=20,b=20))
fig.show()
print('Finding PCA threshold')
#create the PCA object with all the features
pca = PCA(n_components=None) #create PCs equal to features
pca.fit_transform(X_train)
pca.transform(X_test)
#eigenvalue(lambda) for each PCs
#print(pca.explained_variance_)
#variance percentage captured by the PCs
#print(pca.explained_variance_ratio_ * 100)
#create a numpy array with the cumulative sum of variance percentage
cumsum_array=np.cumsum(pca.explained_variance_ratio_*100)
#calculate the index where cumsum cross 90%
threshold = np.where(cumsum_array >= 90)[0][0]
print(f"PCA threshold:{threshold}")
#plot a graph of the cumulative sum of variance percentage
plt.plot(cumsum_array)
plt.show()
#Now do PCA with threshold+1 components
pca = PCA(n_components=threshold+1) #create PCs equal to features
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
#now run the ML model on X_train and X_test
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
y_predict=knn.predict(X_test)
print(f"accuracy:{accuracy_score(y_test,y_predict)}")