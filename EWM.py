import warnings
import numpy as np
warnings.filterwarnings('ignore')

#假设全是正向指标
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data=np.array([[80,90],[70,85],[65,88]]) #n_samples,n_features
n=data.shape[0]
data=scaler.fit_transform(data)
pij= data / data.sum(axis=0) #广播机制
K=1/np.log(n)
temp= pij * np.nan_to_num(np.log(pij))
eij=-K*temp.sum(axis=0)
d=1-eij
weight=d/d.sum()
print(weight)

