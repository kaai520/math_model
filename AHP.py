import warnings
import numpy as np
warnings.filterwarnings('ignore')


A=np.array([[1,1/3,1/2],[3,1,2],[2,1/2,1]])
n=A.shape[0]
eigenvalue,eigenvector=np.linalg.eig(A)
max_value=eigenvalue[0].astype('float')
max_vector=eigenvector[:,0].astype('float')
weight=max_vector/max_vector.sum()
CI=(max_value-n)/(n-1)
RI=[0,0,0.58,0.90,1.12,1.24,1.32,1.41,1.45,1.49,1.52,1.54,1.56,1.58,1.59]
CR=CI/RI[n-1] if n>=3 else 0
print('CI(一致性指标)：{}'.format(CI))
print('CR(一致性比例)：{}'.format(CR))
print('特征值：{}'.format(max_value))
print('特征向量：{}'.format(max_vector))
print('权向量：{}'.format(weight))
if CR<0.1:
    print('通过一致性检验！')
else:
    print('没有通过一致性检验！')

