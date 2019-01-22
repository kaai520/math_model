import numpy as np

'''
排队论
基本参数：
Ls队长：系统中顾客数
Lq排队长：系统中排队的人数
Ws逗留时间：一个顾客在系统中的全部停留时间期望值
Wq等待时间：一个顾客在排队的时间的期望值
Rho服务强度：Rho=Lambda/(s*Mu)
s服务台数
Lambda单位时间内到达系统的人数
Mu单位时间内系统服务人数
s服务台数量
M/M/1模型：单队单服务台
'''

#一般来说Lambda<Mu
Lambda=3
Mu=4
Rho=Lambda/Mu
#系统中大于k人数的概率
def P_greater(k):
    return Rho**(k+1)
#系统中存在人数的概率
def P(n):
    return Rho**n*(1-Rho)

Ls=Rho/(1-Rho)
Lq=Ls*Rho
Ws=1/(Mu-Lambda)
Wq=Ws*Rho
pg=P_greater(1)
p=P(1)
print('Ls:',Ls)
print('Lq:',Lq)
print('Ws:',Ws)
print('Wq:',Wq)
print(pg)
print(p)
