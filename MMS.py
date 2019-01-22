import numpy as np

'''
排队论 M/M/S模型多服务台单队列
'''

s=2
Lambda=3
Mu=4
Rho=Lambda/Mu
Rho_s=Rho/s
#系统中人数为0的概率
P0=1/(np.sum([Rho**x/np.math.factorial(x) for x in range(s)])+Rho**s/(np.math.factorial(s)*(1-Rho_s)))
Lq=P0*Rho**s*Rho_s/(np.math.factorial(s)*(1-Rho_s)**2)
Ls=Lq+Rho
Ws=Ls/Lambda
Wq=Lq/Lambda

def P(n):
    return P0*Rho**n/np.math.factorial(n) if n<s else Rho**n*P0/(np.math.factorial(s)*s**(n-s))

#>=k
def P_greater(k):
    return Rho**k*P0/(np.math.factorial(k)*(1-Rho_s))

print('Ls:',Ls)
print('Lq:',Lq)
print('Ws:',Ws)
print('Wq:',Wq)
print('P0:',P0)
