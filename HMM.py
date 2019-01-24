import numpy as np


#序号都是从0开始
#观测序列
train_O=np.array([0,1,2],dtype='int')
#隐状态的转移概率矩阵
train_A=np.array([[0.7,0.3],
                  [0.4,0.6]])
#发射概率矩阵
train_B=np.array([[0.5,0.4,0.1],
                  [0.1,0.3,0.6]])

train_Pi=np.array([0.6,0.4])

def forward(A,B,Pi,O):
    m=A.shape[0]
    n=len(O)
    alpha=np.empty((n,m))
    alpha[0]=Pi*B[:,O[0]]
    for i in range(1,n):
        alpha[i]=np.dot(alpha[i-1],A)*B[:,O[i]]
    p=alpha[-1].sum()
    return alpha,p

def backward(A, B, Pi, O):
    m = A.shape[0]
    n = len(O)
    beta = np.empty((n, m))
    beta[-1] = 1
    for i in range(n-2,-1,-1):
        beta[i] = np.dot(beta[i+1, :] * B[:, O[i+1]], A.T)
    p=(Pi*B[:,O[0]]*beta[0]).sum()
    return beta,p

def viterbi(A,B,Pi,O):
    m=A.shape[0]
    n=len(O)
    Delta=np.empty((n,m))
    Psi=np.empty((n,m))
    Delta[0]=Pi*B[:,O[0]]
    Psi[0]=-1
    for i in range(1,n):
        temp=Delta[i-1,:]*A.T
        Delta[i]=np.max(temp,axis=1)*B[:,O[i]]
        Psi[i]=np.argmax(temp,axis=1)
    p_best=np.max(Delta[-1])
    best_path=np.empty(n,dtype='int')
    best_path[-1]=np.argmax(Delta[-1])
    for i in range(n-2,-1,-1):
        best_path[i]=Psi[i+1,best_path[i+1]]
    print('最优路径概率：',p_best)
    print('最优路径',best_path)

#Gamma是每个时刻单个隐状态的概率矩阵
def get_Gamma(A,B,Pi,O):
    alpha,p1=forward(A,B,Pi,O)
    beta,p2=backward(A,B,Pi,O)
    Gamma=alpha*beta
    Gamma=Gamma/Gamma.sum(axis=1).reshape((-1,1))
    return Gamma

#Xi是每个时刻到下一个时刻的隐状态转化的概率张量
def get_Xi(A,B,Pi,O):
    m=A.shape[0]
    n=len(O)
    Xi=np.empty((n-1,m,m))
    alpha, p1 = forward(A, B, Pi, O)
    beta, p2 = backward(A, B, Pi, O)
    for i in range(n-1):
        Xi[i]=np.dot(alpha[i].T.reshape((-1,1)),beta[i+1].reshape((1,-1)))*B[:,O[i+1]]*A
        Xi[i]/=Xi[i].sum()
    return Xi

def BaumWelch(A,B,Pi,O):
    Xi=get_Xi(A,B,Pi,O)
    Gamma=get_Gamma(A,B,Pi,O)
    n=len(O)
    pi=Gamma[0]/Gamma[0].sum()
    Gamma_1=Gamma[:-1,:]
    a=Xi.sum(axis=0)/Gamma_1.sum(axis=0).reshape((-1,1))
    temp=list(map(lambda x:np.where(O==x)[0],list(range(n))))
    b=np.empty(B.shape)
    for i in range(n):
        b[:,i]=Gamma[temp[i]].sum(axis=0)
    b/=Gamma.sum(axis=0).reshape((-1,1))
    return a,b,pi

def BaumWelch_n(A,B,Pi,O,epochs):
    for _ in range(epochs):
        A, B, Pi=BaumWelch(A,B,Pi,O)
    return A,B,Pi


