import numpy as np
import matplotlib.pyplot as plt

'''初始化顾客源'''
#总仿真时间
Total_time=10
#队列长度
N=1e8
#到达率和服务率
Lambda=10
Mu=8
#平均到达时间与平均服务时间
arr_mean=1/Lambda
ser_mean=1/Mu
#整个仿真过程的数量，*2防止不够
arr_num=np.round(Total_time*Lambda*2).astype('int')
#仿真每个顾客到达的时间，这是指数分布
arr_moment=np.random.exponential(arr_mean,arr_num)
arr_moment=np.cumsum(arr_moment)
#仿真每个顾客的服务时间
ser_time=np.random.exponential(ser_mean,arr_num)
#仿真的队伍列表，排过队顾客的编号将会在这里记录
queue=[]
#等待时间
wait_time=[]
#离开的时刻，假设一个虚拟的人，离开时间是-1，最后会去掉
leave_moment=[-1]
for i in range(arr_num):
    #说明仿真时间到了
    if arr_moment[i]>Total_time:
        break
    #当前队列长度（不算已经服务的）
    current_q_len=np.sum(list(map(lambda x,y:x>y,leave_moment,[arr_moment[i]]*len(leave_moment))))-1
    current_q_len=0 if current_q_len<0 else current_q_len
    #队列长度过长，直接拒绝服务，相当于顾客看到那么长直接不排了
    if current_q_len>N:
        continue
    #剩下的都会服务
    #现在没人排队
    if len(queue)==0:
        wait_time.append(0)
        leave_moment.append(arr_moment[i]+ser_time[i])
        queue.append(i)
    else:
        wait_time.append(0 if leave_moment[queue[-1]+1]<arr_moment[i] else leave_moment[queue[-1]+1]-arr_moment[i])
        leave_moment.append(arr_moment[i]+wait_time[-1]+ser_time[i])
        queue.append(i)
#去掉多余的-1
leave_moment.pop(0)
fig,ax=plt.subplots(figsize=(10,15))
ax.step(arr_moment[:len(queue)],queue,where='post',color='r',label='arrived time')
ax.step(leave_moment,queue,where='post',color='b',label='leave time')
plt.xlabel('simulation time')
plt.ylabel('index of customer')
plt.legend()
plt.grid()
plt.show()
