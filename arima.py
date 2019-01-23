import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import warnings
warnings.filterwarnings("ignore")

#arima
#构造时间序列
np.random.seed(2019)
df_obj=pd.DataFrame(np.random.randn(1000,1),index=pd.date_range('20170101',periods=1000),columns=['data'])
df_obj['data']=df_obj['data'].cumsum()
df_train=df_obj['20170101':'20180101']
# diff1=df_train.diff(1).dropna()
# acf_diff=plot_acf(diff1,lags=20)
# pacf_diff=plot_pacf(diff1,lags=20)
# plt.show()
model=ARIMA(df_train,order=(1,1,1),freq='D')
arima_result=model.fit()
print(arima_result.summary())
result=arima_result.predict('20180102','20180201',dynamic=True,typ='levels')
data_forcast = pd.concat([df_obj,result],axis=1,keys=['original', 'predicted'])
print(data_forcast.head())
plt.plot(data_forcast)
plt.show()
