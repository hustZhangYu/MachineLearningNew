if __package__:
    from .paths import training_workbook, feature_csv
else:
    from paths import training_workbook, feature_csv

# 这个代码用于数据的读取和测试
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def DataPlot():
    def DataPlotTest(data1,data2):
    # 画出曲线看看
        plt.plot(data1,data2)
    #  plt.show()

    # 读取数据

    z=['01','03','06','12','13','16','19']

    for z1 in z:
        data=pd.read_excel(training_workbook(z1))

    # 去掉第一行数据编号
        data=data.iloc[:,1:]
        DataPlotTest(data.iloc[:,0],data.iloc[:,1])
        print(data)

    plt.show()
    return 

def DealWithOutliers(c):
    data=pd.read_excel(training_workbook(c))
    data=data.iloc[:,1:]
    # print(data.describe())
    
    # data.boxplot(column='放电容量/Ah')    
    # plt.show()

    
    Q1=data['放电容量/Ah'].quantile(0.25)
    Q3=data['放电容量/Ah'].quantile(0.75)
    IQR=Q3-Q1
    lower_bound=Q1-1.5*IQR
    upper_bound=Q3+1.5*IQR

    ## 直接剔除异常值
    # plt.plot(data.iloc[:,0],data.iloc[:,1],'.',label='Original data')
    # data=data.loc[(data['放电容量/Ah']>=lower_bound) & (data['放电容量/Ah']<=upper_bound )]
    # plt.plot(data.iloc[:,0],data.iloc[:,1],label='New data')
    # plt.legend()
    # plt.show()

    # 将异常值使用合适的值替代
    # 直接剔除异常值可能不利于时间序列预测
    # 因此我们先将数据替代为NaN,然后通过插值来平滑过渡
    # 替换异常值为 NaN，再进行插值
    # plt.plot(data.iloc[:,0],data.iloc[:,1],'.',label='Original data')
    outliers = data[(data['放电容量/Ah'] <= lower_bound) | (data['放电容量/Ah'] >= upper_bound)]
    # 将异常值替换为 pd.NA
    data.loc[outliers.index, '放电容量/Ah'] = pd.NA
    data['放电容量/Ah'] = data['放电容量/Ah'].interpolate(method='linear', limit_direction='both')
    ## 关于画图
    # plt.plot(data.iloc[:,0],data.iloc[:,1],label='New data')
    # plt.legend()
    # plt.show()

    return data


def custom_minmax_scaler(data, new_min=0, new_max=1):
    data_min = np.min(data)
    data_max = np.max(data)
    
    if data_max == data_min:
        return np.full_like(data, new_min, dtype=float)

    scaled_data = new_min + (data - data_min) * (new_max - new_min) / (data_max - data_min)
    return scaled_data


def GeneratingInput1(window_size=5):
    # 处理数据和异常值
    data = DealWithOutliers('12')
    data1 = data['放电容量/Ah'].values.reshape(-1, 1)  # 将一维数据转成二维
        
    # 对整个 data 进行归一化
    new_min = 100
    new_max = 200
    data1_scaled = custom_minmax_scaler(data1, new_min, new_max)
    # print(data1_scaled)
    
    
    # plt.plot(data1_scaled)
    # 使用滑动窗口生成 x 和 y
    x = []
    y = []
    for i in range(len(data1_scaled) - window_size):
        x.append(data1_scaled[i:i + window_size].tolist())  # 每个窗口的数据作为输入
        y.append(data1_scaled[i + window_size])  # 窗口结束后的数据作为输出
    
    # 创建包含 x 和 y 的 DataFrame
    df = pd.DataFrame({'x': x, 'y': y})
    
    # print(df)
    return df



if __package__:
    from .dataset_io import ReadInput
else:
    from dataset_io import ReadInput


def GeneratingInput(window_size=5):
    # 处理数据和异常值
    m='12'
    data = DealWithOutliers(m)
    data_column = data['放电容量/Ah'].values.reshape(-1, 1)  # 将其转为二维数组

    # 创建 MinMaxScaler 实例，指定归一化范围为 [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))

    # 对数据进行归一化
    data_normalized = scaler.fit_transform(data_column)

    # 将归一化后的数据再存回到 DataFrame 中
    data['放电容量/Ah'] = data_normalized

    dataDetails=ReadInput(m)

    data=pd.concat([data,dataDetails['矢量']],axis=1)

    print(data)
    data['矢量'] = data.apply(lambda row: row['矢量'] + [row['放电容量/Ah']], axis=1)

    filename=feature_csv(m)

 
    data.to_csv(filename)
    # print(df)
    return data

if __name__=='__main__':
    # DealWithOutliers()
    # ReadInput()
    import os
    print(os.getcwd())
    GeneratingInput()
