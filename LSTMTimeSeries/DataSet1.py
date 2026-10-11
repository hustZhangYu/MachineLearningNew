if __package__:
    from .paths import training_workbook, feature_csv
else:
    from paths import training_workbook, feature_csv

# 这个代码用于数据的读取和测试
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def rescale():
    # 对所有的数据进行收集并做归一化后，返回到各个文件当中
    # 针对每一个文件，读取相应的量
    # 合并之后，去掉异常点
    # 进行归一化
    c_values = ['01', '12', '13']  # 这里替换为实际编号
    file_data = {}
    param_values = []

    # 构建路径并读取文件
    for c in c_values:
        # 根据 c 构建文件路径
        filepath = training_workbook(c)
        
        # 读取 Excel 文件
        df = pd.read_excel(filepath)
        
        # 提取目标列 'param'，收集起来用于归一化
        param_values.extend(df['放电容量/Ah'].values)  # 将所有文件中的 param 数据收集起来
        file_data[filepath] = df  # 记录下每个文件的数据

    # 将所有的 param 数据合并到一起，并进行归一化
    scaler = MinMaxScaler()
    param_values = pd.DataFrame(param_values, columns=['放电容量/Ah'])  # 转为 DataFrame 格式
    scaler.fit(param_values)  # 基于所有文件的 param 进行归一化

    # 逐个文件应用归一化后的参数值，并保存到新列中
    for filepath, df in file_data.items():
        df['放电容量/Ah'] = scaler.transform(df[['放电容量/Ah']]) 
        # 将带有新列的 DataFrame 保存到 Excel 文件中，避免覆盖原始列
        new_filepath = filepath.replace('.xlsx', '_normalized.xlsx')  # 保存为新的文件
        df.to_excel(new_filepath, index=False)

    return


def DealWithOutliers(c):
    data=pd.read_excel(training_workbook(c, normalized=True))
    data=data.iloc[:,1:]
  
    Q1=data['放电容量/Ah'].quantile(0.25)
    Q3=data['放电容量/Ah'].quantile(0.75)
    IQR=Q3-Q1
    lower_bound=Q1-1.5*IQR
    upper_bound=Q3+1.5*IQR

    outliers = data[(data['放电容量/Ah'] <= lower_bound) | (data['放电容量/Ah'] >= upper_bound)]
    # 将异常值替换为 pd.NA
    data.loc[outliers.index, '放电容量/Ah'] = pd.NA
    data['放电容量/Ah'] = data['放电容量/Ah'].interpolate(method='linear', limit_direction='both')

    return data




if __package__:
    from .dataset_io import ReadInput
else:
    from dataset_io import ReadInput




def GeneratingInput(m):
    # 处理数据和异常值
    
    data = DealWithOutliers(m)
 
    # 创建 MinMaxScaler 实例，指定归一化范围为 [0, 1]
    data_column = data['放电容量/Ah'].values.reshape(-1, 1)
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

    data = data.iloc[:-3]
    data.to_csv(filename)
    # print(df)
    return data

if __name__=='__main__':
    # DealWithOutliers()
    # ReadInput()
    import os
    print(os.getcwd())
    rescale()
    GeneratingInput('13')
