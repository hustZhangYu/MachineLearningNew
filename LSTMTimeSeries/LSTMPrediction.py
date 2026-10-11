if __package__:
    from .models import LSTMModel
else:
    from models import LSTMModel

if __package__:
    from .io_helpers import str_to_list
else:
    from io_helpers import str_to_list

if __package__:
    from .windowing import create_sliding_window as _create_sliding_window
else:
    from windowing import create_sliding_window as _create_sliding_window

if __package__:
    from .paths import feature_csv
else:
    from paths import feature_csv

#使用LSTM进行

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd 


# 定义LSTM的模型类


# 实例化模型，创建一个模型，给定输入层大小，隐藏层大小和输出层大小
if __name__ == "__main__":
    input_size = 6
    hidden_size = 30
    output_size = 1


    model = LSTMModel(input_size, hidden_size, output_size)  # 确保使用与保存时相同的参数
    # 加载模型的状态字典
    model.load_state_dict(torch.load('rnn_model.pth'))
    model.eval()


    # 数据的读取

    data = pd.read_csv(feature_csv('01'), converters={'矢量': str_to_list})


    # 第一列作为输入，第二列作为目标
    x = torch.tensor(data['矢量'], dtype=torch.float32)  # 第一列
    y = torch.tensor(data['放电容量/Ah'], dtype=torch.float32)  # 第二列



    # 创建滑动窗口的输入和目标
    def create_sliding_window(data_x, data_y, window_size):
        return _create_sliding_window(data_x, data_y, window_size, reserved_tail=0)

    # 设置窗口大小
    window_size = 5
    x_windows, y_windows = create_sliding_window(x, y, window_size)


    print(x_windows)

    # Use the last sliding window as input for prediction

    # 将第一个滑动窗口作为初始测试输入
    x_test = x_windows[0].unsqueeze(0)  # 扩展维度以适应模型输入


    # 定义要生成的预测步数
    prediction_steps = 300  # 可根据需要调整预测步数
    predictions = []



    with torch.no_grad():
        for step in range(prediction_steps):
            # 模型预测
            pred = model(x_test).squeeze()  # 获取当前预测结果
            predictions.append(pred.item())  # 保存预测结果

            # 获取已知的输入量 (x)，并将其转换为张量
            x1 = x[window_size + step].clone()  # 克隆当前已知量以防止影响原始数据
            x1[-1] = pred.item()  # 替换最后一个特征为新的预测结果

            # 更新窗口：前移并替换最后一个元素
            # 克隆 x_test 以避免内存冲突
            updated_x_test = x_test.clone()  
            updated_x_test[:, :-1, :] = x_test[:, 1:, :]  # 前移窗口，去掉第一个时间步
            updated_x_test[:, -1, :] = x1.unsqueeze(0)  # 将新的已知输入放入窗口的最后一个位置

            # 更新 x_test 为新的张量
            x_test = updated_x_test

            # 进行调试输出
            print('x_test =', x_test)
            x1_test = x_windows[step+1].unsqueeze(0)
            print('x1_test =', x1_test)


    # 将预测结果和真实值转为numpy数组以便绘图
    predictions = np.array(predictions)

    # 获取原始数据
    original_y = y.numpy()

    # 绘图
    plt.figure(figsize=(12, 6))

    # 绘制原始数据
    plt.plot(original_y, label='Original Data', color='gray', alpha=0.5)

    # 绘制预测数据
    plt.plot(range(4, 4+ prediction_steps), predictions, label='Predicted Values', marker='x', color='red')

    # 设置标题和标签
    plt.title('Continuous Prediction Using Sliding Window with Known Inputs')
    plt.xlabel('Sample Index')
    plt.ylabel('Discharge Capacity / Ah')
    plt.legend()
    plt.grid(True)
    plt.show()
