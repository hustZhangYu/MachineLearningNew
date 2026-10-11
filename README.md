# Learn-ML
People learn Machine learning. 

## ML to perdict the time evolution
First, we use the time series to predict the time evolution of the system.
Why people always study the time evolution of the open systems rather than isolated system?

## ML in disorder system

## 时间序列预测

非常幸运能有一个机会来上手实操一个机器学习问题。
简单描述一下题目： 结合数据及系统运行数据中工步号、工步状态、运行时间等各变量与电池放电容量之间的动态变化，同时也可以使用温度、气压等外部数据，构建电池放电容量预测算法模型，预测一段时间内电池放电容量随循环号变化的趋势。

## Environment and external data

Install `requirements.txt` in an isolated Python environment. It lists
direct source dependencies and the Excel reader; original package versions
were not recorded, so it is not a reproduction lock file. The three derived
CSV feature files already tracked in `LSTMTimeSeries/` remain usable by
default. Raw and normalized battery workbooks are external prerequisites.

Set `ML_TIMESERIES_DATA` to your dataset directory, or place the same data
in `LSTMTimeSeries/`. Workbooks use `train/M001.xlsx`, `train/M012.xlsx`
etc.; normalized workbooks have the `_normalized.xlsx` suffix. Derived
features use `NewDataM001.csv`, `NewDataM012.csv` and `NewDataM013.csv`.
`paths.py` defines these locations consistently on Windows and Linux.
The preprocessing formulas and workbook identifiers remain unchanged.

## Experiment modules

The original source notes identify `DataSet1.py` as preprocessing,
`LSTMtrain2.py` as model training, and `LSTMReadAndTest.py` as evaluation.
After configuring the data, run a selected experiment explicitly, e.g.
`python LSTMTimeSeries/LSTMtrain2.py`. Importing a training or inference
module no longer reads files, starts an experiment or overwrites a
checkpoint. Third-party dependencies are still needed to import those
modules. The `models.py` LSTM retains the six scripts' original layers,
constructor defaults and `state_dict` keys.
Both RNN classes now initialize hidden state using their own layer count
and width and the input tensor's dtype/device, so they can be instantiated
after import without a training script's global configuration.

`windowing.py` preserves the original next-step target and ordering.
The four training wrappers still reserve 200 final rows; the other
wrappers still reserve none. No split policy, loss, optimizer, epoch
count or random-state setting is changed. Existing scripts reuse
`rnn_model.pth` for different architectures, so keep separate runs and
verify architecture/parameters when loading an existing checkpoint.

Run bounded checks with `python -m unittest discover -s tests`. The helper
checks use synthetic arrays only and do not train a network or evaluate
prediction accuracy. The legacy random split of overlapping windows and
whole-series normalization do not establish a chronological holdout;
changing those evaluation conventions needs a separate experiment.
