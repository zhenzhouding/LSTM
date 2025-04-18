import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Input
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import random
from joblib import Parallel, delayed

# 设置随机种子
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 添加早停机制
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# 定义KGE损失函数
def kge_loss(y_true, y_pred):
    mean_obs = tf.reduce_mean(y_true)
    mean_sim = tf.reduce_mean(y_pred)
    
    std_obs = tf.math.reduce_std(y_true) + tf.keras.backend.epsilon()
    std_sim = tf.math.reduce_std(y_pred) + tf.keras.backend.epsilon()
    
    correlation = tf.reduce_sum((y_true - mean_obs) * (y_pred - mean_sim)) / (
        tf.sqrt(tf.reduce_sum(tf.square(y_true - mean_obs))) * tf.sqrt(tf.reduce_sum(tf.square(y_pred - mean_sim))) + tf.keras.backend.epsilon()
    )
    
    r = correlation
    alpha = std_sim / std_obs
    beta = mean_sim / mean_obs

    kge = 1 - tf.sqrt(tf.square(r - 1) + tf.square(alpha - 1) + tf.square(beta - 1))
    return 1 - kge

# 定义KGE计算函数（用于后期评估）
def cal_kge(sim, obs):
    mean_obs = np.mean(obs)
    mean_sim = np.mean(sim)
    r = np.corrcoef(sim, obs)[0, 1]
    alpha = mean_sim / mean_obs
    beta = np.std(sim) / np.std(obs)
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    return kge

# 定义模型训练函数
def train_model(units1, units2, batch_size, lr, x_train, y_train, x_val, y_val, x_test, y_test, save_path):
    # 设置 TensorFlow GPU 使用限制，避免 GPU 内存冲突（如果在 GPU 环境下运行）
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    # 构建模型
    model = Sequential()
    model.add(Input(shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Bidirectional(LSTM(units1, return_sequences=True)))
    model.add(Bidirectional(LSTM(units2, return_sequences=False)))
    model.add(Dense(1, activation='relu'))
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=kge_loss)
    
    # 训练模型
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=200, batch_size=batch_size,
                        verbose=0, callbacks=[early_stopping])
    
    # 保存预测结果
    model_name = f'lstm_{units1}_{units2}_batch_{batch_size}_lr_{lr}'
    train_pred = model.predict(x_train).flatten()
    val_pred = model.predict(x_val).flatten()
    test_pred = model.predict(x_test).flatten()
    
    train_result_path = os.path.join(save_path, f'train_result_{model_name}.csv')
    val_result_path = os.path.join(save_path, f'val_result_{model_name}.csv')
    test_result_path = os.path.join(save_path, f'test_result_{model_name}.csv')
    
    pd.DataFrame({'Train Prediction': train_pred}).to_csv(train_result_path, index=False)
    pd.DataFrame({'Validation Prediction': val_pred}).to_csv(val_result_path, index=False)
    pd.DataFrame({'Test Prediction': test_pred}).to_csv(test_result_path, index=False)

# 读取数据
input_path = '/data/gpfs/projects/punim2189/lstm/dpl_input.xlsx'
input_data = pd.read_excel(input_path)
input_data['time'] = pd.to_datetime(input_data['time'])

# 划分数据集
train_data = input_data[input_data['time'].dt.year <= 2014]
val_data = input_data[input_data['time'].dt.year.isin([2015, 2016])]
test_data = input_data[input_data['time'].dt.year >= 2017]

# 删除日期列并缩放数据
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data.drop('time', axis=1))
val_scaled = scaler.transform(val_data.drop('time', axis=1))
test_scaled = scaler.transform(test_data.drop('time', axis=1))

x_train, y_train = train_scaled[:, :-1], train_scaled[:, -1]
x_val, y_val = val_scaled[:, :-1], val_scaled[:, -1]
x_test, y_test = test_scaled[:, :-1], test_scaled[:, -1]

x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
x_val = x_val.reshape(x_val.shape[0], 1, x_val.shape[1])
x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

# 创建保存文件夹
save_path = '/data/gpfs/projects/punim2189/lstm/dpl_kge'
os.makedirs(save_path, exist_ok=True)

# 定义参数范围
units_list = list(range(8, 129, 8))
batch_size_list = list(range(5, 101, 5))
lr_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]

# 使用并行计算进行模型训练
Parallel(n_jobs=64)(
    delayed(train_model)(units1, units2, batch_size, lr, x_train, y_train, x_val, y_val, x_test, y_test, save_path)
    for units1 in units_list
    for units2 in units_list
    for batch_size in batch_size_list
    for lr in lr_list
)

# 计算并保存最终结果
results = []
train_obs = train_scaled[:, -1]
val_obs = val_scaled[:, -1]
test_obs = test_scaled[:, -1]

for units1 in units_list:
    for units2 in units_list:
        for batch_size in batch_size_list:
            for lr in lr_list:
                model_name = f'lstm_{units1}_{units2}_batch_{batch_size}_lr_{lr}'
                
                # 读取预测结果
                train_path = os.path.join(save_path, f'train_result_{model_name}.csv')
                val_path = os.path.join(save_path, f'val_result_{model_name}.csv')
                test_path = os.path.join(save_path, f'test_result_{model_name}.csv')
                
                train_pred = pd.read_csv(train_path)['Train Prediction'].values
                val_pred = pd.read_csv(val_path)['Validation Prediction'].values
                test_pred = pd.read_csv(test_path)['Test Prediction'].values
                
                # 计算KGE
                train_kge = cal_kge(train_pred, train_obs)
                val_kge = cal_kge(val_pred, val_obs)
                test_kge = cal_kge(test_pred, test_obs)
                
                # 保存结果
                results.append({
                    'units1': units1,
                    'units2': units2,
                    'batch_size': batch_size,
                    'lr': lr,
                    'train_kge': train_kge,
                    'val_kge': val_kge,
                    'test_kge': test_kge,
                })

# 保存结果到CSV
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join('/data/gpfs/projects/punim2189/lstm/kge_results_dpl_kge.csv'), index=False)
