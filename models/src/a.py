#%%
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

dir_path = r'C:/Users/Administrator/PycharmProjects/DrugUseageProject/results/daily_demand.csv'



try:
    # 1. 加载数据
    df = pd.read_csv(dir_path, index_col='RQ')  # 包含日期、患者数、用药量等字段
    data = pd.DataFrame(df.iloc[:, 1])
    data.columns = ['usage']

    data['date'] = pd.to_datetime(data.index)


    # %%
    # 2. 特征工程
    # 滞后特征及其处理
    # 计算用户使用数据的时间滚动特征
    data['prev_1_day'] = data['usage'].shift(1)  # 前一天的使用量
    data['prev_7_days_mean'] = data['usage'].rolling(window=7).mean().shift(1)  # 前七天的使用量平均值
    data['prev_30_days_mean'] = data['usage'].rolling(window=30).mean().shift(1)  # 前三十天的使用量平均值


    # 时间特征
    data['weekday'] = data['date'].dt.weekday  # 星期几
    data['is_holiday'] = data['weekday'].isin([5, 6]).astype(int)  # 假设周末为假日

    # 去除缺失值
    data = data.dropna(subset=['prev_1_day', 'prev_7_days_mean', 'prev_30_days_mean'])

    # 特征和目标
    features = ['prev_1_day', 'prev_7_days_mean', 'prev_30_days_mean',
                'weekday', 'is_holiday']
    target = 'usage'

    # 检查特征和目标是否存在
    X = data[features]
    y = data[target]

    # 3. 数据划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. 模型训练
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # 5. 评估模型
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {mae}")

    # 6. 预测未来一天需求量
    latest_data = pd.DataFrame({
        'prev_1_day': [128],  # 前1天用量
        'prev_7_days_mean': [130],  # 前7天平均用量
        'prev_30_days_mean': [135],  # 前30天平均用量
        'weekday': [2],  # 星期三
        'is_holiday': [0],  # 非假日
    })
    predicted_usage = model.predict(latest_data)
    print(f"Predicted Usage for Tomorrow: {predicted_usage[0]}")

except FileNotFoundError:
    print("错误: 文件未找到。请确认 'daily_demand.csv' 文件的路径是否正确。")
except KeyError as e:
    print(f"错误: 数据中缺少必要的列 {e}。请检查数据的完整性。")
except Exception as e:
    print(f"发生错误: {e}")
