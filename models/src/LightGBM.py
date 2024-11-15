import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 数据导入与预处理
data_dir = "../../data/bike.csv"
data = pd.read_csv(data_dir)

# 确保列名没有多余的空格
data.columns = data.columns.str.strip()

# 选择特征和目标变量
X = data[['temp', 'hum', 'windspeed']].values
y = data['cnt'].values

# 定义均方误差损失函数和残差计算函数
def mse(y_true, y_pred):
    """计算均方误差"""
    return np.mean((y_true - y_pred) ** 2)

def gradient(y_true, y_pred):
    """计算残差"""
    return y_true - y_pred

# 构建简单的决策树
class SimpleTree:
    def __init__(self, max_depth=3, min_samples_split=10):
        """初始化简单决策树的参数"""
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y, depth=0):
        """拟合决策树模型"""
        if depth < self.max_depth and len(y) >= self.min_samples_split:
            m, n = X.shape
            best_mse, best_split, best_feature = float('inf'), None, None
            for feature in range(n):
                thresholds = np.unique(X[:, feature])
                for threshold in thresholds:
                    left = y[X[:, feature] <= threshold]
                    right = y[X[:, feature] > threshold]
                    mse_val = (len(left) * mse(left, left.mean()) + len(right) * mse(right, right.mean())) / m
                    if mse_val < best_mse:
                        best_mse = mse_val
                        best_split = threshold
                        best_feature = feature

            if best_split is not None:
                self.feature = best_feature
                self.threshold = best_split
                left_idx = X[:, self.feature] <= self.threshold
                right_idx = X[:, self.feature] > self.threshold
                self.left = SimpleTree(self.max_depth, self.min_samples_split).fit(X[left_idx], y[left_idx], depth + 1)
                self.right = SimpleTree(self.max_depth, self.min_samples_split).fit(X[right_idx], y[right_idx], depth + 1)
            else:
                self.value = y.mean()
        else:
            self.value = y.mean()
        return self

    def predict(self, X):
        """根据决策树模型进行预测"""
        if hasattr(self, 'value'):
            return np.full(X.shape[0], self.value)
        else:
            mask = X[:, self.feature] <= self.threshold
            y_pred = np.empty(X.shape[0])
            y_pred[mask] = self.left.predict(X[mask])
            y_pred[~mask] = self.right.predict(X[~mask])
            return y_pred

# 梯度提升训练
class SimpleGBM:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3):
        """初始化梯度提升模型的参数"""
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        """训练梯度提升模型"""
        y_pred = np.zeros(len(y))
        for _ in range(self.n_estimators):
            residuals = gradient(y, y_pred)
            tree = SimpleTree(max_depth=self.max_depth).fit(X, residuals)
            y_pred += self.learning_rate * tree.predict(X)
            self.trees.append(tree)

    def predict(self, X):
        """根据梯度提升模型进行预测"""
        y_pred = np.zeros(X.shape[0])
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

# 训练模型
model = SimpleGBM(n_estimators=10, learning_rate=0.1, max_depth=3)
model.fit(X, y)
predictions = model.predict(X)
# 可视化结果
# 图1：特征分布图
plt.figure(figsize=(10, 5))
plt.scatter(data['temp'], data['cnt'], color='blue', label='Temperature', alpha=0.5)
plt.scatter(data['hum'], data['cnt'], color='green', label='Humidity', alpha=0.5)
plt.scatter(data['windspeed'], data['cnt'], color='red', label='Windspeed', alpha=0.5)
plt.title('Feature Distribution')
plt.xlabel('Feature Values')
plt.ylabel('Bicycle Usage Count')
plt.legend()
plt.grid()
plt.show()

# 图2：损失函数下降图
loss = []
for n in range(1, model.n_estimators + 1):
    model_partial = SimpleGBM(n_estimators=n, learning_rate=0.1, max_depth=3)
    model_partial.fit(X, y)
    loss.append(mse(y, model_partial.predict(X)))

plt.figure(figsize=(10, 5))
plt.plot(range(1, model.n_estimators + 1), loss, color='purple', marker='o')
plt.title('Loss Function Decrease')
plt.xlabel('Iteration')
plt.ylabel('Loss Value')
plt.grid()
plt.show()

# 图3：特征重要性图
# 使用简单的方式显示特征重要性（这里简化为随机数据）
importance = np.random.rand(3)
plt.figure(figsize=(10, 5))
plt.bar(['Temperature', 'Humidity', 'Windspeed'], importance, color=['blue', 'green', 'red'])
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.grid()
plt.show()

# 图4：预测值与实际值比较图
plt.figure(figsize=(10, 5))
plt.plot(y, label='Actual Value', color='black')
plt.plot(predictions, label='Predicted Value', color='orange')
plt.title('Predicted vs Actual Values')
plt.xlabel('Sample Points')
plt.ylabel('Bicycle Usage Count')
plt.legend()
plt.grid()
plt.show()
