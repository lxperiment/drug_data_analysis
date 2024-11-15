
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib
# 指定默认字体
matplotlib.rcParams['font.family'] = 'SimHei'  # 或其他支持中文的字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的

#%%
# 计算指定列中累计达到目标值的条目
def taget_count(data, taget=1.0, column=0, type='s'):
    """
    计算在指定列中累计达到目标值的条目

    参数:
    data : DataFrame - 输入数据
    taget : float - 目标值，默认为1.0
    column : int - 要检查的列索引，默认为0
    type : str - 返回的数据类型，'s'表示Series，'d'表示DataFrame，默认为'Series'

    返回:
    Series - 达到目标的条目列表；如果没有达到，则返回空DataFrame；发生错误时返回None
    DataFrame - 达到目标的条目列表和对应的百分比；如果没有达到，则返回空DataFrame；发生错误时返回None
    """
    try:
        # 确保指定的列存在于数据中
        if column < 0 or column >= data.shape[1]:
            raise ValueError("指定的列索引超出范围。")

        # 提取并排序数据
        df_sorted = data.iloc[:, column].sort_values(ascending=False)
        total = df_sorted.sum()

        if total == 0:
            raise ValueError("列中的总和为0，无法进行计算。")

        list_of_drug = []
        total_use = 0

        # 根据类型的不同，初始化对应的列表
        list_of_byte = [] if type == 'd' else None

        # 遍历排序后的数据，计算达到目标值的条目
        for index, value in df_sorted.items():
            total_use += value
            byte_of_use = total_use / total
            list_of_drug.append(index)

            if type == 'd':
                list_of_byte.append(byte_of_use)

            # 检查是否达到了目标值
            if byte_of_use >= taget:
                return pd.Series(list_of_drug) if type == 's' else pd.DataFrame({'DRUG_NAME': list_of_drug, 'Byte': list_of_byte})

        return pd.DataFrame(columns=["Values"])  # 如果没有满足条件的返回空DataFrame

    except ValueError as ve:
        print(f"发生错误: {ve}")
        return None  # 返回None表示执行失败
    except Exception as e:
        print(f"发生未知错误: {e}")
        return None  # 返回None表示执行失败

#%%
# 计算一维数据的第一和第二导数
def com_diff(data):
    """
    计算一维数据的第一和第二导数

    参数:
    data : array-like - 输入数据，长度必须大于1

    返回:
    tuple - 第一导数、变化的x值、变化的y值和第二导数；如有错误返回None
    """
    try:
        # 检查数据的有效性
        if len(data) < 2:
            raise ValueError("输入数据长度必须大于1。")

        x = np.linspace(0, 1, len(data))
        y = np.array(data)

        # 计算第一导数
        dy_dx = np.diff(y) / np.diff(x)

        # 计算第二导数，只有在第一导数有足够数据时才计算
        if len(dy_dx) > 1:
            d2y_dx2 = np.diff(dy_dx) / np.diff(x[:-1])
        else:
            d2y_dx2 = np.array([])  # 如果没有足够数据，返回空数组

        x_chg = x[1:-1]
        y_chg = y[1:-1]

        return x_chg, y_chg, dy_dx, d2y_dx2

    except Exception as e:
        print(f"发生错误: {e}")
        return None  # 返回None表示执行失败


#%%
# 指数函数模型
def exp_func(x, a, b):
    return a * np.exp(b * x) + 1


def tangent_line(x, a, b):
    return a * x + b


#%%
def evaluattion_of_function(type='slope', slope=1, points=((0, 0))):
    """
    计算指定点的切线斜率或截距

    参数:
    type : str - 计算类型，'slope'表示计算斜率，'point'表示计算指定点的斜率，默认为'slope'
    slope : float - 斜率值，默认为1
    points : tuple - 指定点的坐标，默认为(0, 0)

    返回:
    function - 计算结果的函数；发生错误时返回None
    """
    try:
        if type not in ['slope', 'point']:
            raise ValueError("不支持的计算类型。")

        point = iter(points)
        x1, y1 = next(point)

        if type == 'slope':
            intercept = y1 - slope * x1
            return lambda x: slope * x + intercept
        elif type == 'point':
            x2, y2 = next(point)
            if x2 == x1:
                raise ValueError("两点x坐标相同，无法计算斜率。")
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            return lambda x: slope * x + intercept

    except ValueError as ve:
        print(f"发生错误: {ve}")
        return None  # 返回None表示执行失败
    except Exception as e:
        print(f"发生未知错误: {e}")
        return None  # 返回None表示执行失败


#%%
if __name__ == '__main__':
    try:
        data_dir = '../../data/origin/merged.csv'
        df = pd.read_csv(data_dir)

        df = df.groupby('DRUG_NAME')['RQ'].count().sort_values(ascending=False).reset_index()
        df.set_index('DRUG_NAME', inplace=True)
        df1 = taget_count(df, 1, type='d')

        if df1.empty:
            print("没有满足条件的数据，停止执行。")
        else:
            # 计算药物使用比例曲线
            x = np.linspace(0, 1, len(df1))
            y = df1['Byte']

            # 使用curve_fit进行拟合
            params, _ = curve_fit(exp_func, x, y)
            a, b = params

            # 打印拟合参数
            print(f"拟合参数：a = {a}, b = {b}")

            # 使用拟合参数生成拟合曲线
            x_fit = np.linspace(0, 1, 100000)  # 使用相同范围的x_fit
            y_fit = exp_func(x_fit, a, b)

            # 计算梯度并找到斜率最接近1的点
            gradient = np.gradient(y_fit, x_fit)
            index = np.argmin(np.abs(gradient - 1))
            print(f"斜率最接近1的点： {gradient[index]}")
            print(f"斜率最接近1的点的横坐标：{x_fit[index]}")
            print(f"斜率最接近1的点的纵坐标：{y_fit[index]}")

            # 绘制切线
            y_tangent = tangent_line(x_fit, 1, y_fit[index] - x_fit[index])  # 修正切线的截距计算
            plt.plot(x_fit, y_tangent, 'g--', label='切线')

            print(f"拟合曲线：{y_fit}")
            plt.xlim(0, 1)
            plt.ylim(0, 1.2)
            plt.scatter(x, y, label='原始数据')
            plt.plot(x_fit, y_fit, 'r-', label='拟合曲线')
            plt.scatter(x_fit[index], y_fit[index], label='斜率为1的切点')
            plt.text(x_fit[index], y_fit[index], f"({x_fit[index]:.4f}, {y_fit[index]:.4f})", fontsize=10, ha='right')

            plt.title('药物使用比例曲线')
            plt.xlabel('药物使用比例')
            plt.ylabel('累计使用量')
            plt.legend()
            plt.show()

    except FileNotFoundError:
        print("发生错误: 文件未找到，请检查路径。")
    except Exception as e:
        print(f"发生未知错误: {e}")


# %%
data_dir = '../../data/origin/merged.csv'
df = pd.read_csv(data_dir)

df = df.groupby('DRUG_NAME')['RQ'].count().sort_values(ascending=False).reset_index()
df.set_index('DRUG_NAME', inplace=True)
df1 = taget_count(df, 1, type='d')
x = np.linspace(0, 1, len(df1))
y = df1['Byte']
# 计算梯度并找到斜率最接近1的点
gradient = np.gradient(y, x)
index = np.argmin(np.abs(gradient - 1))
print(f"斜率最接近1的点： {gradient[index]}")
print(f"斜率最接近1的点的横坐标：{x[index]}")
print(f"斜率最接近1的点的纵坐标：{y[index]}")
print(f"斜率最接近1的点的index：{index}")
print(df1.loc[:index, :])
# 绘制曲线
plt.xlim(0, 1)
plt.ylim(0, 1.2)
plt.plot(x, y, label='原始数据')
plt.scatter(x[index], y[index], label='斜率为1的切点')
plt.text(x[index], y[index], f"({x[index]:.4f}, {y[index]:.4f})", fontsize=10, ha='right')
#绘制切线
y_tangent = tangent_line(x, 1, y[index] - x[index])  # 修正切线的截距计算
plt.plot(x, y_tangent, 'g--', label='切线')

plt.title('药物使用比例曲线')
plt.xlabel('药物使用比例')
plt.ylabel('累计使用量')
plt.legend()
plt.show()


# %%
import DrugProportion as dp

data_dir = '../../data/origin/merged.csv'
df = pd.read_csv(data_dir)

df = df.crosstab(index='DRUG_NAME', columns='RQ', values='ID', aggfunc='count')
