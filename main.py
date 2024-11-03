
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
def TagetCount(data, taget=1.0, column=0, type='s'):
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
if __name__ == '__main__':
    data_dir = './data/origin/merged.csv'
    df = pd.read_csv(data_dir)

    df = df.groupby('DRUG_NAME')['RQ'].count().sort_values(ascending=False).reset_index()
    df.set_index('DRUG_NAME', inplace=True)
    df1 = TagetCount(df, 1, type='d')
    x = np.linspace(0, 1, len(df1))
    y = df1['Byte']

    # 使用curve_fit进行拟合
    params, covariance = curve_fit(exp_func, x, y)
    # 提取拟合参数
    a, b = params

    # 打印拟合参数
    print(f"拟合参数：a = {a}, b = {b}")

    # 使用拟合参数生成拟合曲线
    x_fit = np.linspace(min(x), max(x), 10000)
    y_fit = exp_func(x_fit, a, b)

    # 绘制切线
    y_tangent = tangent_line(x_fit, 1, 0.578)
    plt.plot(x_fit, y_tangent, 'g--', label='切线')


    print(f"拟合曲线：{y_fit}")

    # 绘制原始数据点
    plt.scatter(x, y, label='原始数据')

    # 绘制拟合曲线
    plt.plot(x_fit, y_fit, 'r-', label='拟合曲线')

    # 添加图例
    plt.legend()

    # 显示图表
    plt.show()


    # %%


    # 定义函数
    def f(x):
        return -0.8806799818246278 * np.exp(-6.480351687210877 * x) + 1


    # 定义 x 的范围
    x_values = np.linspace(-2, 2, 400)
    y_values = f(x_values)

    # 绘制图形
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label='f(x) = -0.8806799818246278 * (-6.480351687210877)^x + 1', color='b')
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.axvline(0, color='black', lw=0.5, ls='--')
    plt.title('Plot of the Exponential Function')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.legend()
    plt.grid()
    plt.show()


    # 读取数据并处理药物名称的交集和并集
    # try:
    #     data_dir = './data/data_crosstab.csv'
    #     df = pd.read_csv(data_dir)
    #     intersection = set(df['DRUG_NAME'].unique())
    #     df.set_index('DRUG_NAME', inplace=True)
    #     taget = 0.831727
    #     union = set()
    #
    #     # 遍历数据框的每一列以计算目标计数和更新交集与并集
    #     for i in range(df.shape[1]):  # 使用 shape 属性提高效率
    #         result = TagetCount(df, taget, i)
    #         if result is not None:  # 检查 TagetCount 的返回值是否合法
    #             section = set(result.unique())
    #             intersection &= section  # 更新交集
    #             union |= section  # 更新并集
    #
    #     # 将并集和交集结果保存为Excel文件
    #     pd.DataFrame({'DRUG_NAME': list(union)}).to_excel('./results/union.xlsx', index=False)
    #     pd.DataFrame({'DRUG_NAME': list(intersection)}).to_excel('./results/intersection.xlsx', index=False)
    #
    # except FileNotFoundError:
    #     print("发生错误: 文件未找到，请检查路径。")
    # except Exception as e:
    #     print(f"发生错误: {e}")









