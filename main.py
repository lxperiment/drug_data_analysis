
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 计算指定列中累计达到目标值的条目
def TagetCount(data, taget=1.0, column=0):
    """
    计算在指定列中累计达到目标值的条目

    参数:
    data : DataFrame - 输入数据
    taget : float - 目标值，默认为1.0
    column : int - 要检查的列索引，默认为0

    返回:
    Series - 达到目标的条目列表；如果没有达到，则返回空DataFrame；发生错误时返回None
    """

    try:
        # 确保指定的列存在于数据中
        if column < 0 or column >= data.shape[1]:
            raise ValueError("指定的列索引超出范围。")

        df = pd.Series(data.iloc[:, column], index=data.index)
        df_sorted = df.sort_values(ascending=False)
        total = df_sorted.sum()

        if total == 0:
            raise ValueError("列中的总和为0，无法进行计算。")

        list_of_drug = []
        total_use = 0

        for index,value in df_sorted.items():
            total_use += value
            byte_of_use = total_use / total
            list_of_drug.append(index)
            if byte_of_use >= taget:
                return pd.Series(list_of_drug)

        return pd.DataFrame(columns=["Values"])  # 如果没有满足条件的返回空DataFrame

    except ValueError as ve:
        print(f"发生错误: {ve}")
        return None  # 返回None表示执行失败
    except Exception as e:
        print(f"发生未知错误: {e}")
        return None  # 返回None表示执行失败


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



if __name__ == '__main__':
    # data_dir = './data/count_nunique.csv'
    # df = pd.read_csv(data_dir)
    # df.set_index('DRUG_NAME', inplace=True)
    # data = TagetCount(df, 0.83)
    # print(data)
    # print(df)

    # #导出交叉表
    # data_dir = './data/data.csv'
    # df = pd.read_csv(data_dir)
    # df = pd.crosstab(index=df['DRUG_NAME'], columns=df['RQ']).reset_index()
    # df.to_csv('./data/data_crosstab.csv', index=False)

    # 读取数据并处理药物名称的交集和并集
    try:
        data_dir = './data/data_crosstab.csv'
        df = pd.read_csv(data_dir)
        intersection = set(df['DRUG_NAME'].unique())
        df.set_index('DRUG_NAME', inplace=True)
        taget = 0.831727
        union = set()

        # 遍历数据框的每一列以计算目标计数和更新交集与并集
        for i in range(df.shape[1]):  # 使用 shape 属性提高效率
            result = TagetCount(df, taget, i)
            if result is not None:  # 检查 TagetCount 的返回值是否合法
                section = set(result.unique())
                intersection &= section  # 更新交集
                union |= section  # 更新并集

        # 将并集和交集结果保存为Excel文件
        pd.DataFrame({'DRUG_NAME': list(union)}).to_excel('./results/union.xlsx', index=False)
        pd.DataFrame({'DRUG_NAME': list(intersection)}).to_excel('./results/intersection.xlsx', index=False)

    except FileNotFoundError:
        print("发生错误: 文件未找到，请检查路径。")
    except Exception as e:
        print(f"发生错误: {e}")









