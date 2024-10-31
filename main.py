
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# def TagetCount(data, taget=1.0, column=0):
#     df = pd.Series(data.iloc[:, column].tolist(), index=data.index.tolist())
#     df.sort_values()
#     total_use = 0
#     byte_of_use = 0
#     sum_of_list = []
#     total = df.sum()
#     df_sum = pd.DataFrame(df)
#     print(df_sum)
#     for i in range(len(df)):
#         total_use += df.iloc[i]
#         byte_of_use = total_use / total
#         sum_of_list.append(byte_of_use)
#         if byte_of_use >= taget:
#             break
#     df_sum['Total'] = sum_of_list
#     return df_sum


def TagetCount(data, taget=1.0, column=0):
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


def com_diff(data):
    x = np.linspace(0, 1, len(data))
    y = np.array(data)
    dy_dx = np.diff(y) / np.diff(x)
    d2y_dx2 = np.diff(dy_dx) / np.diff(x[:-1])
    x_chg = x[1: -1]
    y_chg = y[1: -1]
    return dy_dx,x_chg, y_chg, d2y_dx2


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

    data_dir = './data/data_crosstab.csv'
    df = pd.read_csv(data_dir)
    intersection = set(df['DRUG_NAME'].unique().tolist())
    df.set_index('DRUG_NAME', inplace=True)
    taget = 0.831727
    union = set()
    i = 0
    for i in range(len(df.columns)):
        result = TagetCount(df, taget, i)
        section = set(result.unique().tolist())
        intersection = section & intersection
        union = section | union
    pd.DataFrame({'DRUG_NAME':list(union)}).to_excel('./data/union.xlsx', index=False)
    pd.DataFrame({'DRUG_NAME':list(intersection)}).to_excel('./data/intersection.xlsx', index=False)








