import pandas as pd
import numpy as np
from typing import Union, List, Tuple, Any


def find_min_subset_dp_geq(data: Union[pd.Series, pd.DataFrame, np.ndarray, List[Any]], target_sum: float):
    """
    在一个数据集中找到和大于或等于目标值的最小子集。该函数支持多种数据类型的输入，包括Pandas的DataFrame和Series、
    Numpy数组以及Python列表，并返回与输入类型匹配的输出格式，同时保留原始索引（如果适用）。

    参数:
        data (Union[pd.Series, pd.DataFrame, np.ndarray, List[Any]]): 输入数据，可以是DataFrame、Series、Numpy数组或列表。
        target_sum (int): 目标和，要求子集的总和大于或等于该值。

    返回:
        Union[pd.DataFrame, pd.Series, np.ndarray, Tuple[List[Any], List[int]]]:
        返回满足条件的最小子集，数据格式与输入相同。对于列表输入，返回子集值和索引的元组。
        如果无法找到满足条件的子集，则返回None。

    异常:
        ValueError: 当输入数据类型不受支持时抛出。

    示例:
        df = pd.DataFrame({'values': [1, 2, 3, 4, 5, 6, 7, 8, 9]}, index=[10, 11, 12, 13, 14, 15, 16, 17, 18])
        result = find_min_subset_dp_geq(df, 15)
        print(result)
    """

    # Step 1: 将输入数据转换为带有索引的列表格式，便于后续统一处理
    if isinstance(data, pd.DataFrame):
        # 假设DataFrame只有一列，提取该列的值和索引
        values = data.iloc[:, 0].tolist()
        indices = data.index.tolist()
    elif isinstance(data, pd.Series):
        # 提取Series的值和索引
        values = data.tolist()
        indices = data.index.tolist()
    elif isinstance(data, np.ndarray):
        # 将Numpy数组平展成列表，并生成相应的索引
        values = data.flatten().tolist()
        indices = list(range(len(values)))
    elif isinstance(data, list):
        # 对于Python列表，直接使用列表中的值，并生成索引
        values = data
        indices = list(range(len(data)))
    else:
        # 如果输入类型不支持，抛出异常
        raise ValueError("Unsupported data type. Please use a DataFrame, Series, ndarray, or list.")

    # Step 2: 使用动态规划来寻找满足和大于或等于target_sum的最小子集
    max_sum = sum(values)  # 计算数据集中所有元素的和，用于动态规划的上限
    dp = [float('inf')] * (max_sum + 1)  # 初始化DP数组，dp[i]表示达到和i所需的最小元素个数
    dp[0] = 0  # 和为0时需要0个元素

    # 记录路径，prev[i]保存使得和为i时最后一个元素的索引，用于回溯组合
    prev = [-1] * (max_sum + 1)

    # 动态规划填充dp数组
    for idx, num in enumerate(values):
        for i in range(max_sum, int(num) - 1, -1):
            # 如果dp[i - num] + 1比当前dp[i]更小，则更新dp[i]和prev[i]
            if dp[i - int(num)] + 1 < dp[i]:
                dp[i] = dp[i - int(num)] + 1
                prev[i] = idx  # 保存元素的索引，用于回溯

    # Step 3: 找到第一个满足和大于或等于target_sum的最小元素组合
    min_sum_index = -1  # 保存满足条件的最小和的索引
    for i in range(target_sum, max_sum + 1):
        if dp[i] != float('inf'):
            min_sum_index = i
            break

    # 如果未找到符合条件的组合，返回None
    if min_sum_index == -1:
        return None

    # Step 4: 回溯找到符合条件的子集
    subset_indices = []  # 存储满足条件的子集的索引
    while min_sum_index > 0:
        idx = prev[min_sum_index]
        subset_indices.append(indices[idx])
        min_sum_index -= int(values[idx])  # 减去当前元素值，继续回溯

    subset_indices = sorted(subset_indices)  # 保持原始顺序
    subset_values = [values[indices.index(idx)] for idx in subset_indices]

    # Step 5: 根据输入数据类型返回相应格式的结果
    if isinstance(data, pd.DataFrame):
        # 返回带有原始索引的DataFrame
        return data.loc[subset_indices]
    elif isinstance(data, pd.Series):
        # 返回带有原始索引的Series
        return pd.Series(subset_values, index=subset_indices)
    elif isinstance(data, np.ndarray):
        # 返回Numpy数组
        return np.array(subset_values)
    elif isinstance(data, list):
        # 返回子集值和索引的元组
        return subset_values, subset_indices
    else:
        return None  # 理论上不会到达这里

def min_subset(data, target_sum):
    data = pd.Series(data).sort_values()
    sum = 0
    i = 0
    while sum < target_sum:
        sum += data.iloc[i]
        i += 1
    return data.iloc[:i]
