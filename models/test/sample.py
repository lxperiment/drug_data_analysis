import numpy as np
import heapq


def timeit(func):
    import time
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"函数 {func.__name__} 耗时 {end_time - start_time:.4f} 秒")
        return result

    return wrapper
class Sample:
    def __init__(self, data):
        # 确保输入数据是一个一维数组
        if not isinstance(data, (list, np.ndarray)):
            raise ValueError("数据必须是一个列表或numpy数组")
        self.data = np.asarray(data).flatten()

    @timeit
    def k_largest(self, k):
        if k < 1:
            raise ValueError("k 必须大于0")
        return heapq.nlargest(k, self.data)

    @timeit
    def k_smallest(self, k):
        if k < 1:
            raise ValueError("k 必须大于0")
        return heapq.nsmallest(k, self.data)

    def mean(self):
        return np.mean(self.data)

    def median(self):
        return np.median(self.data)

    def mode(self):
        # 使用np.unique获取唯一值及其计数
        values, counts = np.unique(self.data, return_counts=True)
        return values[np.argmax(counts)] if counts.size > 0 else None

    def sort(self):
        return np.sort(self.data)

    @timeit
    def n_largest(self, n):
        if n < 1:
            raise ValueError("n 必须大于0")
        return self.sort()[:n]

    def __eq__(self, other):
        return np.array_equal(self.data, other.data)

    def __ne__(self, other):
        return not self.__eq__(other)

    def variance(self):
        return np.var(self.data)

    def standard_deviation(self):
        return np.std(self.data)

    def skewness(self):
        mean_val = np.mean(self.data)
        std_val = np.std(self.data)
        if std_val == 0:
            return 0
        return np.mean((self.data - mean_val) ** 3) / std_val ** 3



if __name__ == '__main__':
    data = np.random.rand(100000)
    sample = Sample(data)
    try:
        sample.k_largest(20000)
        sample.k_smallest(20000)
        print(sample.n_largest(20000))

    except ValueError as e:
        print(f"错误: {e}")
