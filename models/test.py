import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import AffinityPropagation
from itertools import cycle

# 设置为使用一个线程
os.environ["OMP_NUM_THREADS"] = '1'

try:
    # 设置样本数量和特征数量
    n_samples = 1000
    n_features = 4

    # 为不同的聚类创建随机数据
    clusters = [
        np.random.normal(size=(200, n_features), scale=0.5) + np.random.normal([0, 0, 0, 0], scale=0.5),
        np.random.normal(size=(200, n_features), scale=1) + np.array([5, 5, 5, 5]),
        np.random.normal(size=(200, n_features), scale=1.5) + np.array([-5, -5, -5, -5]),
        np.random.normal(size=(200, n_features), scale=2) + np.array([5, -5, 5, -5]),
        np.random.normal(size=(200, n_features), scale=2.5) + np.array([-5, 5, -5, 5]),
    ]

    # 将聚类合并为一个数据集
    X = np.vstack(clusters)

    # 创建含有聚类 ID 的 DataFrame
    cluster_id = np.concatenate([np.full(200, i) for i in range(len(clusters))])
    df = pd.DataFrame(X, columns=["feature_1", "feature_2", "feature_3", "feature_4"])
    df["cluster_id"] = cluster_id

    # 拟合模型
    af = AffinityPropagation(preference=-563, random_state=0).fit(X)
    cluster_centers_indices = af.cluster_centers_indices_
    af_labels = af.labels_
    n_clusters_ = len(cluster_centers_indices)

    # 打印聚类数量
    print(n_clusters_)

    # 绘制数据
    plt.close("all")
    plt.figure(1)
    plt.clf()

    colors = cycle("bgrcmykbgrcmyk")
    for k, col in zip(range(n_clusters_), colors):
        class_members = af_labels == k
        cluster_center = X[cluster_centers_indices[k]]
        plt.plot(X[class_members, 0], X[class_members, 1], col + ".")
        plt.plot(cluster_center[0], cluster_center[1], "o", markerfacecolor=col, markeredgecolor="k", markersize=14)
        for x in X[class_members]:
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

    # 设置图表标题
    plt.title("Estimated number of clusters: %d" % n_clusters_)
    plt.show()

except Exception as e:
    # 处理异常
    print("发生错误：", e)
