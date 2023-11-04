import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# 读取数据
data = pd.read_csv('../row_data/train_set.csv')
X = data[['刑期']]

# 绘图确定K值
# inertias = []
# range_k = range(1, 10)  # 尝试 k 从 1 到 10
# for k in range_k:
#     kmeans = KMeans(n_clusters=k)
#     kmeans.fit(X)
#     inertias.append(kmeans.inertia_)
#
# plt.plot(range_k, inertias, '-o')
# plt.xlabel('Number of clusters, k')
# plt.ylabel('Inertia')
# plt.title('The Elbow Method showing the optimal k')
# plt.show()

# 使用K均值聚类
kmeans = KMeans(n_clusters=4)
data['Cluster'] = kmeans.fit_predict(X)

# 获取每个聚类的统计信息
cluster_stats = data.groupby('Cluster')['刑期'].describe()

# 按最大值的大小对聚类进行升序排序
cluster_stats = cluster_stats.sort_values(by='max', ascending=True)

# 为每个聚类重新分配类别数字
cluster_stats['New_Cluster'] = range(0, len(cluster_stats))

# 将新的类别数字映射回原始数据
data = data.merge(cluster_stats[['New_Cluster']], left_on='Cluster', right_index=True)

# 删除原始的'Cluster'列
data.drop('Cluster', axis=1, inplace=True)

# 绘制重新标记的聚类结果
plt.figure(dpi=300)
plt.scatter(data.index, data['刑期'], c=data['New_Cluster'], cmap='rainbow')
plt.xlabel('Index')
plt.ylabel('刑期')
plt.title('K-means Clustering (Sorted by Max Value)')
plt.show()

# # 打印重新标记的聚类的统计信息
# print(cluster_stats[['New_Cluster', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']], end='')
# 打印重新标记的聚类的统计信息
print(cluster_stats[['New_Cluster', 'count', 'mean', 'std', 'min', 'max']])
