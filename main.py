import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
df_data = pd.read_csv("bank.csv")
print(df_data.shape)
print(df_data.head())
print(df_data.info())
print(df_data.describe())
print(df_data.duplicated().sum())
X = StandardScaler()
scaled_data = pd.DataFrame(X.fit_transform(df_data.iloc[:, 1:6]), columns=df_data.columns[1:])
print(scaled_data)

wss = []
for i in range(1, 11):
    k_means = KMeans(n_clusters = i, random_state = 425)
    k_means.fit(scaled_data)
    k_means.inertia_
    wss.append(k_means.inertia_)
print(wss)
a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
plt.title('Wss Vs K_Value')
sns.pointplot(x=a, y=wss)
plt.savefig('wss_vales.jpg')
plt.show()
max_= -1
K_value=3
for j in range(2,5):
    k_means = KMeans(n_clusters=j, random_state=425)
    print(k_means)
    k_means.fit(scaled_data)
    labels = k_means.labels_
    s_score=silhouette_score(scaled_data,labels,random_state=425)
    if(max_<s_score):
        max_=s_score
        K_value=j
print(K_value, max_)
k_means = KMeans(n_clusters=K_value, random_state=425)
k_means.fit(scaled_data)
labels = k_means.labels_

df_data["K_mean_cls"] = labels
print(df_data)
plt.title('K_Means Clustering')
plt.scatter(scaled_data['Withdrawals'], scaled_data['DD'],c=labels)
plt.savefig('K_mean_dist.jpg')
plt.show()

print(df_data.K_mean_cls.value_counts().sort_index())
cluster_table = df_data.drop(['Bank'], axis=1)
cluster_table = df_data.groupby('K_mean_cls').mean()
cluster_table['frequency']=df_data.K_mean_cls.value_counts().sort_index()
print(cluster_table)
cluster_table.to_csv('Bank_Clusters.csv')