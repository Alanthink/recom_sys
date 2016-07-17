# 电影推荐系统

## 原始数据

本实验采用数据集[MovieLens 100K Dataset](http://grouplens.org/datasets/movielens/)。这个数据集由国外GroupLens Research团队整理提供。 

这个数据集收集了943个用户对1682部电影的总共10,000条记录。所用文件如下

- u.data: 10,000条记录，每一条记录格式为 user id | item id | rating | timestamp，其中每一个用户至少有20条评价。
- u.user: 每一条记录格式为 user id | age | gender | occupation | zip code。
- u.occupation: 所有工作的列表。



## 数据处理

虽然每个评价在1到5之内，由于每个用户的习惯不一样，导致某个用户评价普遍偏高，或者普遍偏低。为了消除这种影响，需要对每个用户的评价进行正规化，以下是正规化的代码。

```python
# 读入数据
base=np.loadtxt("ml-100k/u.data", int)
# 初始化: 默认情况下每个用户对每部电影都是-1分, 表示没有评价
# rate是一个二维数组, 行数为user行, 列数为movies列
rate=np.ones([users, movies])*(-1)
# 将已经知道的数据录入到rate表里
for j in range(base.shape[0]):
    rate[base[j, 0]-1, base[j, 1]-1]=base[j, 2]
avg=np.zeros(users)
std=np.zeros(users)
for j in range(users):
    tem=rate[j, :]
    tem=tem[tem!=-1]
    avg[j]=(tem.sum()+0.0)/tem.shape[0]
    std[j]=np.std(tem)
rate_V=np.zeros([users, movies])
for j in range(users):
    rate_V[j, :]=avg[j]*np.ones(movies)
rate_V[rate!=-1]=rate[rate!=-1]
# Z-normalization
for i in range(users):
    for j in range(movies):
        rate_V[i, j]=(rate_V[i, j]-avg[i])/std[i]
```



## 算法基本思想

为了向用户进行推荐，

* 对于每一部用户$U$尚未评价的电影$M$，找到与其“相似”并且对电影$M$有评价的那些用户
* 然后取平均值，作为用户$U$对电影$M$的评价
* 将所有估算的电影的评价从高到低排序，选取前几部进行推荐