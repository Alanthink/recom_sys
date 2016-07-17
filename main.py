# coding=utf-8
import numpy as np
from scipy.spatial import distance

# 用户总数, 用户用0到942来标识
users=943
# 电影数目, 电影用0到1681来标识
movies=1682
# 用户工作的种类
jobs_num=21
# 所有的工作
jobs=['administrator', 'artist','doctor','educator','engineer','entertainment','executive',
	'healthcare','homemaker','lawyer','librarian','marketing','none','other','programmer','retired','salesman'
	,'scientist','student','technician','writer']
# 用户数据
user_raw=np.loadtxt("ml-100k/u.user", str, delimiter='|')
# 默认推荐5部电影
guess=5
# 选取相似的用户数目
selection=15
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


def compute_dist1(user_guess):
    # tem_1d为一维数据
    tem_1d=rate_V[user_guess, :]
    # 将tem_1d转化为二维数据
    tem_2d=np.zeros([1, movies])
    tem_2d[0, :]=tem_1d
    # 算法二 根据用户以往观看过的电影进行推荐
    # 计算任意两个用户的距离, 距离越小代表相似度越小
    dist=distance.cdist(tem_2d, rate_V, 'euclidean')
    # 将得到的结果转化为一维数据
    dist=dist[0, :]
    return dist


def compute_dist2(user_guess):
    user_data=np.zeros([users, jobs_num+4])
    for j in range(user_raw.shape[0]):
        if user_raw[j, 2]=='M':
            user_data[j, jobs_num]=1
        for k in range(jobs_num):
            if user_raw[j, 3]==jobs[k]:
                user_data[j, k]=1
                break
        age=int(user_raw[j, 1])
        # 将年龄分为3个年临段分为(0~30), (30~60), (60~)
        if age<30:
            user_data[j, jobs_num+1]=1
        elif age<60:
            user_data[j, jobs_num+2]=1
        else:
            user_data[j, jobs_num+3]=1
    tem_1d=user_data[user_guess, :]
    tem_2d=np.zeros([1, jobs_num+4])
    tem_2d[0, :]=tem_1d
    dist=distance.cdist(tem_2d, user_data, 'cityblock')
    dist=dist[0, :]
    return dist


def recom(dist, i, user_guess):
    # 初始化猜测的分数
    pred=np.ones(movies)*(-100)
    for j in range(movies):
        # 当没有对一部电影评价的时候才需要预测
        if rate[user_guess, j]==-1:
            tem=np.array([rate_V[:, j], dist]).T
            # 选取所有对这部电影有评价的其它用户的评价
            tem=tem[rate[:, j]!=-1]
            # 根据距离进行排序
            tem=tem[tem[:, 1].argsort()]
            # 取所有的评价
            tem=tem[:, 0]
            if tem.shape[0]>selection:
                tem=tem[0:selection]
            if tem.shape[0]>0:
                pred[j]=(tem.sum()+0.0)/tem.shape[0]
    tem=np.array([range(movies), pred]).T
    # 去掉那些已经评价的
    tem=tem[rate[:, user_guess]!=-1]
    tem=tem[tem[:, 1].argsort()][::-1]
    print("算法%d推荐的电影的编号(排名不分先后):"%(i))
    print tem[0:guess, 0]


if __name__ == '__main__':
    user_guess=int(raw_input("请输入需要预测的用户编号(0到1681):\n"))
    if user_guess not in range(1682):
        print("没有这个用户")
        exit(0)
    dist1=compute_dist1(user_guess)
    dist2=compute_dist2(user_guess)
    a=0.3
    dist3=dist1+a*dist2
    recom(dist1, 1, user_guess)
    recom(dist2, 2, user_guess)
    recom(dist3, 3, user_guess)
