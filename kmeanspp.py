from numpy import *
import codecs
 
#计算欧氏距离
def distance(x1,x2):
    return sqrt(sum(power(x1-x2,2)))
 
#对一个样本找到与该样本距离最近的聚类中心
def nearest(point, cluster_centers):
    min_dist = inf
    m = np.shape(cluster_centers)[0]  # 当前已经初始化的聚类中心的个数
    for i in range(m):
        # 计算point与每个聚类中心之间的距离
        d = distance(point, cluster_centers[i, ])
        # 选择最短距离
        if min_dist > d:
            min_dist = d
    return min_dist
#选择尽可能相距较远的类中心
def get_centroids(dataset, k):
    m, n = np.shape(dataset)
    cluster_centers = np.zeros((k , n))
    index = np.random.randint(0, m)
    cluster_centers[0,] = dataset[index, ]
    # 2、初始化一个距离的序列
    d = [0.0 for _ in range(m)]
    for i in range(1, k):
        sum_all = 0
        for j in range(m):
            # 3、对每一个样本找到最近的聚类中心点
            d[j] = nearest(dataset[j, ], cluster_centers[0:i, ])
            # 4、将所有的最短距离相加
            sum_all += d[j]
        # 5、取得sum_all之间的随机值
        sum_all *= random.rand()
        # 6、获得距离最远的样本点作为聚类中心点
        for j, di in enumerate(d):
            sum_all=sum_all - di
            if sum_all > 0:
                continue
            cluster_centers[i,] = dataset[j, ]
            break
    return cluster_centers
 
#主程序
def Kmeans(dataset,k):
    row_m=shape(dataset)[0]
    cluster_assign=zeros((row_m,2))
    center=get_centroids(dataset,k)
    change=True
    while change:
        change=False
        for i in range(row_m):
            mindist=inf
            min_index=-1
            for j in range(k):
                distance1=distance(center[j,:],dataset[i,:])
                if distance1<mindist:
                    mindist=distance1
                    min_index=j
            if cluster_assign[i,0] != min_index:
                change=True
            cluster_assign[i,:]=min_index,mindist**2
        for cen in range(k):
            cluster_data=dataset[nonzero(cluster_assign[:,0]==cen)]
            center[cen,:]=mean(cluster_data,0)
    return center ,cluster_assign
cluster_center,cluster_assign=Kmeans(datas,3)

