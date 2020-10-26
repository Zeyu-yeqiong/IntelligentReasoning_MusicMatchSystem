import matplotlib.pyplot as plt


def QuickSort(array, left, right):
    if left < right:
        #第一步定好那个基准
        #然后在基准的左边右边用递归的形式使用QuickSort
        IndexBaseline = ZoneDivision(array, left, right)
        #表示当前Indexbaseline的值就是一个中间值，不用再Sort
        #所以下面两个函数的结束和开始为IndexBaseline-1 和 IndexBaseline+1
        QuickSort(array, left, IndexBaseline-1)
        QuickSort(array, IndexBaseline+1, right)
    return array

def ZoneDivision(array, left, right):
    baseline = left
    index = baseline + 1
    for i in range(index, right+1):
        if array[i]<array[baseline]:
            array[i], array[index] = array[index], array[i]
            index = index + 1
    # 这个时候index相当于已经到了这个分解点，此时就要把当前baseline和index-1处的数进行交换
    # 交换的目的就是把baseline对应的值放到此次分区的最右边
    array[baseline],array[index-1] = array[index-1],array[baseline]
    #最终返回的是当前次分区的分区索引
    return index-1

if __name__ == '__main__':
    inputarray = [7,1,4,2,3,6,5,0,9,8]
    left = 0
    right = len(inputarray)-1
    output = QuickSort(inputarray,left,right)
    print(output)




def fast_sort(slist, start, end):
    if start >= end:
        return
    low = start
    high = end
    mid_slist = slist[start]
    while low < high:
        while high > low and slist[high] >= mid_slist:
            high -= 1
        print('1', slist)
        slist[low] = slist[high]
        print('1',slist)
        print('1', low)
        print('1', high)
        while high > low and slist[low] < mid_slist:
            low += 1
        print('2', slist)
        slist[high] = slist[low]
        print('2', slist)
        print('2', low)
        print('2', high)
    slist[low] = mid_slist
    fast_sort(slist, start, low - 1)
    fast_sort(slist, low + 1, end)
    return slist
inputarray = [7,1,4,2,3,6,5,0,9,8,-1]
#inputarray[1] = inputarray[2]
#print(inputarray)
print(fast_sort(inputarray,0,len(inputarray)-1))



from numpy import *
# 加载数据
def loadDataSet(fileName):  # 解析文件，按tab分割字段，得到一个浮点数字类型的矩阵
    dataMat = []              # 文件的最后一个字段是类别标签
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)    # 将每个元素转成float类型
        dataMat.append(fltLine)
    return dataMat
# 计算欧几里得距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) # 求两个向量之间的距离
# 构建聚簇中心，取k个(此例中为4)随机质心
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))   # 每个质心有n个坐标值，总共要k个质心
    for j in range(n):
        minJ = min(dataSet[:,j])
        maxJ = max(dataSet[:,j])
        rangeJ = float(maxJ - minJ)
        centroids[:,j] = minJ + rangeJ * random.rand(k, 1)
    return centroids
# k-means 聚类算法
def kMeans(dataSet, k, distMeans =distEclud, createCent = randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))    # 用于存放该样本属于哪类及质心距离
    # clusterAssment第一列存放该数据所属的中心点，第二列是该数据到中心点的距离
    centroids = createCent(dataSet, k)
    clusterChanged = True   # 用来判断聚类是否已经收敛
    while clusterChanged:
        clusterChanged = False;
        for i in range(m):  # 把每一个数据点划分到离它最近的中心点
            minDist = inf; minIndex = -1;
            for j in range(k):
                distJI = distMeans(centroids[j,:], dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j  # 如果第i个数据点到第j个中心点更近，则将i归属为j
            if clusterAssment[i,0] != minIndex: clusterChanged = True;  # 如果分配发生变化，则需要继续迭代
            clusterAssment[i,:] = minIndex,minDist**2   # 并将第i个数据点的分配情况存入字典
        print(centroids)
        for cent in range(k):   # 重新计算中心点
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]   # 去第一列等于cent的所有列
            centroids[cent,:] = mean(ptsInClust, axis = 0)  # 算出这些数据的中心点
    return centroids, clusterAssment

print(inputarray.pop(-1))

fpr = [0,        0.383286 ,1.        ]
tpr = [0,         0.683286 ,1.        ]

plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='darkorange',
         lw=2)  ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

test_true = np.array([0, 0, 0, 0, 1, 1, 1])
test_pre = np.array([0.3, 0.2, 0.7, 0.5, 0.4, 0.9, 0.6])
fpr, tpr, thresholds = metrics.roc_curve(test_true, test_pre, pos_label=1)

print(fpr)