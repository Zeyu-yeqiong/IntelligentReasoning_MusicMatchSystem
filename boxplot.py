# -*- coding:utf-8 -*-


"""

绘制箱体图

Created on 2017.09.04 by ForestNeo

"""

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

"""

generate data from min to max

"""


def list_generator(number, min, max):
    dataList = list()

    for i in range(1, number):
        dataList.append(np.random.randint(min, max))

    return dataList


# generate 4 lists to draw

list1 = [0.559, 0.562,0.554,0.571,0.548,0.553, 0.554,0.553,0.557,0.559]

list2 = [0.589,0.593,0.573,0.579,0.587,0.583,0.589,0.589,0.592,0.593]

list3 = [0.636,0.637,0.642,0.649,0.648,0.642,0.641,0.632,0.629,0.621]

list4 = [0.668,0.669,0.675,0.664,0.663,0.678,0.673,0.656,0.654,0.661]
list5 = [0.675,0.686,0.689,0.676,0.684,0.665,0.670,0.660,0.665,0.661]
list6 = [0.678,0.685,0.691,0.676,0.684,0.668,0.670,0.660,0.665,0.661]
list7 = [	0.645,	0.646,	0.648,	0.648,0.652,0.652,0.654,0.654,0.656,	0.657]
list8 = [	0.601,	0.603,	0.605,	0.605,0.608,0.608,0.609,0.609,0.611,	0.612]
list9 = [	0.652,	0.652,	0.654,	0.654,0.655,0.655,	0.656,0.656,0.657,0.658]
list10 = [0.640,0.641,0.643,0.643,0.645,0.645,	0.649,0.649,0.652,	0.651]
list11 = [0.636,0.635,0.639,0.639,0.642,0.642,0.643,0.643,0.645,0.645]


for i in range(10):
    list1[i] = list1[i]*100
for i in range(10):
    list2[i] = list2[i]*100
for i in range(10):
    list3[i] = list3[i]*100
for i in range(10):
    list4[i] = list4[i]*100
for i in range(10):
    list5[i] = list5[i]*100
for i in range(10):
    list6[i] = list6[i]*100
for i in range(10):
    list7[i] = list7[i]*100
for i in range(10):
    list8[i] = list8[i]*100
for i in range(10):
    list9[i] = list9[i]*100
for i in range(10):
    list10[i] = list10[i]*100
for i in range(10):
    list11[i] = list11[i]*100

data = pd.DataFrame({
"network based feature": list1,
    "Eigenpooling GCN": list2,
    "Population GCN": list3,
    "hi-GCN(two step)": list4,
    "hi-GCN(pre-training)": list5,
    "hi-GCN(jointly learning)": list6,
"BrainNetCNN": list7,
"AveDe": list8,
"t-BNE": list9,
"Graph Boosting": list10,
"Ordinal Pattern": list11
})
print(data)
data.drop([],axis = 1)
print(data)
# draw



data.to_csv('G:\\jh\\boxPlot\\perceptions-master\\probly_ABIDE.csv')
# draw
fig = plt.figure(figsize=(11,7))
data.boxplot()

plt.ylabel("Accuracy")

plt.xlabel("Different Methods on ABIDE")
plt.tight_layout()

plt.savefig('plotbox_ABIDE.jpg',dpi=500)

plt.show()



