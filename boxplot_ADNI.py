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

ADNI = pd.read_excel('G:\\jh\\boxPlot\\perceptions-master\\probly_ABIDE.xlsx')
# generate 4 lists to draw

list1 = [0.683, 0.683,0.683,0.683,0.711,0.711, 0.663,0.663,0.643,0.733]

list2 = [0.719,0.778,0.778,0.778,0.803,0.719,0.682,0.782,0.719,0.719]

list3 = [0.737,0.737,0.777,0.778,0.782,0.812,0.648,0.719,0.682,0.651]

list4 = [0.734,0.774,0.833,0.678,0.734,0.734,0.734,0.774,0.774,0.774]
list5 = [0.734,0.779,0.833,0.694,0.734,0.714,0.740,0.774,0.774,0.774]
list6 = [0.734,0.784,0.833,0.694,0.734,0.714,0.744,0.774,0.774,0.774]

list7 = [0.716,	0.716,0.721,0.721,0.729,0.729,0.736,	0.736, 0.743, 0.743]
list8 = [0.690,0.690,0.697,0.697,0.702,0.702,0.709,0.709,		0.716,	0.716]
list9 = [0.704,0.704,0.71,0.71,0.718,0.718,0.722,		0.722,	0.729,		0.729]
list10 = [0.721,0.721,0.726,0.726,	0.731,0.731,		0.736,0.736,0.743,	0.743]
list11 = [	0.719,	0.720,	0.726,	0.726,0.73,0.73,0.732,0.732,0.739,	0.739]

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
data.drop([],axis = 1)
print(data)
# draw



data.to_csv('G:\\jh\\boxPlot\\perceptions-master\\probly_ADNI.csv')
fig = plt.figure(figsize=(11,7))
data.boxplot()

plt.ylabel("Accuracy")

plt.xlabel("Different Methods on ADNI")
plt.tight_layout()
plt.ylim(0.55, 0.9)
plt.savefig('plotbox_ADNI.jpg',dpi=500)

plt.show()



