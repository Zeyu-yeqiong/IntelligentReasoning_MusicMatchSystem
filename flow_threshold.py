import matplotlib.pyplot as plt
from pylab import *                                 #支持中文
mpl.rcParams['font.sans-serif'] = ['SimHei']
x = [i/10 for i in range(0,11)]
names = [i/10 for i in range(0,11)]

ASD_ACC = [0.562,
0.595,
0.625,
0.651,
0.677,
0.712,
0.731,
0.722,
0.701,
0.685,
0.672,
]
AD_ACC=[0.457,
0.528,
0.556,
0.602,
0.651,
0.712,
0.763,
0.789,
0.777,
0.752,
0.73,
]
ASD_AUC = [0.623,
0.688,
0.721,
0.764,
0.803,
0.811,
0.823,
0.806,
0.787,
0.775,
0.769,

]
AD_AUC=[0.675,
0.708,
0.756,
0.782,
0.816,
0.829,
0.844,
0.851,
0.869,
0.852,
0.849,
]
#plt.plot(x, y, 'ro-')
#plt.plot(x, y1, 'bo-')
#pl.xlim(-1, 11)  # 限定横轴的范围
#pl.ylim(-1, 110)  # 限定纵轴的范围
plt.ylim(0.4, 0.9)
plt.plot(x, ASD_ACC, marker='o', mec='r', color= 'r',label='ASD_ACC')
plt.plot(x, AD_ACC, marker='o', mec='b',color= 'b',label='AD_ACC')
plt.plot(x, ASD_AUC, marker='^', mec='r', color= 'r',label='ASD_AUC')
plt.plot(x, AD_AUC, marker='^', mec='b', color= 'b',label='AD_AUC')
plt.legend()  # 让图例生效
plt.xticks(x, names, rotation=45)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
# plt.xlabel('Various of Thresholds') #X轴标签
plt.ylabel("Performance") #Y轴标签
# plt.title("ACC and AUC on Different Threshold") #标题
plt.savefig('flow_threshold.pdf')
plt.show()