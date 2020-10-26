
import numpy as np
from scipy.stats import ttest_ind
from scipy.stats import ttest_ind_from_stats
statistic, pvalue = ttest_ind_from_stats(mean1=0.749, std1=np.sqrt(0.0015), nobs1=100,
                    mean2=0.785, std2=np.sqrt(0.0015), nobs2=10)

a = [0.683, 0.683,0.683,0.683,0.711,0.711, 0.663,0.663,0.643,0.733]

list2 = [0.719,0.778,0.778,0.778,0.832,0.719,0.682,0.782,0.719,0.719]

b = [0.737,0.737,0.777,0.778,0.782,0.812,0.648,0.719,0.682,0.621]

list4 = [0.734,0.774,0.833,0.678,0.734,0.734,0.734,0.774,0.774,0.774]
list5 = [0.734,0.779,0.803,0.694,0.734,0.714,0.740,0.774,0.774,0.774]
bc = [0.734,0.784,0.803,0.694,0.734,0.714,0.744,0.774,0.774,0.774]
print(np.mean(a), np.var(b))

a, b = ttest_ind(a, b)
print(statistic, pvalue)
print(a, b)
