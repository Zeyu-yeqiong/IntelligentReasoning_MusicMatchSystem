class ppp(object):
    def func(self, x, acc):
        y = (866*acc-x*464)/402
        return y
    def func2(self, x, acc):
        y = (133*acc-x*99)/34
        return y

print(ppp.func(object,0.734,0.719))

import numpy as np
print(ppp.func2(object,0.781,0.756))
ccc = np.ones((100, 2))
print(ccc[:,1:])
cos_angle = 0
angle=np.arccos(cos_angle)
print(angle*180/np.pi)