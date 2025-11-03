import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('dark_background')
plt.close('all')
s1 = s2 = 1
dm = np.linspace(-4, 4, 100)
Dkl = np.log(s2/s1) + (s1**2 + dm**2) / (2*s2**2) - 1/2
Dw = dm**2 + s1 + s2 - 2*np.sqrt(s1*s2)

plt.plot(dm, Dkl)
plt.plot(dm, Dw)
plt.legend(['Dkl', 'Dw'])
plt.show()