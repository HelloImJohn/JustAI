import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import math
from matplotlib.pyplot import figure

#var definition..
start = 0
stop = 100 #cm
#length of board that light falls on to..
length = stop - start

distGridBoard = 100 #cm
#slitSize..
b = 0.1 #cm
#slit distance..
d = 1 #cm
#wave length..
w = 550 #nm

figure(figsize=(6, 4), dpi=100)

#amount of light rays per slit..
lps = 100
#amount of slits..
slits = 2


#(slits, light rays, length, dist)
def calc(s, lr, l, d, wl):
  for i in range(s - 1):
    for j in range(lr):
      m = l / 2
      distCenter = abs(m - j + i * d)
      #lIntens = 
      #y values..
      y = np.arange(start, stop, length /lps)
      x = np.arange(start, stop, length /lps)   # start,stop,step..
      y[(i+1)*j] = distCenter
      #print(lIntens)

      plt.plot(x,y)

calc(slits, lps, length, d, w)







#y = np.sin(x)



plt.show()
