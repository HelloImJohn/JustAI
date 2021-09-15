import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import math
from matplotlib.pyplot import figure

#var bestimmung..
start = 0
stop = 100 #cm
#length of board that light falls on to..
length = stop - start

distGridBoard = 100 #cm
#spaltGr..
b = 0.1 #cm
#spalt abstand..
d = 1 #cm
#wave length..
w = 550 #nm

grad = 360

figure(figsize=(6, 4), dpi=100)

#anzahl der Lichtstrahle (adl) pro spalt..
adl = 100
#anzahl der Spalte..
spalte = 2

#y values..
y = np.arange(start, stop, length /adl)
#(spalte, light rays, length, dist)
def calc(s, lr, l, d, wl):
  for i in range(s - 1):
    for j in range(lr):
      p = j * (l / lr)
      m = l / 2
      lIntens = np.tan((p - (m + d + (b / 2))) / d) - np.tan(p - (m - d - (b / 2)) / d)
      y[j] = lIntens
      print(lIntens)

calc(spalte, adl, length, d, w)






x = np.arange(start, stop, length /adl)   # start,stop,step..
#y = np.sin(x)


plt.plot(x,y)
plt.show()
