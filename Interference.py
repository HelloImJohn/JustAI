import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import math
from matplotlib.pyplot import figure

#var definition..
#start should be less that stop!
start = 0
stop = 100 #cm
#length of board that light falls on to..
length = stop - start
if start > stop:
  raise ValueError('stop must be greater than start!')

#dist grid board must be greater than zero!
distGridBoard = 100 #cm
if distGridBoard <= 0:
  raise ValueError('distGridBoard must be greater than zero!')
#slitSize..
b = 0.1 #cm
#distance from one slit to another..
d = 1 #cm
#wave length..
w = 550 * (1 / np.power(10, 9)) #m to nm..

figure(figsize=(6, 4), dpi=100)

#amount of light rays per slit..
lps = 100
#amount of slits..
slits = 4
#amount to scale the final output interference by..
scale = 2


#(slits, light rays, length, dist, wave length, distance from grid board)
def calc(s, lr, l, d, wl, dGB):
  #matr = np.linspace((1,2,4),(10,20,40),10)
  matr = np.linspace(np.zeros(lr),np.zeros(lr), s)
  for i in range(s):
    y = np.zeros(lr)
    x = np.arange(start, stop, length /lps) 
    for j in range(lr):
      m = l / 2
      #print(j)
      distCenter = abs(m - j + i * d - ((s/2) * d))
      #lIntens = 
      #y values..
      #totalDistance (hypotinuse)
      tD = np.sqrt(np.power(dGB, 2) + np.power(distCenter, 2))     
      #y[j] = distCenter      ## use this to plot the distance from the centerline..
      matr[i][j] = tD
    #plt.plot(x,matr[i])

  matrIntens = np.linspace(np.zeros(lr),np.zeros(lr), s)
  for i in range(s - 1):
    for j in range(s - 1):
      for k in range(lr):
        y[j] += (((abs(matr[i][k] - matr[j + 1][k]) / wl) % 1) - 0.5) * scale
    plt.plot(x,y)
  plt.show()
  #print(matrIntens)

calc(slits, lps, length, d, w, distGridBoard)

#(((outputDiff / waveLength) % 1) - 0.5) * scale
print((((250 / 500) % 1) - 0.5) * 2)
