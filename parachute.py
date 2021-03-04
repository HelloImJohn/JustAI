import numpy as np
import matplotlib.pyplot as plt

#initail velocity in m/s..
vInit = 0
#acceleration in m/s..
aG = 9.81
#coefficient of drag..
cw = 1.4
#air density in kg^-(m^3)..
q = 1.29
#area of drag producing surface..
ad = 4* np.pi
#number of runs..
f = 1000
#amount of kg the payload weights..
#this example will use 9.81m/s of acceleration by default.. 
pKG = 80
#time to calculate fall for in seconds..
t = 10
#storing temp results for graphing..
#speed..
tempValS = np.arange(0.0, f)
#acceleration..
tempValA = np.arange(0.0, f)
#x axis for plot..
xAxis = np.arange(0, f)
xAxis = xAxis/(f/t)

#calculate parachute resistance..
def fwl(vInit ,f):
  #local speed variable..
  vLoc = vInit
  #amount of time that passes per step..
  
  tOfA = t/f
  for i in range(f):
    tempFwl = (0.5*cw) * ad * q * (vLoc ** 2)
    #print(tempFwl)
    a = diff(pKG, tempFwl, tOfA, vLoc)
    vLoc += a
    tempValS[i] = vLoc
    tempValA[i] = a * f/t
  #print(tempVal)

#calculate difference and potential acceleration due to not being at terminal velocity..
def diff(payloadWeightKG, Fwl, t, vBase):
  #this is the acceleration that will be experienced by the payload (+ goes down (with gravity).. - goes up (against gravity))..
  a = (((payloadWeightKG * aG) - Fwl) / payloadWeightKG) * t
  print(a)
  return a



#graph the values..
def graph(inputS, inputA, fidelity):
  plt.title('Speed of Fall with parachute',fontsize=16)
  plt.xlabel('time in s',fontsize=14)
  plt.ylabel('velocity in m/s',fontsize=14)
  #x = np.arange(0, 100)
  #y = np.sin(x)
  #print(input)
  plt.plot(xAxis, inputS, color='orange',label='velocity')
  plt.plot(xAxis, inputA, color='lightblue',label='downward acceleration')
  plt.legend()
  plt.show()

fwl(vInit, f)
graph(tempValS, tempValA, f)