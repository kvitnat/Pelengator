import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from scipy import linalg
from numpy.linalg import inv


#x = np.matrix([[3.,17,27,12,0], [2,0,14,16,14], [46,120,211,257,320]]);m = 5
x = np.matrix([[2.,2,9,11], [2,9,5,1], [60,99,14,160]]); m = 4

plt.xlim(0, 11)
plt.ylim(0, 11)
plt.axis('scaled')
minx=0
maxx=11
x[2] = np.tan(np.deg2rad(x[2]))
b = -np.multiply(x[0], x[2])+x[1]
a = np.asmatrix(np.column_stack((np.squeeze(np.asarray(-x[2])),np.ones(m))))

b = np.asmatrix(np.squeeze(np.asarray(b))).T
xaxis=np.arange(minx,maxx,0.1)
xaxis=np.append(xaxis,maxx)
k=-a.T[0]


for i in range(a.shape[0]): #lines
    yaxis=xaxis*k[0,i]+b[i,0]
    plt.plot(xaxis,yaxis,"c")

plt.plot(x[0],x[1],'r*')    #dots

ata=np.dot(a.T,a)
r=np.dot(a.T,b)
x=np.dot(inv(ata),r)
plt.plot(x[0],x[1],'bo')
plt.show()
print("X: "); print(x)
#------------------------------------------------------------

plt.xlim(-10, 25)
plt.ylim(-10, 25)
plt.axis('scaled')

XYH = np.matrix([[2., 11, 16, 10, 4],
                 [3, 1, 6, 9, 9],
                 [6.8, 7, 6.8, 4.2, 5.8]])
m = 5
def J(b):
    J_ = np.empty((m, 2))
    for i in range(m):
        J_[i, 0] = (XYH[0, i] - b[0])/math.sqrt((b[0] - XYH[0, i])**2 + (b[1] - XYH[1, i])**2)
        J_[i, 1] = (XYH[1, i] - b[1])/math.sqrt((b[0] - XYH[0, i])**2 + (b[1] - XYH[1, i])**2)
    return J_

def r(b):
    r_ = np.empty((m))
    for i in range(m):
        r_[i] = XYH[2, i] - math.sqrt((b[0] - XYH[0, i])**2 + (b[1] - XYH[1, i])**2)
    return r_

plt.plot(XYH[0],XYH[1],'y*')  #dots

for i in range(XYH.shape[1]): #circles
    x = np.linspace(XYH[0, i] - XYH[2, i] - 1, XYH[0, i] + XYH[2, i] + 1, 100)
    y = np.linspace(XYH[1, i] - XYH[2, i] - 1, XYH[1, i] + XYH[2, i] + 1, 100)
    X, Y = np.meshgrid(x,y)
    F = XYH[2, i]*XYH[2, i] - (X - XYH[0, i])*(X - XYH[0, i]) - (Y - XYH[1, i])*(Y - XYH[1, i])
    plt.contour(X, Y, F, [0], colors = "green")

e = 0.1
b = np.array([[5], [5]])
for i in range(100):
    jtj = np.dot(np.asmatrix(J(b)).T, np.asmatrix(J(b)))
    jtr = -np.dot(np.asmatrix(J(b)).T, r(b))
    deltaB = np.dot(np.linalg.inv(jtj), jtr.T)
    b = b + deltaB
    if(np.linalg.norm(deltaB) < e):
        break
print("X = "); print(b)
plt.plot(b[0], b[1], 'ro')
plt.show()

