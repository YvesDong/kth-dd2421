import functions as fun
import numpy as np
import random, math
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# GENERATE DATAPOINTS ###############################################################################
Num = 20 # number of data points
classA = np.concatenate((np.random.randn(int(Num/2), 2)*0.2 + [1.5, 0.5], np.random.randn(int(Num/2), 2)*0.2 + [-1.5, 0.5]))
classB = np.random.randn(Num, 2)*0.2 + [0.0, -0.5]

inputs = np.concatenate((classA, classB))
targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))

N = inputs.shape[0] # Number of rows (samples)

permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute,:]
targets = targets[permute]

# PLOT DATA ########################################################################################
plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')

plt.axis('equal')
plt.savefig('svmplot.pdf')
plt.show()


# OPTIMIZATION #####################################################################################

# This part needs to be worked on 

# pre compute kernel matrix P
P = fun.kernelMatrix(N, targets, inputs)

# modeling
C = None # slack variable
start = np.zeros((N))
b = np.array([0,C])
bounds = [(0, C) for b in range(N)]

constraints = {'type':'eq', 'fun':fun.zerofun   }
XC = [constraints for b in constraints]

ret = minimize(fun.objective, start, bounds, constraints)

# PLOT DECISION BOUNDARIES ########################################################################
xgrid = np.linspace(-5, 5)
ygrid = np.linspace(-4, 4)

# grid =np.array([[indicator(x,y) for x in xgrid] for y in ygrid])

# plt.contour(xgrid, ygrid, grid, (-1.0,0.0,1.0), colors=('red','black','blue'), linewidths=(1,3,1))
