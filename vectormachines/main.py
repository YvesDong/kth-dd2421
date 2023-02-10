import functions as fun
import numpy as np
import random, math
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def zerofun(alpha):
    return np.dot(alpha,targets)

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
# plt.show()

# OPTIMIZATION #####################################################################################

# pre compute kernel matrix P
p = 2 #polynomial order if kerneltype is poly
fun.kernelMatrix(N, targets, inputs)  # kernel type to be changed in function definition

C = None # slack variable
start = np.zeros(N)
B = [(0, C) for b in range(N)]
XC = {'type':'eq', 'fun':zerofun}

ret = minimize(fun.objective, start, bounds=B, constraints=XC)
if (ret['success'] == True):
    print("Optimization successful!")
else:
    print("Optimization failed!")
    exit()

# extract support vectors and calculate threshold 
alpha = ret['x']
supportvectors = []

for i in range(N):
    if (alpha[i] > pow(10,-5)):
        # print(alpha[i], inputs[i], targets[i])
        supportvectors.append([alpha[i], inputs[i], targets[i]])

b = fun.threshold(supportvectors, C)
print("threshold b = ", b)

# PLOT DECISION BOUNDARIES AND DATAPOINTS ##########################################################
xgrid = np.linspace(-5, 5)
ygrid = np.linspace(-4, 4)

grid = np.array([[fun.indicator(x, y, supportvectors, b) for x in xgrid] for y in ygrid])

plt.contour(xgrid, ygrid, grid, (-1.0,0.0,1.0), colors=('red','black','blue'), linewidths=(1,3,1))
plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
plt.axis('equal')
plt.show()