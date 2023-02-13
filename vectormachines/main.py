import functions as fun
import numpy as np
import random, math
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# GENERATE DATAPOINTS ###############################################################################
def generate_data(Num, pos1=[-1.5, 0.5]):
    # Num = 20 # number of data points
    # np.random.seed(100)
    classA = np.concatenate((np.random.randn(int(Num/2), 2)*0.5 + [1.5, 0.5], np.random.randn(int(Num/2), 2)*0.5 + pos1))
    classB = np.random.randn(Num, 2)*0.5 + [0.0, -0.5]

    inputs = np.concatenate((classA, classB))
    targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))

    N = inputs.shape[0] # Number of rows (samples)

    permute = list(range(N))
    random.shuffle(permute)
    inputs = inputs[permute,:]
    targets = targets[permute]

    return inputs, targets, classA, classB

# OPTIMIZATION #####################################################################################

def training(kernel="linKernel", p=2, sigma=1):

    def zerofun(alpha):
        return np.dot(alpha, targets)

    fun.KernelMatrix(2*Num, targets, inputs, kernel, sigma, p)  

    start = np.zeros(2*Num)
    B = [(0, C) for b in range(2*Num)]
    XC = {'type':'eq', 'fun':zerofun}

    ret = minimize(fun.objective, start, bounds=B, constraints=XC)

    if (ret['success'] == True):
        print("Optimization successful!")
    else:
        print("Optimization failed!")
        exit()
    alpha = ret['x']

    return alpha

# extract support vectors and calculate threshold ##########################################################
def extractSV(C=None, kernel="linKernel", p=2, sigma=1):
    supportvectors = []

    for i in range(2*Num):
        if (alpha[i] > pow(10,-5)):
            # print(alpha[i], inputs[i], targets[i])
            supportvectors.append([alpha[i], inputs[i], targets[i]])
    print("number of non-zero alpha: ", len(supportvectors))

    b = fun.threshold(supportvectors, C, kernel, sigma, p)
    print("threshold b = ", b)

    return supportvectors, b

# PLOT DECISION BOUNDARIES AND DATAPOINTS ##########################################################
def plot_results(axs, kernel="linKernel", p=2, sigma=1):
    xgrid = np.linspace(-3, 3)
    ygrid = np.linspace(-3, 3)
    grid = np.array([[fun.indicator(x, y, supportvectors, b, kernel, sigma, p) for x in xgrid] for y in ygrid])

    axs.contour(xgrid, ygrid, grid, (-1.0,0.0,1.0), colors=('red','black','blue'), linewidths=(1,3,1))
    axs.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
    axs.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
    axs.set_title("sigma in RBFK: {}, % pos alpha: {}".format(sigma, "%.2f" % (len(supportvectors)/(2*Num))))
    # axs.set_title("# data: {}, % pos alpha: {}".format(2*Num, "%.2f" % (len(supportvectors)/(2*Num))))
    # plt.axis('equal')


if __name__ == "__main__":
    # hyperparameters
    Num = 100 # no of data for each of the class
    C = 10 # slack variable
    kList = ['linKernel', 'polyKernel', 'RBFKernel']
    kernel = kList[2]
    p = 3
    sigma = 1
    pos1 = [-1.5, 0.5]

    ## exploring and reporting
    # numList = [20, 80, 160, 320] # discussion 1 - change Num
    # pos1List = [[-1.5, 0.5], [-.5, 0.4], [.5, 0.3], [-1, 0.2]] # discussion 1 - change cluster position
    # pList = [1,2,3,4] # discusstion 3 - p in polyKernel
    sigmaList = [.5,.7,2,5] # discusstion 3 - p in polyKernel
    # cList = [.01, .08, .2, 3] # discusstion 4 - C in slack variable

    inputs, targets, classA, classB = generate_data(Num, pos1)
    fig, axs = plt.subplots(2, 2) # plots
    for i in range(4):
        # Num = numList[i]
        # pos1 = pos1List[i]
        # p = pList[i]
        sigma = sigmaList[i]
        # C = cList[i]

        # optimization
        alpha = training(kernel=kernel, p=p, sigma=sigma)
        supportvectors, b = extractSV(C=C, kernel=kernel, p=p, sigma=sigma)

        # plots added in https://docs.google.com/presentation/d/1KMSgdGcQlxe5VYL4EcEQJhvyfu7i9W2UMSrSK3vZ3-o/edit?usp=sharing
        idx1, idx2 = [0,0,1,1], [0,1,0,1]
        plot_results(axs[idx1[i], idx2[i]], kernel=kernel, p=p, sigma=sigma)
    fig.tight_layout()
    plt.show()

