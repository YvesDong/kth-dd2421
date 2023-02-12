import numpy as np

# definitions of kernels
def linKernel(vector1, vector2):
    return np.dot(vector1, vector2)

def polyKernel(vector1, vector2, p):
    return (np.dot(vector1, vector2) + 1)**p

def RBFKernel(vector1, vector2, sigma):
    norm = np.linalg.norm(vector1 - vector2)
    return np.exp(-norm/(2*sigma**2))

# kernel matrix calculations
def KernelMatrix(N, targets, inputs, kerneltype, sigma, p):

    global P
    P = np.zeros((N,N))

    match kerneltype:
        case "linKernel":
            for i in range(N):
                for j in range(N):
                    P[i,j] = targets[i]*targets[j]*linKernel(inputs[i], inputs[j])

        case "polyKernel":
            for i in range(N):
                for j in range(N):
                    P[i,j] = targets[i]*targets[j]*polyKernel(inputs[i], inputs[j], p)

        case "RBFKernel":
            for i in range(N):
                for j in range(N):
                    P[i,j] = targets[i]*targets[j]*RBFKernel(inputs[i], inputs[j], sigma)

        case _:
            print("No viable kernel type defined!")
            exit()


def objective(alpha):
    result = 1/2*np.dot(np.dot(P,alpha),alpha) - np.sum(alpha)
    return result

def threshold(data, C, kerneltype, sigma, p):

    if C != None:
        for i in range(len(data)):
            if data[i][0] > C:
                data.pop(i) 
                
    alpha = [i[0] for i in data]
    t_sv = [i[2] for i in data]
    x_sv = [i[1] for i in data]
    
    b = 0 

    match kerneltype:
        case "linKernel":
            for i in range(len(x_sv)):
                b = b + alpha[i]*t_sv[i]*linKernel(x_sv[0],x_sv[i])

        case "polyKernel":
            for i in range(len(x_sv)):
                b = b + alpha[i]*t_sv[i]*polyKernel(x_sv[0],x_sv[i], p)

        case "RBFKernel":
            for i in range(len(x_sv)):
                b = b + alpha[i]*t_sv[i]*RBFKernel(x_sv[0],x_sv[i], sigma)
    
    return b - t_sv[0]

def indicator(x, y, supportvectors, b, kerneltype, sigma, p):

    alpha = [i[0] for i in supportvectors]
    t_sv = [i[2] for i in supportvectors]
    x_sv = [i[1] for i in supportvectors]
    
    datapoint = [x,y]

    ind = 0

    match kerneltype:
        case "linKernel":
            for i in range(len(x_sv)):
                ind = ind + alpha[i]*t_sv[i]*linKernel(datapoint, x_sv[i])

        case "polyKernel":
            for i in range(len(x_sv)):
                ind = ind + alpha[i]*t_sv[i]*polyKernel(datapoint, x_sv[i], p)

        case "RBFKernel":
            for i in range(len(x_sv)):
                ind = ind + alpha[i]*t_sv[i]*RBFKernel(datapoint, x_sv[i], sigma)
    
    return ind - b