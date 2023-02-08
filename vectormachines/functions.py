import numpy as np

# definitions of kernels
def linKernel(vector1, vector2):
    return np.dot(vector1, vector2)

def polyKernel(vector1, vector2, p):
    return (np.dot(vector1, vector2) + 1)^p

def RBFKernel(vector1, vector2, sigma):
    norm = np.linalg.norm(vector1 - vector2)
    return np.exp(-norm/(2*sigma^2))


def kernelMatrix(N, targets, inputs):
    global P
    P = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            P[i,j] = targets[i]*targets[j]*linKernel(inputs[i], inputs[j])
    return P

def objective(alpha):
    result = 1/2*np.dot(np.dot(P,alpha),alpha) - np.sum(alpha)
    return result

def zerofun(alpha):
    return np.dot(alpha,targets)

