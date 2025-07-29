import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
  return 1/(1+np.exp(-x))

def tanh(x):
  return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def ReLU(x):
  return np.maximum(0,x)

def softplus(x):
  return np.log(1+np.exp(x))
x=np.linspace(-2,2,100)
plt.plot(x,sigmoid(x),label='Sigmoid')
plt.plot(x,tanh(x),label='Tanh')
plt.plot(x,ReLU(x),label='ReLU')
plt.plot(x,softplus(x),label='Softplus')
plt.legend()
plt.show()

