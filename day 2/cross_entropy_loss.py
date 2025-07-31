import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
    r=1/(1+np.exp(-x))
    return r
y_true=np.array([1,0,1,0,0,1,1])
y_pred=np.array([0.9,0.1,0.8])
def cross_entropy_loss(y_true, y_pred):
    y_pred=sigmoid(y_pred)
    loss=np.zeros(len(y_pred))
    for i in range(len(y_pred)):
        loss[i]=-1*(y_true[i]*np.log(y_pred[i]) + (1-y_true[i])*np.log(1-y_pred[i]))
        return loss
    
r=cross_entropy_loss(y_true, y_pred)
print(r)

a=np.linspace(0,len(r)-1,len(r))
plt.plot(a, r)
plt.show()