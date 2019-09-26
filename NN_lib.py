import numpy as np


class Sigmoid():
    def __call__(self, x):
        """
        x : 順伝搬入力
        x を 0-1 に変換します
        """
        # オーバーフローを避けるため、以下のように実装
        # if x > 0 :  1   / ( 1 + e(-x) )
        # else     : e(x) / ( 1 + e(x)  )
        self.fx = np.exp(np.where(x >= 0, 0, x)) / (1 + np.exp(np.where(x >= 0, -x, x)))
        return self.fx
    
    def backward(self, delta):
        return self.fx * (1 - self.fx) * delta

class Relu():
    def __call__(self, x):
        self.mask = np.where(x > 0, 1, 0)
        return self.mask * x
    
    def backward(self, delta):
        return self.mask * delta

class PRelu():
    def __init__(self, size, alpha=0.25, momentum=0.9):
        self.alpha = alpha * np.ones(size)
        self.lr = 0.01
        self.momentum = 0
        self.v = 0

    def __call__(self, x):
        self.x = x
        self.mask = np.where(x > 0, 1, alpha)
        return self.mask * x
    
    def backward(self, d):
        da = np.where(self.x > 0, 0, delta * self.x)
        self.v *= self.momentum
        self.v += self.lr * da
        self.alpha -= self.v

        return self.mask * d

class BatchNormalization():
    def __init__(self, size):
        self.gamma = np.ones(size, dtype=np.float)
        self.beta = np.zeros(size, dtype=np.float)
        self.dgamma = None
        self.dbeta = None
        self.N = None

    @property
    def update_params(self):
        return {"gamma":self.dgamma, "beta":self.dbeta}

    def __call__(self, x, epsilon=1e-7):
        self.x = x
        self.mean = x.mean(axis=0)
        self.var = x.var(axis=0) + epsilon
        self.norm = (x - self.mean) / np.sqrt(self.var)
        self.N = x.shape[0]
        return self.gamma * self.norm + self.beta
    
    def backward(self, d):
        self.dgamma = (d*self.norm).sum(axis=0)
        self.dbeta = d.sum(axis=0)

        x_mean = self.x - self.mean
        dnorm = self.gamma * d
        delta = dnorm / np.sqrt(self.var)
        dvar = (dnorm * x_mean).sum(axis=0) * -0.5 / (self.var ** 1.5)
        dmean = -delta.sum(axis=0) + dvar * (-2 * x_mean).sum(axis=0) / self.N
        return delta + (self.var * 2 * x_mean + dmean) / self.N

def softmax(x, axis=1):
    """
    x.shape -> (B, C)
    B : Batch_size
    C : Class_number
    """
    # オーバーフロー対策 -x.max()
    exp_x = np.exp(x - x.max(axis=axis)[..., None])
    # [..., None or np.newaxis]は1次元増やす
    # [None, ...].shape -> (1, N), [..., None].shape -> (N, 1)
    return exp_x / np.sum(exp_x, axis=axis)[..., None]

def softmax_closs_entropy(y, t):
    """
    誤差関数
    softmax_closs_entropy の微分は [予測]-[正解] となる
    """
    batch_size = y.shape[0]
    y = softmax(y)
    delta = (y - t) / batch_size
    return y, delta #loss, delta

def loss_and_accracy(y, t):
    """
    y : 予測
    t : 正解
    return loss, accuracy
    """
    N = y.shape[0]
    y = softmax(y)
    loss = -np.sum(t * np.log(y + 1e-7)) / N
    y_index = np.argmax(y, axis=1)
    t_index = np.argmax(t, axis=1)
    accuracy = np.count_nonzero(y_index == t_index) / N
    return loss, accuracy

class FullConnectedLayer():
    def __init__(self, in_size, out_size):
        # Xavier, He
        self.w = np.random.normal(size=(in_size, out_size)) / np.sqrt(in_size) * np.sqrt(2)
        self.b = np.random.normal(size=out_size)
        self.dw = None
        self.db = None

    @property
    def update_params(self):
        return {"w":self.dw, "b":self.db}
        
    def __call__(self, x):
        self.x = x
        return x @ self.w + self.b
        
    def backward(self, d):
        self.dw = self.x.T @ d
        self.db = np.sum(d, axis=0)
        return d @ self.w.T

class SGD():
    def __init__(self, lr=0.01):
        self.lr = lr

    def __call__(self, update_params):
        for key in update_params:
            update_params[key] *= self.lr

    def lr_decay(self, lr):
        self.lr = lr

class MomentumSGD():
    def __init__(self, lr=0.01, momentum=0.9): 
        self.lr = lr
        self.momentum = momentum
        self.v = {}

    def __call__(self, update_params):
        for key, gradient in update_params.items():
            if key not in self.v:
                self.v[key] = np.zeros_like(gradient)
            
            self.v[key] *= self.momentum
            self.v[key] -= self.lr * gradient
            update_params[key] = -self.v[key]

    def lr_decay(self, lr):
        self.lr = lr

class AdaGrad():
    def __init__(self, lr=0.001): 
        self.lr = lr
        self.h = {}

    def __call__(self, key,update_paramsgradient):
        for key, gradient in update_params.items():
            if key not in self.h:
                self.h[key] = np.zeros_like(gradient)

            self.h[key] += gradient ** 2
            update_params[key] = self.lr * gradient / (np.sqrt(self.h[key]) + 1e-8)

    def lr_decay(self, lr):
        self.lr = lr

class RMSprop():
    def __init__(self, lr=0.01, d = 0.99):
        self.lr = lr
        self.d = d
        self.h = {}

    def __call__(self, update_params):
        for key, gradient in update_params.items():
            if key not in self.h:
                self.h[key] = np.zeros_like(gradient)
        
            self.h[key] *= self.d
            self.h[key] += (1 - self.d) * gradient ** 2
            update_params[key] = self.lr * gradient / (np.sqrt(self.h[key]) + 1e-8)

    def lr_decay(self, lr):
        self.lr = lr

class Adam():
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0
        self.m = {}
        self.v = {}

    def __call__(self, update_params):
        self.t += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.t) / (1.0 - self.beta1**self.t)

        for key, gradient in update_params.items():
            if key not in self.m:
                self.m[key] = np.zeros_like(gradient)            
                self.v[key] = np.zeros_like(gradient)

            self.m[key] *= self.beta1
            self.m[key] += (1 - self.beta1) * gradient
            self.v[key] *= self.beta2
            self.v[key] += (1 - self.beta2) * gradient ** 2        
            # m_c = self.m[key] / (1 - self.beta1**self.t)
            # v_c = self.v[key] / (1 - self.beta2**self.t)
            # return lr * m_c / (np.sqrt(v_c) + 1e-8)

            update_params[key] = lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-8)

    def lr_decay(self, lr):
        self.lr = lr

class NN():
    def update(self, optimizer):
        temp = {}
        for k, v in self.__dict__.items():
            temp.update(self.get_params(k, v))
        optimizer(temp)
        for k, v in temp.items():
            self.set_params(k.split("/"), v)

    def get_params(self, k, v):
        if hasattr(v, "update_params"):
            return {k+"/"+vk:vv for vk, vv in v.update_params.items()}
        else:
            for vk, vv in v.__dict__.items():
                return self.get_params(k+"/"+vk, vv)

    def set_params(self, k, v):
        layer = self
        for key in k[:-1]:
            layer = layer.__dict__[key]
        else:
            layer.__dict__[k[-1]] -= v

def bool_to_onehot(x):
    return np.identity(2)[x.astype(np.uint8)]