# Machine Learning Models

import numpy as np

class UnivariateLinearRegression():

    def get_wb0():
        file = open("wb0.txt", "r")
        wb = file.readlines()
        w = float(wb[0].strip())
        b = float(wb[1].strip())
        file.close()
        return w, b
    
    w,b = get_wb0()

    def predict(self, x_test):
        return x_test * self.w + self.b
    
class MultivariateLinearRegression():
    
    def get_wb1():
        file = open("wb1.txt", "r")
        wb = file.readlines()
        w_arr = wb[0].strip().split()
        w = np.array([float(x) for x in w_arr])
        b = float(wb[1].strip())
        file.close()
        return w, b
    
    w,b = get_wb1()

    def predict(self, x_test):
        return np.dot(x_test, self.w) + self.b

class LogisticRegression():

    def get_wb2():
        file = open("wb2.txt", "r")
        wb = file.readlines()
        num1 = float(wb[0].strip())
        num2 = float(wb[1].strip())
        w = np.array([[num1], [num2]])
        num3 = float(wb[2].strip())
        b = np.array([num3])
        file.close()
        return w, b
    
    w,b = get_wb2()

    def predict(self, x_test):
        m = x_test.shape[0]
        p = np.zeros(m)
        for i in range(m):   
            z_i = np.dot(x_test[i], self.w) + self.b
            f_wb = 1 / (1 + np.exp(-z_i))
            p[i] = f_wb >= 0.5
        return p