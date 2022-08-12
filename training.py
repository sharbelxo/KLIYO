# model training

import copy
import math
import numpy as np
import pandas as pd

class UnivariateLinearRegressionTraining():

    x_train = np.array([ 6.1101,  5.5277,  8.5186,  7.0032,  5.8598,  8.3829,  7.4764,
        8.5781,  6.4862,  5.0546,  5.7107, 14.164 ,  5.734 ,  8.4084,
        5.6407,  5.3794,  6.3654,  5.1301,  6.4296,  7.0708,  6.1891,
       20.27  ,  5.4901,  6.3261,  5.5649, 18.945 , 12.828 , 10.957 ,
       13.176 , 22.203 ,  5.2524,  6.5894,  9.2482,  5.8918,  8.2111,
        7.9334,  8.0959,  5.6063, 12.836 ,  6.3534,  5.4069,  6.8825,
       11.708 ,  5.7737,  7.8247,  7.0931,  5.0702,  5.8014, 11.7   ,
        5.5416,  7.5402,  5.3077,  7.4239,  7.6031,  6.3328,  6.3589,
        6.2742,  5.6397,  9.3102,  9.4536,  8.8254,  5.1793, 21.279 ,
       14.908 , 18.959 ,  7.2182,  8.2951, 10.236 ,  5.4994, 20.341 ,
       10.136 ,  7.3345,  6.0062,  7.2259,  5.0269,  6.5479,  7.5386,
        5.0365, 10.274 ,  5.1077,  5.7292,  5.1884,  6.3557,  9.7687,
        6.5159,  8.5172,  9.1802,  6.002 ,  5.5204,  5.0594,  5.7077,
        7.6366,  5.8707,  5.3054,  8.2934, 13.394 ,  5.4369])
    
    y_train = np.array([17.592  ,  9.1302 , 13.662  , 11.854  ,  6.8233 , 11.886  ,
        4.3483 , 12.     ,  6.5987 ,  3.8166 ,  3.2522 , 15.505  ,
        3.1551 ,  7.2258 ,  0.71618,  3.5129 ,  5.3048 ,  0.56077,
        3.6518 ,  5.3893 ,  3.1386 , 21.767  ,  4.263  ,  5.1875 ,
        3.0825 , 22.638  , 13.501  ,  7.0467 , 14.692  , 24.147  ,
       -1.22   ,  5.9966 , 12.134  ,  1.8495 ,  6.5426 ,  4.5623 ,
        4.1164 ,  3.3928 , 10.117  ,  5.4974 ,  0.55657,  3.9115 ,
        5.3854 ,  2.4406 ,  6.7318 ,  1.0463 ,  5.1337 ,  1.844  ,
        8.0043 ,  1.0179 ,  6.7504 ,  1.8396 ,  4.2885 ,  4.9981 ,
        1.4233 , -1.4211 ,  2.4756 ,  4.6042 ,  3.9624 ,  5.4141 ,
        5.1694 , -0.74279, 17.929  , 12.054  , 17.054  ,  4.8852 ,
        5.7442 ,  7.7754 ,  1.0173 , 20.992  ,  6.6799 ,  4.0259 ,
        1.2784 ,  3.3411 , -2.6807 ,  0.29678,  3.8845 ,  5.7014 ,
        6.7526 ,  2.0576 ,  0.47953,  0.20421,  0.67861,  7.5435 ,
        5.3436 ,  4.2415 ,  6.7981 ,  0.92695,  0.152  ,  2.8214 ,
        1.8451 ,  4.2959 ,  7.2029 ,  1.9869 ,  0.14454,  9.0551 ,
        0.61705])
    
    def compute_cost(self, x, y, w, b): 
        m = x.shape[0] 
        total_cost = 0
        for i in range(m):
            f_wb = w * x[i] + b
            cost_wb = (f_wb - y[i]) ** 2
            total_cost += cost_wb
        total_cost = total_cost / (2 * m)
        return total_cost

    def compute_gradient(self, x, y, w, b): 
        m = x.shape[0]
        dj_dw = 0
        dj_db = 0
        for i in range(m):
            f_wb = w * x[i] + b
            dj_db = dj_db + (f_wb - y[i])
            dj_dw = dj_dw + ((f_wb - y[i]) * x[i])
        dj_db = dj_db / m
        dj_dw = dj_dw / m
        return dj_dw, dj_db
    
    def gradient_descent(self, x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
        m = len(x)
        J_history = []
        w_history = []
        w = copy.deepcopy(w_in)
        b = b_in
        for i in range(num_iters):
            dj_dw, dj_db = gradient_function(x, y, w, b )  
            w = w - alpha * dj_dw               
            b = b - alpha * dj_db               
            if i<100000:
                cost =  cost_function(x, y, w, b)
                J_history.append(cost)
            if i% math.ceil(num_iters/10) == 0:
                w_history.append(w)
                print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        return w, b, J_history, w_history
    
    def run_gradient_descent(self):
        initial_w = 0.
        initial_b = 0.
        iterations = 1500
        alpha = 0.01
        w,b,_,_ = self.gradient_descent(self.x_train ,self.y_train, initial_w, initial_b, self.compute_cost, self.compute_gradient, alpha, iterations)
        return w,b

smoking = {"no": 0, "yes": 1}
class MultivariateLinearRegressionTraining():
    
    data = pd.read_csv(r"DATASET-LOCATION")
    rows = data.shape[0]
    columns = data.shape[1]
    max_age = float(data["age"].max())
    max_bmi = float(data["bmi"].max())
    max_children = float(data["children"].max())
    max_charges = float(data["charges"].max())
    smokers = data[(data.smoker == "yes")]
    non_smokers = data[(data.smoker == "no")]
    data = data.dropna()
    data.drop(["region", "sex"], axis = 1, inplace = True)
    smoking = {"no": 0, "yes": 1}
    data["smoker"] = data["smoker"].apply(lambda x: smoking[x])

    normalize_data0 = data.divide(data.max())

    # normalizing by z-score
    normalize_data1 = (data - data.mean()).divide(data.std())

    # normalizing by mean
    normalize_data2 = (data - data.mean()).divide(data.max() - data.min())

    data = normalize_data0

    x_train = data[["age", "bmi", "children", "smoker"]]
    y_train = data["charges"]

    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()

    w_initial = np.zeros(shape = 4)
    b_initial = 0.

    def compute_cost(self, X, y, w, b):
        m = X.shape[0]
        cost = 0.0
        for i in range(m):
            f_wb_i = np.dot(X[i], w) + b
            cost = cost + (f_wb_i - y[i]) ** 2
        cost = cost / (2 * m)
        return cost

    def compute_gradient(self, X, y, w, b):
        m,n = X.shape
        dj_dw = np.zeros((n,))
        dj_db = 0.
        for i in range(m):
            err = (np.dot(X[i], w) + b) - y[i]
            for j in range(n):
                dj_dw[j] = dj_dw[j] + err * X[i, j]
            dj_db = dj_db + err
        dj_dw = dj_dw / m
        dj_db = dj_db / m
        return dj_db, dj_dw

    def gradient_descent(self, X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
        J_history = []
        w = copy.deepcopy(w_in)
        b = b_in
        for i in range(num_iters):
            dj_db,dj_dw = gradient_function(X, y, w, b)
            w = w - alpha * dj_dw
            b = b - alpha * dj_db
            if i<100000:
                J_history.append( cost_function(X, y, w, b))
            if i% math.ceil(num_iters / 10) == 0:
                print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        return w, b, J_history

    def run_gradient_descent(self):
        
        initial_w = np.zeros(shape = 4)
        initial_b = 0.
        iterations = 1000
        alpha = 0.001
        w_final, b_final,_ = self.gradient_descent(self.x_train, self.y_train, initial_w, initial_b,
                                                        self.compute_cost, self.compute_gradient, 
                                                        alpha, iterations)
        print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
        m,_ = self.x_train.shape
        for i in range(m):
            print(f"prediction: {np.dot(self.x_train[i], w_final) + b_final:0.2f}, target value: {self.y_train[i]}")
        return w_final, b_final

class LogisticRegressionTraining():

    df = pd.read_csv(r"DATASET-LOCATION")
    df = df[["math score","reading score"]]
    df.rename(columns = {"math score" : "Exam 1", "reading score" : "Exam 2"}, inplace = True)
    df["Admission"] = 0
    for index, row in df.iterrows():
        if (row["Exam 1"] + row["Exam 2"]) / 2 > 60:
            row["Admission"] = 1
    exam_grades = df[["Exam 1", "Exam 2"]]
    x_train = exam_grades.to_numpy()
    y_train = df["Admission"].to_numpy()

    def sigmoid(self, z):
        g = 1 / (1 + np.exp(-z))
        return g

    def compute_cost(self, X, y, w, b, lambda_= 1):
        m = X.shape[0]
        cost = 0
        for i in range(m):
            z_i = np.dot(X[i], w) + b
            f_wb_i = self.sigmoid(z_i)
            cost += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)
        cost /= m
        return cost

    def compute_gradient(self, X, y, w, b, lambda_=None): 
        m, n = X.shape
        dj_dw = np.zeros(w.shape)
        dj_db = 0.
        for i in range(m):
            z_i = np.dot(X[i], w) + b
            f_wb = self.sigmoid(z_i)
            dj_db_i = f_wb - y[i]
            dj_db += dj_db_i
            for j in range(n):
                dj_dw_ij = (f_wb - y[i])* X[i][j]
                dj_dw[j] += dj_dw_ij      
        dj_dw = dj_dw / m
        dj_db = dj_db / m
        return dj_db, dj_dw

    def gradient_descent(self, X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_):
        J_history = []
        w_history = []
        for i in range(num_iters):
            dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)   
            w_in = w_in - alpha * dj_dw               
            b_in = b_in - alpha * dj_db              
            if i<100000:
                cost =  cost_function(X, y, w_in, b_in, lambda_)
                J_history.append(cost)
            if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
                w_history.append(w_in)
                print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        return w_in, b_in, J_history, w_history

    def run_gradient_descent(self):
        np.random.seed(1)
        initial_w = 0.01 * (np.random.rand(2).reshape(-1,1) - 0.5)
        initial_b = -8
        iterations = 10000
        alpha = 0.001
        w,b,_,_ = self.gradient_descent(self.x_train ,self.y_train, initial_w, initial_b, 
                                    self.compute_cost, self.compute_gradient, alpha, iterations, 0)
        return w,b

def wb0():
    w,b = UnivariateLinearRegressionTraining().run_gradient_descent()
    file = open("wb0.txt","w")
    file.write(str(w) + "\n" + str(b))
    file.close()

def wb1():
    w,b = MultivariateLinearRegressionTraining().run_gradient_descent()
    w_str = str(w).replace(' [', '').replace('[', '').replace(']', '')
    file = open("wb1.txt","w")
    file.write(str(w_str) + "\n" + str(b))
    file.close()

def wb2():
    w,b = LogisticRegressionTraining().run_gradient_descent()
    w_str = str(w).replace(' [', '').replace('[', '').replace(']', '')
    b_str = str(b).replace(' [', '').replace('[', '').replace(']', '')
    file = open("wb2.txt","w")
    file.write(str(w_str) + "\n" + str(b_str))
    file.close()

# train the models you want and comment the others:

# wb0()
# wb1()
# wb2()