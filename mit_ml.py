# Author:Manni
# -*- coding = utf-8 -*-
# @Time :9:57 PM
# @Author:manniau
# @Site : 
# @File : mit_ml.py
# @Software: PyCharm

import numpy as np

# def randomization(n):
#     A=np.random.random([n,1])
#     return A
#
#
# if __name__ == '__main__':
#     n=int(input('Enter a number: '))
#     if n>=0:
#         r=randomization(n)
#         print(r)

# def neural_networks(inputs,weights):
#     '''
#     Takes an input vector and runs it through a 1-layer neural network
#      with a given weight matrix and returns the output.
#
#     :param inputs: 2 x 1 NumPy array
#     :param weights: 2 x 1 NumPy array
#     :return: out - a 1 x 1 NumPy array, representing the output of the neural network
#     '''
#
#     z = np.tanh(weights.T.dot(inputs))
#
#     return z

def get_sum_metrics(predictions, metrics=[]):
    if metrics is None:
        metrics=[]
    for i in range(0,3):
        metrics.append(lambda x:x+i)

    sum_metrics = 0
    for metric in metrics:
        sum_metrics += metric(predictions)

    return sum_metrics

