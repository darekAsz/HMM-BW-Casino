# -*- coding:utf-8 -*-
# Filename: test_weather.py
# Author：hankcs
# Date: 2016-08-06 PM6:04
import numpy as np

import hmm
import random


A = np.array([[0.5, 0.5], [0.5, 0.5]])
B = np.array([[0.16, 0.16, 0.16, 0.16, 0.16, 0.16], [0.16, 0.16, 0.16, 0.16, 0.16, 0.16]])
pi = np.array([0.5, 0.5])
h = hmm.HMM(A, B, pi)

# print observations_data
# print states_data
for i in range(100):
    size = 100
    observations_data = np.empty([size], dtype=int)
    for j in range(size):
        rand = random.randint(1, 100)
        if rand <= 10:
            observations_data[j] = 0
        elif rand <= 20:
            observations_data[j] = 1
        elif rand <= 30:
            observations_data[j] = 2
        elif rand <= 40:
            observations_data[j] = 3
        elif rand <= 50:
            observations_data[j] = 4
        else:
            observations_data[j] = 5
    # print observations_data
    guess = hmm.HMM(A, B, pi)
    A, B, pi = guess.baum_welch_train(observations_data)

print(B)
