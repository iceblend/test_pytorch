# https://wikidocs.net/61073
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, cache=True)

X = mnist.data / 255.0
y = mnist.target

print(X[0])
print(y[0])














