from sklearn.datasets import fetch_openml
import numpy as np

mnist_X, mnist_y = fetch_openml('mnist_784', version=1, data_home=".", return_X_y=True)

x_all = mnist_X.astype(np.float32) / 255
y_all = mnist_y.astype(np.int32)