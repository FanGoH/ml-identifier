import numpy as np

test_imgs = np.load("kmnist-test-imgs.npz",  allow_pickle=True)['arr_0']

print(test_imgs)