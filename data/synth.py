import numpy as np

epsilon = 1

data = np.load('cifar-10.npz')

x = data['arr_0'].astype(np.float32)[50000:]
y = data['arr_1'][50000:]

data_grad = np.load('adversarial_example_signs.npy').astype(np.float32)
x += data_grad * epsilon
print(x[0])

print(x.shape)
print(y.shape)
np.savez('adversarial_examples.npz', x, y)