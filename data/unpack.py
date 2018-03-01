import pickle
import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']
data = []
labels = []

for file in files:
	d = unpickle(file)
	data.append(d[b'data'])
	labels.append(np.array([d[b'labels']]).T)

data = np.vstack(data)
labels = np.vstack(labels)
print(data.shape)
print(labels.shape)

output_filename = 'cifar-10.npz'
np.savez(output_filename, data, labels)
print('wrote to ' + output_filename)