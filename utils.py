import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def visualize(data, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    _ = ax1.hist(data, bins='auto', cumulative=True)
    _ = ax2.hist(data, bins='auto', cumulative=False)
    fig.savefig(path)      

def normalize(data, path, mode='train'):
    if mode == 'train':
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(np.array(data).reshape(-1, 1)).reshape(1, -1)[0]
        with open(os.path.join(path, 'scaler.pickle'), 'wb') as handle:
            pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return normalized
    elif mode == 'eval':
        scaler = None
        with open(os.path.join(path, 'scaler.pickle'), 'rb') as handle:
            scaler = pickle.load(handle)
        normalized = scaler.fit_transform(np.array(data).reshape(-1, 1)).reshape(1, -1)[0]
        return normalized
    else:
        raise 'mode parameter can only take eval or train as its values'




