import numpy as np

def load_data():
    print('Loading Training Data...')
    XY_mat = np.loadtxt('../src/training_data/coordinate.txt')
    EToV = np.loadtxt('../src/training_data/EToV.txt')
    data_test = np.loadtxt('../src/training_data/initial_cond_0.txt').T.reshape(1,401,1536)
    data_training = np.zeros((49,401,1536))
    for ic in range(1,50):
        data_i = np.loadtxt('../src/training_data/initial_cond_' + str(ic) + '.txt').T
        data_training[ic-1,:,:] = data_i
    print('Data Loaded!')
    print('Training Data Shape: ' + str(data_training.shape))
    print('Test Data Shape: ' + str(data_test.shape))
    return XY_mat, EToV, data_training, data_test

def load_test():
    print('Loading Test Data...')
    XY_mat = np.loadtxt('../src/training_data/coordinate.txt')
    EToV = np.loadtxt('../src/training_data/EToV.txt')
    data_test = np.loadtxt('../src/training_data/initial_cond_0.txt').T.reshape(1,401,1536)
    print('Data Loaded!')
    print('Test Data Shape: ' + str(data_test.shape))
    return XY_mat, EToV, data_test
