### Neural network for predicting Formation Amplitudes of alpha particles for given nuclei. 

__author__ = "Daniel Karlsson"
__copyright__ = ""
__version__ = "1.0"
__maintainer__ = "Daniel Karlsson"
__email__ = "carlzone@kth.se"
__status__ = "In development"

import pandas as pd
import numpy as np
np.set_printoptions(precision = 10, threshold = 10000)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization

from keras import backend as K
from keras import callbacks

from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.optimizers import Adam

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def root_mean_squared_error(y_true, y_pred):    
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def predictionlist(nuclei): ## Create a list of [A,Z] pairs to predict formation amplitude for, this selectes all possible from proton to neutron driplines.
    predlist = []
    zdriplines = pd.read_fwf('Data/result-xu.txt', header = 1, names = ['Z', 'N', 'A', 'Etot', 'Eshell+pairing', 'beta2', 'gamma', 'beta4'], widths = [5,5,5,10,10,12,12,10]).fillna('').sort_values(by =['Z'])
    if nuclei == 'ee':
        step = 2
    else:
        step = 1
    for Z in range(min(zdriplines.Z), max(zdriplines.Z) + step,step):
        if Z%2 == 0:
            Nmin = min((min(zdriplines[zdriplines['Z'] == Z].Z)),(min(zdriplines[zdriplines['Z'] == Z].N) - 2))
            Nmax = max(zdriplines[zdriplines['Z'] == Z].N) + 2
        else:
            Nmin = min((min(zdriplines[zdriplines['Z'] == Z + 1].Z)),(min(zdriplines[zdriplines['Z'] == Z + 1].N) - 2))
            Nmax = max(zdriplines[zdriplines['Z'] == Z+1].N)
        for N in range(Nmin, Nmax + step,step):
            A = Z + N
            if nuclei == 'oe':
                if A%2 == 0:
                    continue
            elif nuclei == 'oo':
                if Z%2==0 or N%2==0:
                    continue
            predlist.append([A,Z])
    return predlist, True
 
device = '/CPU:0'
#device = '/GPU:0'

nuclei = 'ee'                   # Choose which nuclei to predict, ee = even-even, oe = odd-even (and even-odd), oo = odd-odd, all = all
transition = 'gs-gs-0'          # Ground state to ground state, l = 0
info = 'log'                    # Choose to predict log value log_10(RFR) or '' for F(R), they use different models but are comparable in accuracy.
totalmodels = 30
choosemodels = 20
usemodels = nuclei
#usemodels = False              #Set False to train to data
predict = False
if usemodels:
    modellist = pd.read_excel('Results/' + transition + '/' + nuclei +  '/FR/goodmodels_'+ info + '.xlsx')      #load list of good and bad models
    goodmodels = modellist[modellist['use'] == 'yes'].model.values                                              #Select only models marked as good

predlist, predict = predictionlist(nuclei)                                                                      #Sets predict to true for chosen array of [A,Z]

#Y = df_small['logRF(R)'].astype(float)                                                                         # Insert array of Experimental data of formation amplitudes here
#X = []                                                                                                         #insert array of [A,Z] or more features to train, for which you have experimental data.

X = np.array(predlist)
A = np.array([X[i][0] for i in range(len(X))])
Z = np.array([X[i][1] for i in range(len(X))])
N = A - Z

#x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)                                       #split data into training and validation set

models = list()
if  __name__ == "__main__":    
    with tf.device(device):
        #tf.random.set_seed(1337)
        if not usemodels:
            for n in range(totalmodels):
                model = Sequential()
                model.add(Dense(32, input_dim= 4, activation='relu'))   # Input Neural Network Layer 1
                model.add(BatchNormalization())                         # Normalize batch of data
                model.add(Dense(8, activation='relu'))                  # Hidden Neural Network Layer 2
                model.add(Dense(1, activation='linear'))                # Hidden Neural Network Layer 3 - Final Output layer
                
                opt = Adam(learning_rate = 0.0001)
                model.compile(loss= root_mean_squared_error, optimizer=opt, metrics = root_mean_squared_error)

                model.fit(x_train, y_train, epochs= 2000, batch_size= 35, validation_data=(x_test, y_test))
                scores = model.evaluate(X, Y)
                print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100),n)
                model.save('Results/' + transition + '/' + nuclei + '/FR/' + nuclei + '_' + str(n) + '_' + info + '.h5')                #save models to file after training
                model.save_weights('Results/' + transition + '/' + nuclei + '/FR/' + nuclei + 'weights' + str(n) + '_' + info + '.h5')  #save weights of models into files after training
                models.append(model) 
        else:
            models = [load_model('Results/' + transition + '/'+ usemodels + '/FR/' + usemodels + '_' + str(m) + '_' + info + '.h5', custom_objects={'root_mean_squared_error': root_mean_squared_error}) for m in range(totalmodels) if m in goodmodels]

    yhats = [np.array([x[0] for x in model.predict(X)]) for model in models]
    
    print('Number of nuclear data: ',len(yhats[0]))
    print(X, yhats)
    
