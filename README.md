# Single-speaker-localization

The trained models are related to the single source localization method presented in 

**"Broadband DOA estimation with convolutional neural networks trained using noise signals." WASPAA 2017.** 

However, there are a few differences from the acoustic and array geometry setup described in the paper. Some of the main differences that should be kept in mind before trying to run the code is as follows:

- The inter-microphone distance is 0.08 m. This facilitates experiments with Measured RIRs from the Multichannel Impulse Response dataset from Bar-Ilan, with a single trained model.

- The STFT window length was also modified to 512 samples, thereby giving a feature rate of 16 ms. 

- The phase map dimensions are: 4x256

The model was trained using Keras with Theano backend. 

An example code to load the model and weights, and run it with a test file named "DOA_test":

```
from __future__ import print_function
import numpy as np
import h5py
from keras.models import model_from_json, load_model

# Load trained model from the json file

  json_file = open('Model_sin_CNN.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)

# Load weights 
  
  loaded_model.load_weights('Weights_sin_CNN.h5')
  print("Loaded model from disk")
  
# Define the optimizer and compile the model

  lrate = 0.001
  adam = Adam(lr =lrate,beta_1=0.9,beta_2=0.999,epsilon=1e-08)
  loaded_model.compile(loss='categorical_crossentropy', optimizer= adam)
  
# Read test data from an hdf5 file called DOA_test

  Test_data = h5py.File('DOA_test.hdf5')
  X_test = Test_data['features']                   # These are the phase maps for each time frame
  Y_test = Test_data['targets']

# Convert to numpy arrays

  X_test = np.array(X_test)                        # size = (Number of time frames,1,256,4)
  Y_test = np.array(Y_test)                        # size = (Number of time frames,37)

# Estimate DOA for each time frame in the test set 

  Output = loaded_model.predict(X_test)           
  
```


