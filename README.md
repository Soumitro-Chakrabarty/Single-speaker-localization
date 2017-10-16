# Single-speaker-localization

The files here are related to the single source localization method presented in **"Broadband DOA estimation with convolutional neural networks trained using noise signals." WASPAA 2017.** 

However, there are a few differences from the acoustic and array geometry setup described in the paper. Some of the main differences that should be kept in mind before trying to run the code is as follows:

- The inter-microphone distance is 0.08 m. This facilitates experiments with Measured RIRs from the Multichannel Impulse Response dataset from Bar-Ilan, with a single trained model.

- The STFT length was also modified to 512 samples, thereby giving a feature rate of 16 ms. 

- The phase map dimensions are: 4x257

Example test code and furhter instructions on python package requriements *Coming Soon*
