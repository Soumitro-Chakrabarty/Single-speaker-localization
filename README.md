# Single-speaker-localization

The files here are related to the single source localization method presented in "Broadband DOA estimation with convolutional neural networks trained using noise signals." WASPAA 2017. 

However, there are a few differences from the acoustic and array geometry setup described in the paper. Some of the main differences that should be kept in mind before trying to run the code is as follows:

- The inter-microphone distance is now changed to 0.08 m. This facilitates experiments with Measured RIRs from the Multichannel Impulse Response dataset from Bar-Ilan, with one trained model.
