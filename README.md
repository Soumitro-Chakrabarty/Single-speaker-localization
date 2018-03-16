# Single-speaker-localization with CNNs

The repository provides trained models that are related to the CNN based single source localization method presented in the paper

**Title**: [Broadband DOA estimation with convolutional neural networks trained using noise signals](http://ieeexplore.ieee.org/document/8170010/)  
**Authors**: [Soumitro Chakrabarty](https://www.audiolabs-erlangen.de/fau/assistant/chakrabarty), [Emanuël A.P. Habets](https://www.audiolabs-erlangen.de/fau/professor/habets)  
**Conference**: *IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), 2017.*  
[ArXiv version](https://arxiv.org/abs/1705.00919) 

However, there are a few differences from the acoustic and array geometry setup described in the paper. Some of the main differences that should be kept in mind before trying to run the code is as follows:

- The inter-microphone distance is 0.08 m. 

- The STFT window length was modified to 512 samples, thereby giving a feature rate of 16 ms. 

- The phase map dimensions are: 4x256, we exclude the highest frequency sub-band.

A small test dataset, with the features (phase maps) and targets, created by convolving a 13 s long speech signal with Measured RIRs from the [Bar-Ilan Multi-Channel Impulse Response Database](http://www.eng.biu.ac.il/gannot/downloads/) for 9 different angles from the 4 middle microphones in the [8,8,8,8,8,8,8] ULA setup is included (**DOA_test.hdf5**), as well as the output .mat file (**DOA_test_OP.mat**). 

Running the code would generate an output file called ***DOA_OP.mat*** and it should be the same as **DOA_test_OP.mat**. 

In addition a MATLAB script to visualize the output is also provided.  

The acoustic setup for the provided test data is as follows:

  - Reverberation time = 0.610 s
  - Source-array distance = 2 m
  - SNR = 30 dB (Spatially white Gaussian noise)
  - Fs= 16 kHz

### Usage

The python dependencies can be installed by using the requirements file 

```
pip install -r requirements.txt
```
You can now run the script
```
python cnn_test_github.py
```

### Citation

If you find the provided model useful in your research, please cite:

```
@INPROCEEDINGS{Chakrabarty2017a
	author = {S. Chakrabarty and E. A. P. Habets},
	title = {Broadband DOA Estimation Using Convolutional Neural Netowrks Trained with Noise signals},
	booktitle = {IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)},
	year = {2017},
	month = {Oct.}
}
```
