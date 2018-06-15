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

Please note that the angle convention in the Bar-Ilan dataset is different to ours. To account for that, the original ground truth angles from the dataset were translated to our convention. The below figure shows the Bar-Ilan convention, as given in their example code. In brackets are the corresponding angles from our convention. All angles are in degrees.


          +---------------------------------------------------------+
          |                                                         |
          |  (0) 90  -1-2-3-4- mic array -5-6-7-8-  270 (180)       |
          |                                                         |
          |   (15) 75                               285 (165)       |
          |                                                         |
          |    (30) 60                             300 (150)        |
          |                                                         |
          |      (45) 45                         315 (135)          |
          |                                                         |
          |         (60) 30                   330 (120)             |     
          |               (75) 15        345 (105)                  |         
          |                     (90) 0                              |
          |                                                         |
          +---------------------------------------------------------+


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

### Training data generation - Pseudo code

**Generate RIRs**

This pseudo-code explains the generation of RIRs for the different acoustic conditions. For the specific acoustic parameters used in this work, please refer to Table 1.

```python
Select R rooms of different sizes

for nb_room in range(1,R) 
    Randomly select P array positions
    Choose D source-array distances
    for nb_pos in range(1,P)
        for nb_dist in range(1,D)
            Generate RIRs corresponding to each of the 37 discrete DOAs and M microphones

Store the NR = R*P*D RIRs

NOTE: Each RIR file corresponds to a specific acoustic setup and contains 37 x M source-mic RIRs for each DOA and microphone in the array
```
In the referenced paper:
  - R = 2 
  - P = 7
  - D = 2
  
**Training data - Features and Target generation**

```python

for nb_rir in range(1,NR)
    for nb_ang in range(1,37)
        sig_anechoic = 2 s long white Gaussian noise  # each iteration a different variance was used
        sig_spatial = sig_anechoic convolved with the M RIRs
        sig_noisy = sig_spatial + noise  ## noise = spatially uncorrelated white noise with a randomly chosen SNR in the range of [0,20]dB
        
        sig_STFT = STFT(sig_noisy)   ## size M (mics) x K (frequency bins) x N (time frames)
        phase_component = angle(sig_STFT)
        
        for nb_frame in range(1,N)
            phase_map(nb_frame) = phase_component(:,:,nb_frame) # matrix of size M x K taken from phase_component
            target(nb_frame) = one-hot encoded vector of size 37 x 1 with the true DOA label as 1, rest 0s
        
# Training pairs
X_train = phase_map tensor of size M x K x 1 x (N*NR*37) # resizing done for input to Conv2D in Keras
Y_train = target matrix of size 37 x (N*NR*37)

NOTE: Since the SNRs for each nb_ang and nb_rir is randomly chosen, the whole procedure was repeated 
several times to have a balanced dataset in order to avoid a specific SNR bias. The size of the training data was influenced by the memory constraints.
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
