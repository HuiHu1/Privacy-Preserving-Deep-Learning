C implementation of "Training Privacy-Preserving Deep Neural Networks under Side-Channel Power Attacks".

### Abstract
Privacy in deep learning is receiving tremendous attention with its wide applications in industry and academics. Recent studies have shown a traditional deep neural network is extremely vulnerable to side-channel attacks due to the static internal structure in the training process. In particular, side-channel power attacks are powerful to infer the internal structure of a deep neural network such that users' extremely sensitive predictions can be exposed severely. 

To solve this pressing privacy issue, in this paper, we propose a novel approach for training privacy-preserving deep neural networks under side-channel power attacks, called PD^2NN. The design principle of PD^2NN is introducing randomness into the model internal structure and model training process to generate random power traces in the temporal domain. PD^2NN includes three modules: First, Independent Sub-network Construction Module, which generates multiple independent sub-networks via randomly selecting nodes in each hidden layer. Second, Sub-network Random Training Module, which randomly trains multiple sub-networks such that power traces are also random. Third, Prediction Module, which outputs the predictions made by the most accurate sub-network to achieve high classification performance. The experimental results on two benchmark datasets demonstrate that PD^2NN decreases privacy inference accuracy by at least 38.07\% while maintaining competitive classification accuracy compared with traditional DNNs. We also theoretically analyze the power consumption of the proposed model and show the relation between the internal structure of PD^2NN and its power consumption.

### Requirements

ChipWhisperer Lite 1200, XMEGA Target Board, Jupyter Notebook.

### Run the code

```
Step 1: Get the VM and Jupyter up and running.

(1) Download the latest version of VirtualBox (Link: https://www.virtualbox.org/wiki/Downloads).
(2) Download/Install the extension pack (Link: https://download.virtualbox.org/virtualbox/6.1.18/). 
(3) Download the current ChipWhisperer VM (Link: https://github.com/newaetech/chipwhisperer/releases/). 
(4) Download 7Zip (Link: https://www.7-zip.org/download.html). 
(5) UnZip the VM you just downloaded.
(6) Launch Virtualbox > Machine > Add... > Select File you just unzipped.
(7) Start/Run the virtual machine and login.
(8) Setup a password for Jupyter.
(9) Reboot the VM. sudo reboot and open Firefox/Chrome (ONLY) - navigate to 127.0.0.1:8888 or localhost:8888.

Step 2: Target board setup (SCOPETYPE = 'OPENADC'  PLATFORM = 'CWLITEXMEGA'  SS_VER='SS_VER_1_1').

Step 3: %run /home/vagrant/work/projects/chipwhisperer/jupyter/Setup_Scripts/Setup_Generic.ipynb.

Step 4: Compile program to run on board. 
PATH = "/home/vagrant/work/projects/chipwhisperer/hardware/victims/firmware/"
TARGET_MODEL = "PP_DNN" 
%%bash -s "$PLATFORM" "$PATH" "$TARGET_MODEL" "$SS_VER"
cd $2$3 
make PLATFORM=$1 CRYPTO_TARGET=NONE SS_VER=$4

Step 5: Pass data from Jupyter Notebook to the target board.

Step 6: Run model and collect power traces.
```
### Citation
```
Update later
```
