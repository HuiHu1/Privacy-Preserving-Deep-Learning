This repository aims to record the related work on hardware-based privacy-preserving deep learning.

C and Python implementation of privacy-preserving deep learning:

1. Stealing Model Parameters via Side-Channel Power Attacks (ISVLSI 2021) [[Paper]](https://ieeexplore.ieee.org/document/9516772).

2. Training Privacy-Preserving Deep Neural Networks under Side-Channel Power Attacks (ISES 2022) [[Paper]](https://ieeexplore.ieee.org/abstract/document/10027138).

3. Robust Privacy-Preserving Deep Learning under Side-Channel Power Attacks (SDM 2022) [[Poster]](https://github.com/HuiHu1/Cooperative-Projects/blob/main/SDM2022.pdf).

### Abstract
Privacy in deep learning is receiving tremendous attention with its wide applications in industry and academics. Recent studies have shown the internal structure of a deep neural network is easily inferred via side-channel power attacks in the training process. To address this pressing privacy issue, we propose TP-NET, a novel solution for training privacy-preserving deep neural networks under side-channel power attacks. The key contribution of TP-NET is the introduction of randomness into the internal structure of a deep neural network and the training process. Specifically, the workflow of TP-NET includes three steps: First, Independent Sub-network Construction, which generates multiple independent sub-networks via randomly selecting nodes in each hidden layer. Second, Sub-network Random Training, which randomly trains multiple sub-networks such that power traces keep random in the temporal domain. Third, Prediction, which outputs the predictions made by the most accurate sub-network to achieve high classification performance. The performance of TP-NET is evaluated under side-channel power attacks. The experimental results on two benchmark datasets demonstrate that TP-NET decreases the inference accuracy on the number of hidden nodes by at least 38.07\% while maintaining competitive classification accuracy compared with traditional deep neural networks. Finally, a theoretical analysis shows that the power consumption of TP-NET depends on the number of sub-networks, the structure of each sub-network, and atomic operations in the training process.

### Requirements

ChipWhisperer Lite 1200, XMEGA Target Board, Visual Studio Code, Jupyter Notebook.

![]([https://github.com/HuiHu1/Privacy-Preserving-Deep-Learning/blob/main/CW_Lite.png])

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
(8) Set up a password for Jupyter.
(9) Reboot the VM. sudo reboot and open Firefox/Chrome (ONLY) - navigate to 127.0.0.1:8888 or localhost:8888.

Step 2: Target board setup (SCOPETYPE = 'OPENADC'  PLATFORM = 'CWLITEXMEGA'  SS_VER='SS_VER_1_1').

Step 3: %run /home/vagrant/work/projects/chipwhisperer/jupyter/Setup_Scripts/Setup_Generic.ipynb.

Step 4: Compile program to run on board.
PATH = "/home/vagrant/work/projects/chipwhisperer/hardware/victims/firmware/"
TARGET_MODEL = "TP-NET"
%%bash -s "$PLATFORM" "$PATH" "$TARGET_MODEL" "$SS_VER"
cd $2$3
make PLATFORM=$1 CRYPTO_TARGET=NONE SS_VER=$4

Step 5: Pass data from Jupyter Notebook to the target board.

Step 6: Run the model and collect power traces.

```
Note: [Update the firmware on the CW-Lite board](https://wiki.newae.com/Manual_SAM3U_Firmware_Update).

### Citation
```
{
@inproceedings{wolf2021stealing,
  title={Stealing Machine Learning Parameters via Side Channel Power Attacks},
  author={Wolf, Shaya and Hu, Hui and Cooley, Rafer and Borowczak, Mike},
  booktitle={2021 IEEE Computer Society Annual Symposium on VLSI (ISVLSI)},
  pages={242--247},
  year={2021},
  organization={IEEE}
}, 
and
@inproceedings{hu2022tp,
  title={TP-NET: Training Privacy-Preserving Deep Neural Networks under Side-Channel Power Attacks},
  author={Hu, Hui and Gegax-Randazzo, Jessa and Carper, Clay and Borowczak, Mike},
  booktitle={2022 IEEE International Symposium on Smart Electronic Systems (iSES)},
  pages={439--444},
  year={2022},
  organization={IEEE}
}
```
