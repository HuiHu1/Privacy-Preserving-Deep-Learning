C and Python implementation of the paper "Stealing Model Parameters via Side Channel Power Attacks" (ISVLSI-2021)[[Link]](https://ieeexplore.ieee.org/document/9516772), and the paper "Training Privacy-Preserving Deep Neural Networks under Side-Channel Power Attacks".

### Abstract
Privacy in deep learning is receiving tremendous attention with its wide applications in industry and academics. Recent studies have shown the internal structure of a deep neural network is easily inferred via side-channel power attacks in the training process. To address this pressing privacy issue, we propose TP-NET, a novel solution for training privacy-preserving deep neural networks under side-channel power attacks. The main idea of TP-NET is to introduce randomness into the internal structure of a deep neural network and the training process. Specifically, the workflow of TP-NET includes three steps: First, Independent Sub-network Construction, which generates multiple independent sub-networks via randomly selecting nodes in each hidden layer. Second, Sub-network Random Training, which randomly trains multiple sub-networks such that power traces keep random in the temporal domain. Third, Prediction, which outputs the predictions made by the most accurate sub-network to achieve high classification performance. The performance of TP-NET is evaluated under side-channel power attacks. The experimental results on two benchmark datasets demonstrate that TP-NET decreases the inference accuracy on the number of hidden nodes by at least 38.07\% while maintaining competitive classification accuracy compared with traditional DNNs. Finally, we also theoretically analyze the power consumption of TP-N

### Requirements

ChipWhisperer Lite 1200, XMEGA Target Board, Visual Studio Code, Jupyter Notebook.

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
}, and
 update later
```
