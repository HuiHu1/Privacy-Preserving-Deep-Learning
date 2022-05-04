C implementation of "Training Privacy-Preserving Deep Neural Networks under Side-channel Power Attacks".

### Abstract
Privacy in deep learning is receiving tremendous attention with its wide applications in industry and academics. Recent studies have shown a traditional deep neural network is extremely vulnerable to side-channel attacks due to the static internal structure in the training process. In particular, side-channel power attacks are powerful to infer the internal structure of a deep neural network such that users' extremely sensitive predictions can be exposed severely. 

To solve this pressing privacy issue, in this paper, we propose a novel approach for training privacy-preserving deep neural networks under side-channel power attacks, called PD$^2$NN. The design principle of PD$^2$NN is introducing randomness into the model internal structure and model training process to generate random power traces in the temporal domain. PD$^2$NN includes three modules: First, Independent Sub-network Construction Module, which generates multiple independent sub-networks via randomly selecting nodes in each hidden layer. Second, Sub-network Random Training Module, which randomly trains multiple sub-networks such that power traces are also random. Third, Prediction Module, which outputs the predictions made by the most accurate sub-network to achieve high classification performance. The experimental results on two benchmark datasets demonstrate that PD$^2$NN decreases privacy inference accuracy by at least 38.07\% while maintaining competitive classification accuracy compared with traditional DNNs. We also theoretically analyze the power consumption of the proposed model and show the relation between the internal structure of PD$^2$NN and its power consumption.

### Requirements

ChipWhisperer Lite 1200, XMEGA Target Board, Jupyter Notebook.
