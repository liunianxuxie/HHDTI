# HHDTI: An approach based Heterogeneous Hypergraph for Drug-Target Interactions(DTIs) Prediction

HHDTI is a deep learning method that uses hypergraphs to model heterogeneous biological networks, integrates various interaction information, and accurately captures the topological properties of individual in the biological network to generate suitable structured embeddings for interactions prediction.

# Quick start
We provide an example script to run experiments on deepDTnet_20 dataset:

+   Run `python train.py`: predict drug-target interactions, and evaluate the results with 5-fold cross-validation.
+   You can change the dataset used, adjust the learning rate, hidden layer dimensions, etc
    `python train.py --dataset DTInet --lr 0.002 --hidden 32`


# Code

+   `train.py`:run HHDTI to predict drug-target interactions
+   `models.py`:the modules of VHAE and HGNN
+   `layers.py`:the layers used in VHAE and HGNN
+   `kl_loss.py`:KL divergence loss function for training
+   `hypergraph_utils.py`:generate degree matrices of hypergraph
+   `utils_deepDTnet.py`:load deepDTnet dataset for training and testing
+   `utils_DTInet.py`:load DTInet dataset for training and testing
+   `utils_KEGG_MED.py`:load KEGG_MED dataset for training and testing


# Data
+   `DTInet`:The DTIs in the DTInet dataset used for training and testing have been divided into the form of 10-fold cross-validation, as well as disease-drug interactions, disease-target interactions.
+   `deepDTnet`:The DTIs in the deepDTnet dataset used for training and testing have been divided into the form of 5-fold cross-validation, as well as disease-drug interactions, disease-target interactions.
+   `KEGG_MED`:The DTIs in the KEGG_MED dataset used for training and testing have been divided into the form of 10-fold cross-validation, as well as disease-drug interactions, disease-target interactions(the disease-drug interaction matrix file is larger than 25M, and cannot be uploaded temporarily).

# Note
+   You can change the files used in the code to perform other experiments, such as cold start experiments.
+   You can also use your own dataset for experiments, and DTIs are expressed in the form of lists.

# Requirements
+   python(v3.7.0)
+   numpy(v1.18.1)
+   pytorch(v1.1.0)
+   scikit-learn(v0.22.1)

# License
Our code is released under MIT License (see LICENSE file for details).
