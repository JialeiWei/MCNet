# **Overview**

This repository contains the code and datasets used for training the MCNet model, as described in the paper:

**"Enhancing Black Hole Image Parameter Regression via Multiscale Feature Fusion and Cosine Annealing Optimization"**

MCNet is a deep learning framework for black hole image parameter regression, combining multi-scale feature extraction and cosine annealing optimization to enhance the accuracy of parameter predictions. This repository includes code for training and testing the model on the GRRT-simulated datasets, along with instructions for using the provided dataset and reproducing results from the paper.

This work is currently under review for submission to **The Visual Computer** journal.

# **Dataset**

The dataset used in this repository is the GRRT-simulated dataset, which is organized as follows:

```
GRRT_dataset/
    ├── train/
    └── val/
```

Each image in the dataset represents a black hole simulation and is accompanied by a label that corresponds to a physical parameter (e.g., mass, spin) encoded in the image filename.

# **Installation**

To run the code, you need the following dependencies:

```
Python 3.9
Pytorch
Matplotlib
Pillow
NumPy
torchvision
```

# **Training**

To train the model, run the following command:

```
python train_MCNet.py
```

This will train the MCNet model on the training set (GRRT_dataset/train) and evaluate it on the validation set (GRRT_dataset/val). The training process logs the loss and R² scores, and the best model is saved to the saved_models directory.

# **Evaluation**

Once the model is trained, you can evaluate it using:

```
python test_MCNet.py
```

This will compute the $R^2$ score for the trained model and save the results.
