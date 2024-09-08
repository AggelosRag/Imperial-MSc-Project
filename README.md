# Controlling Information Leakage in Concept Bottleneck Models with Trees
### MSc Thesis Project by Angelos Ragkousis

## Table of Contents
- [Abstract](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Abstract
In this work, we propose a new method to train a CBM while retaining Leakage Inspection and Control properties using Decision Trees. This approach can be used for both sequentially and jointly trained CBMs, and is also tested in a Mix- ture of Experts Framework. Our method uses Information Leakage to extend the Decision Paths of a classifier which have missing concept information. It allows for Inspection and Control for specific subsets of data, improves the task accuracy for these subsets, and provides more meaningful explanations.
## Installation
Steps to set up the environment and install dependencies:
1. Clone the repository:
    ```bash
    git clone https://github.com/AggelosRag/AR-Imperial-Thesis.git
    cd your-repo
    ```
2. Create and activate a virtual environment:
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```
3. Install dependencies (not added yet, packages need to be installed manually):
    ```bash
    pip install -r requirements.txt
    ```

## Usage
How to run an experiment: The ```train.py``` script should be called, using the appropriate configuration file from the ```./configs``` folder. Two examples are given below:
- To train a Vanilla CBM model on the Morpho-MNIST dataset with independent training, use the following configuration file (change hyperparameters as needed):
    ```bash
    python train.py --config ./configs/CBM/Independent/No_Regularisation/mnist_full.json
    ```
- To train the MCBM-Seq algorithm on the Morpho-MNIST dataset, use the following configuration file (change hyperparameters as needed):
    ```bash
    python train.py --config ./configs/LeakageInspection/mnist.json
    ```
## License
Project license:
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
