# HistNet. A DNN for quantification.

HistNet is a DNN for quantification. This repository is the code for reproduce the experiments in the paper **Histogram-based Deep Neural Network for Quantification**.

## Preparing the enviroment
First, lets create a virtual enviroment using conda:
```bash
conda create --name histnetenv python=3.7.9
conda activate histnetenv
```

The next step is to install some requirements.

```bash
conda install pytorch==1.7.1 -c pytorch
conda install tensorflow==2.4.1 pandas==1.1.5 scikit-learn==0.22 matplotlib==3.3.2
conda install -c conda-forge pytorch-lightning==1.2.4
```

The main requirement is Pytorch and Pytorch Lightning. You will note that we are installing TensorFlow. The reason is that I found loading the dataset easier with this framework. Even though, this can be changed and erase this dependency as HistNet depends only on PyTorch.

The project also depends on the quantification framework [QuaPy](https://github.com/HLT-ISTI/QuaPy). Note that HistNet does not depend on the framework but as we are comparing with QuaNet and other methods, this dependency is neccesary to execute the scripts. As the framework is quite new and constantly evolving, I recomend to use my local fork of the project to reproduce the results.

Lets download it:

```bash
git clone https://github.com/pglez82/QuaPy
```
Note that the scripts have an import to this repository `sys.path.append(r'/media/nas/pgonzalez/QuaPy')`. You should change that path so it matches your configuration.

## Running the experiments
Just run:

```bash
python fashionmnist_experiments.py
```
or for the IMDB experiments:
```bash
python IMDB_experiments.py
```
The results will be saved in a folder in the same directory where the two scripts live.