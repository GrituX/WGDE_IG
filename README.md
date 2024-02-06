# WGDE IG architecture
This is the PyTorch code for the paper [Leveraging Interactional Sociology for Trust Analysis in Multiparty Human-Robot Interaction](https://dl.acm.org/doi/abs/10.1145/3623809.3623973) presented at [HAI 2023](https://hai-conference.net/hai2023/).

The code presents our WGDE-IG architecture and can be used to train all variations of it as described in the paper. The architecture was inspired by works from [Maman et al.](https://dl.acm.org/doi/abs/10.1145/3462244.3479940) and [Atamna et al.](https://telecom-paris.hal.science/hal-02922102/). The repo also contains all data described in the paper collected from the [Vernissage dataset](https://ieeexplore.ieee.org/abstract/document/6483545).

If you do not manage to run the code, please add an issue or contact us.

## Citing

If you find this repo or paper useful, please cite the following paper:

> @inproceedings{hulcelle2023wgdeig, author = {Hulcelle, Marc and Hemamou, L\'{e}o and Varni, Giovanna and Rollet, Nicolas and Clavel, Chlo\'{e}}, title = {Leveraging Interactional Sociology for Trust Analysis in Multiparty Human-Robot Interaction}, year = {2023}, isbn = {9798400708244}, publisher = {Association for Computing Machinery}, address = {New York, NY, USA}, url = {https://doi.org/10.1145/3623809.3623973}, doi = {10.1145/3623809.3623973}, booktitle = {Proceedings of the 11th International Conference on Human-Agent Interaction}, pages = {484–486}, numpages = {3}, keywords = {Trust, Recurrent Neural Networks, Interactional Sociology, HRI}, location = {<conf-loc>, <city>Gothenburg</city>, <country>Sweden</country>, </conf-loc>}, series = {HAI '23}}

# File structure

The structure of the code is as follows :

- Root folder
  - Annotations
  - Features
    - Aggregation
    - MFCC
    - Semantics
  - Result

### Root folder

In this folder, you can find all the python files to run the code. You can also find a log file once you've started running the code.

- **data_helper.py** : Contains a DataHelper class that handles all operations linked to data loading, and informations about features location in the vectors.
- **logger.py** : Contains a Logger class that handles log writing.
- **utility.py** : A utility file for misc operations.
- **wgde_ig_modules.py** : Contains all the classes as PyTorch modules to build the final WGDE-IG architecture.
- **train_sequential.py** : The main file to train the model. Detailed information on how to run the code is given later in this description.

### Annotations

This folder contains all annotations collected on the [Vernissage dataset](https://ieeexplore.ieee.org/abstract/document/6483545) using the [ELAN software](https://archive.mpi.nl/tla/elan). Annotations are available for each interaction either as raw ".eaf" ELAN files, as numpy arrays, or as a CSV file.

### Features

This folder contains all feature extracted as described in the paper. Features can either be loaded as a CSV dataframe whose format is described later. They can also be loaded as numpy arrays. The aggregated feature vector of each segment can be found in the folder corresponding to the interaction it was extracted from. 

### Result

This is the default folder in which the code outputs the training results (training scores, saved models, and graphs).

# Usage

## Running the code

To start running the code right away, run the following command :

    python3 train_sequential.py --rnn_archi IG --seq_len 10 --split_participants 1

By default, the code checks for a CUDA device on the system and runs the training on it if it finds one. Further information on all options are provided in the "train_sequential.py" file.

## Feature Dataframe format

The dataframe contains all aggregated feature vectors from all segments of all interactions. The rows correspond to the feature vector of a single segment of an interaction. The first columns "Interaction", "Step", "Label" provide information on respectively the originating interaction, the segment number within the interaction, and the ground truth as provided by the annotations.

Additional columns describe a single feature. Some feature names contains suffix which we describe here:
- p1 : features correponding to the participant on the right through the rear camera view.
- p2 : features correponding to the participant on the left through the rear camera view.
- Nao : features correponding to the Nao robot.
- d : these features correspond to the derivatives of the original feature, computed between the current and the following segment.

# Requirements

We ran this code with the following setup :

- Python 3.9 environment with the following key elements :
  - numpy 1.23.5
  - pytorch 1.11.0
  - pandas 1.4.4
  - torchmetrics 0.9.1
  - scikit-learn 1.1.1
  - seaborn 0.12.2
  - matplotlib 3.7.1
- GPU: Nvidia RTX A3000
- CPU: 11th Gen Intel(R) Core(TM) i7-11850H @ 2.50GHz   2.50 GHz
- RAM: 16 Go
- OS: Windows 10 Pro

# Contacts

- [Marc Hulcelle](https://gritux.github.io)
- [Léo Hemamou](https://lhemamou.github.io/)
- [Chloé Clavel](https://clavel.wp.imt.fr/)
- [Giovanna Varni](https://scholar.google.it/citations?user=7AM4CZIAAAAJ&hl=it)
- [Nicolas Rollet](https://www.telecom-paris.fr/nicolas-rollet)
