# SignalSage
Design Principles of Zero-Shot Self-Supervised Unknown Emitter Detectors

Mikhail Krasnov, Ljupcho Milosheski, Mihael Mohorčič and Carolina Fortuna

Paper link: [2511.07026](https://arxiv.org/abs/2511.07026)

<img width="826" height="564" alt="image" src="https://github.com/user-attachments/assets/aa1c5794-062b-430f-b611-04e20600f53b" />


# Background

In this paper, we investigate different approaches to the unknown radio-emitter detection task and propose our own approach. 

Evaluated approaches:
  - [Auto Encoders ](https://ieeexplore.ieee.org/document/10623390)
  - [Deep Clustering](https://openaccess.thecvf.com/content_ECCV_2018/html/Mathilde_Caron_Deep_Clustering_for_ECCV_2018_paper.html)
  - [Sim CLR](https://www.researchgate.net/publication/371460960_Contrastive_Self-supervised_Clustering_for_Specific_Emitter_Identification)


We have chosen two datasets for evaluation:
  - [WiSig](https://cores.ee.ucla.edu/downloads/datasets/wisig/)
  - [ORACLE](https://repository.library.northeastern.edu/files/neu:m044q520q)

# Installation
1. ```mamba env create -f env.yaml```
2. ```mamba activate SignalSage```
3. Create an account at https://wandb.ai
4. Login with your token (can be found in https://wandb.ai/quickstart?product=models) ```wandb login```
5. Download the data
   ```
   python download_data.py
   ```
6. Run experiemnts from config ```python run_experiments configs/ae_config.yaml ```

### Alternative way for steps 1-2:
```pip install -r reqs.txt```

# Repository structure

```configs/``` Configurations of epxeriments.

```src/``` Dir of the source code. 

    - src/trainers.py       Trainers that handles learning process of different approaches.
    - src/datasets.py       Datasets objects.
    - src/metrics.py        Code for metrics computation.
    - src/config_manager.py Handles configs files.  
    - src/architectures`    Dir with Features extractors, Mlp head and Viewmakers.

```run_experiments.py``` Code for runing experiments using config.

```reqs.txt``` and ```env.yaml``` Dependences.

```download_data.py``` Script to download datasets.

# Acknowledgment
The authors would like to acknowledge funding from the European Union's Horizon Europe Framework Programme NANCY project under Grant Agreement No. 101096456.
