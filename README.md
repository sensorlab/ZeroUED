# SignalSage
Self-supervised specific emitter identification

Mikhail Krasnov, Ljupcho Milosheski, Mihael Mohorčič and Carolina Fortuna

Paper link: -

# Background

In this paper, we evaluate different approaches to the unknown radio-emitter detection task and propose our own approach. 

Evaluated approaches:
  - Auto Encoders ([Classical](https://ieeexplore.ieee.org/document/10623390) and [KAN](https://openreview.net/forum?id=Ozo7qJ5vZi))
  - [Deep Clustering](https://openaccess.thecvf.com/content_ECCV_2018/html/Mathilde_Caron_Deep_Clustering_for_ECCV_2018_paper.html)
  - [Classical Sim CLR](https://www.researchgate.net/publication/371460960_Contrastive_Self-supervised_Clustering_for_Specific_Emitter_Identification)
  - [Viewmaker Sim CLR](https://www.researchgate.net/publication/344678406_Viewmaker_Networks_Learning_Views_for_Unsupervised_Representation_Learning)

We have chosen two datasets for evaluation:
  - WiSig
  - Lora

# Installation
1. ```mamba env create -f env.yaml```
2. ```mamba activate SignalSage```
3. Create an account at https://wandb.ai
4. Login with your token (can be found in https://wandb.ai/quickstart?product=models) ```wandb login```
5. Download the data
   ```
   pip install gdown
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
