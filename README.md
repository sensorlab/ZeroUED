# SignalSage
Design Principles of Zero-Shot Self-Supervised Unknown Emitter Detectors

Mikhail Krasnov, Ljupcho Milosheski, Mihael Mohorčič and Carolina Fortuna

Paper link: [2511.07026](https://arxiv.org/abs/2511.07026)

<img width="413" height="282" alt="image" src="https://github.com/user-attachments/assets/aa1c5794-062b-430f-b611-04e20600f53b" />
<img width="354" height="216" alt="image" src="https://github.com/user-attachments/assets/ad8c7fce-5482-43fa-8439-d7d5bf1e755e" />



# Background

In this paper, we investigate the design space for unknown emitter detectors over the two data transmitions scenarios: same and different messages. 

Evaluated approaches:
  - [Auto Encoders ](https://ieeexplore.ieee.org/document/10623390)
  - [Deep Clustering](https://openaccess.thecvf.com/content_ECCV_2018/html/Mathilde_Caron_Deep_Clustering_for_ECCV_2018_paper.html)
  - [Sim CLR](https://www.researchgate.net/publication/371460960_Contrastive_Self-supervised_Clustering_for_Specific_Emitter_Identification)


We have chosen two datasets for evaluation:
  - [WiSig](https://cores.ee.ucla.edu/downloads/datasets/wisig/)
  - [ORACLE](https://repository.library.northeastern.edu/files/neu:m044q520q)

# Installation
1. ```pip install -r reqs.txt```
2. Create an account at https://wandb.ai
3. Login with your token (can be found in https://wandb.ai/quickstart?product=models) ```wandb login```
4. Download the data (if does not work for ORACLE - download it manualy from [here](https://repository.library.northeastern.edu/files/neu:m044q520q))
   ```
   python download_data.py
   ```
5. Create ORACLE Dataset
   ```
   python create_oracle_dataset.py 62ft/ data_256_62ft.h5
   ```
6. Run experiemnts from config ```python run_experiments configs/wisig/raw_iq/ae_config.yaml ```

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
This work was supported in part by the European Union’s Horizon Europe research and innovation programme under the NANCY project (GA No. 101096456), EnerTEF (GA No. 101172887) and in part by the Slovenian Research and Innovation Agency under the grant P2-0016.
