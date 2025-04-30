# SignalSage
Self-supervised specific emitter identification

Mikhail Krasnov, Ljupcho Milosheski, Mihael Mohorčič and Carolina Fortuna

Paper link: -

# Background

In this paper, we evaluate different approaches to the unknown radio-emitter detection task and propose our own approach. 

Evaluated approaches:
  - Auto Encoders (Classical and KAN)
  - Deep Clustering
  - Classical Sim CLR
  - Viewmaker Sim CLR
  - Viewmaker Sim CLR Large Augs(ours)

We have chosen three datasets for evaluation:
  - Drones Dataset
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
    ```src/trainers.py``` Trainers that handles learning process of different approaches.
    ```src/datasets.py``` Datasets objects.
    ```src/metrics.py```  Code for metrics computation.
    ```src/config_manager.py``` Handles configs files.  
    ```src/architectures``` Dir with Features extractors, Mlp head and Viewmakers.

```run_experiments.py``` Code for runing experiments from terminal using config.

```reqs.txt``` and ```env.yaml``` Dependences.

```download_data.py``` script to download datasets.