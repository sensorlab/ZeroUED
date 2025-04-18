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

# How to start
1. ```mamba env create -f env.yaml```
2. ```mamba activate SignalSage```
3. Create an account at https://wandb.ai
4. Login with your token ```wandb login```
5. Download the data. For example https://cores.ee.ucla.edu/downloads/datasets/wisig/#/downloads
   ```
   pip install gdown
   gdown.download('https://drive.google.com/uc?id=1szuns8MhcYocdbipK9t9TM9MLgEMklxk', 'many_sig', quiet=False)
   ```
6. Run experiemnts from config ```python run_experiments configs/ae_config.yaml ```

## Alternative way for steps 1-2:
```pip install -r reqs.txt```
