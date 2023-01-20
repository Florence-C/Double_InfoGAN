# Double InfoGAN

## Presentation 

This folder is a clone of the repo 

Implementation of the Moment Matching Contrastive VAE from 
"Moment Matching Deep Contrastive Latent Variable Models" (AISTATS 2022).
Notebooks also included to preprocess data used in our paper.




## Launch code 

### Preprocess 

```
python -m preprocess.preprocess_celeba
```
Then move created files in datasets folder


### Training code


### Evalutation

```
python -m eval.eval_target_sep_celeba_infogan
```

