# Double InfoGAN for Constrative Analysis

## Presentation 

This folder contains the implementation of Double InfoGAN for Contrastive Analysis. 

It is originally a clone of the repo [MM-cVAE](https://github.com/suinleelab/MM-cVAE)

It uses pytorh 1.9 together with pytorch lightning. 

## Launch code 

### Prepare data

In folder preprocess you can find all the codes to create the datasets (celeba, cifar+mnist, etc...). The codes create datasets for training, evaluation and tests, and the created files must be moved in /datasets folder. To run the preprocess, run the following code with 

```
python -m preprocess.preprocess_datasetName
```

For celeba, download the celeba dataset and put the images in preproces/celeba_data/img_align_celeba


Dowload dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz from the repo [dsprites-dataset](https://github.com/deepmind/dsprites-dataset)

For Brats Dataset, images (https://www.med.upenn.edu/cbica/brats2020/data.html) must be splitted in healty and cancerous. 


### Training code

The training code is run_experiment.py. To launch the code, select a config file and chose a directory to store the results and run : 

```
python run_experiment.py --config_file config/config_file.yml --logdir ./results_directory/
```

In the config file you can modify every hyperparameter of the training. 

### Evalutation

Evaluation codes is also provide in /eval directory. Codes are provided to measure the target class separation for celeba and cifar/mnist dataset, as well as fvae metrics for mnist/dsprite dataset. 

Tu run the code (here target class separation for infogan), chose the directory you wan to evaluate inside the code, then launch : 

```
python -m eval.eval_target_sep_celeba_infogan
```

