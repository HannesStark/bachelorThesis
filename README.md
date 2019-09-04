# Understanding Variational Autoencoders' Latent Representations of Remote Sensing Images


## Abstract

*In computer vision, neural networks have been successfully employed to solve
multiple tasks in parallel in one model. The joint exploitation of related objectives
can improve the performance of each individual task. The architecture of
these multi task models can be more complex since the networks branch into the
atomic tasks at a certain depth. For this reason the process of designing such
architectures often involves a lot of time consuming trial-and-error. Therefore
a more systematic taxonomy is desirable to replace this experimental process.
For constructing this taxonomy it is necessary to have an understanding of the
latent information learned by specific layers of single task models.
This work uses convolutional variational autoencoders to produce latent representations
of aerial images which are analyzed to understand the inner workings
of the models. The method relies on testing whether or not learned clusters
can be attributed to different high level input features like topographic classes
common for the field of remote sensing. Visualizations are produced to gain insight
and an understanding of the information captured in the latent space
of the variational autoencoders. Moreover, it is observed how dierent architectural
choices aect the reconstructions and the latent space. Code and
the trained models to reproduce the experiments are publicly available here:
https://github.com/HannesStaerk/bachelorThesis.*


## Models

Every script like `Kernel3adjusted2x2x256.py` contains an architecture with an adjustable
coding size for the latent vector. The a `data_source_dir` with 128x128 images
has to be specified and functions to train, to make predictions and generations and to
create the t-SNE and PCA visualizations can be called.



## Saved Weights

The weights of all trained models can be found in the directory `savedModels`.
These include the weights for the models that were trained on 3000 images
for the experiments regarding the architecture and the quality of the reconstructions
as well as the models trained on 178,112 images for the evaluation of the latent space.
Each script can load the weights by uncommenting 
`variational_ae.save_weights("./savedModels/" + file_tag + ".h5")`.


## Data

In `utils.py` there is a function to split and resize images. 
This can be used to split or risize the 1024x1024 images given by the dataset into the 128x128
images that the models take as input.

The Data can be downloaded here http://www.grss-ieee.org/community/technical-committees/data-fusion/2019-ieee-grss-data-fusion-contest-data/