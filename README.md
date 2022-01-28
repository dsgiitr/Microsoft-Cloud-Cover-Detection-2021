# Microsoft-Cloud-Cover-Detection-2021


This repository contains the solution code to the [Microsoft Cloud-cover detection challenge 2021](https://www.drivendata.org/competitions/83/cloud-cover/).



## Problem Description

In this challenge, our goal was to label clouds in satellite imagery. In many uses of satellite imagery, clouds obscure what we really care about - for example, tracking wildfires, mapping deforestation, or visualizing crop health. Being able to more accurately remove clouds from satellite images filters out interference, unlocking the potential of a vast range of use cases.

The challenge uses publicly available satellite data from the Sentinel-2 mission, which captures wide-swath, high-resolution, multi-spectral imaging. Data is publicly shared through Microsoft's [Planetary Computer](https://planetarycomputer.microsoft.com/).

## Models implemented and experiments

- We implemented the DeeplabV3-Plus model for semantic segmentation of cloud   cover in semantic segmentation. Additionally, we used xgboost classifier to   improve the jaccard index on the validation dataset.
- DataAugmentation was performed with the help of  [kornia](https://kornia.readthedocs.io/) CV library in pytorch.
- Additional bands were fetched from the PlanetaryComputer in addition to the 4 bands per chip in the training dataset, but this did not improve the jaccard score on the validation dataset.


### Performance on the test dataset

Jaccard index, also known as Generalized Intersection over Union (IoU) was used as the performance mmetric. Our submission obtained 0.8596 jaccard score on the testset. 

We obtained a rank of 74 in the competiton.

#### References 

- [Kornia-docs](https://kornia.readthedocs.io/)
- [DeepLab-V3Plus](https://arxiv.org/pdf/1802.02611) research paper
- [Pytorchlightning-docs](https://pytorch-lightning.readthedocs.io/en/latest/)
