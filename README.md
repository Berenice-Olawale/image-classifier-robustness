# Image Classifier Robustness Under Common Image Distortions

**Status:** In progress  
**Research area:** Computer Vision · Machine Learning · Model Robustness  
**Tools:** Python, PyTorch, Torchvision, NumPy, Pillow, Matplotlib

## Overview

This independent research project investigates how common image distortions affect the predictions of image classification models.

Image classifiers are often evaluated using clean images, but real-world images may contain noise, blur, or changes in lighting. Understanding how these changes affect model predictions is an important step towards evaluating model robustness.

The project began with a pilot experiment using an ImageNet-pretrained ResNet-18 model. The experiment compares predictions and confidence scores for an original image and versions modified with Gaussian noise, Gaussian blur, and reduced brightness.

The next stage is to evaluate a model trained for CIFAR-10 on a labelled test dataset to investigate how distortions affect classification accuracy across multiple images.

## Research Questions

1. How do Gaussian noise, Gaussian blur, and reduced brightness affect image classification?
2. Does increasing distortion severity change the model's confidence in its predictions?
3. Which distortions have the greatest effect on classification accuracy across a labelled dataset?

## Project Objectives

- Understand the inference process of a pretrained image classification model.
- Implement common image distortions using Python.
- Compare model predictions and confidence scores under different image conditions.
- Evaluate classification accuracy using a labelled test dataset.
- Analyse experimental results and document limitations.

## Methodology

### Stage 1: Pilot Experiment

A pretrained ResNet-18 model with ImageNet weights was used to classify an example image.

The original image was compared with modified versions created using:

- **Gaussian noise:** Adds random variations to pixel values.
- **Gaussian blur:** Reduces image sharpness and fine detail.
- **Reduced brightness:** Simulates darker lighting conditions.

The pilot experiment records the predicted categories and their confidence scores. It is exploratory and uses a single image, so its findings cannot establish general model robustness.

### Stage 2: Dataset-Based Evaluation

The planned evaluation uses CIFAR-10, a labelled dataset containing 10 image categories.

A model trained specifically for CIFAR-10 will be evaluated on a separate test set. Accuracy on clean images will be compared with accuracy under different distortion types and strengths.

The experiment will keep the test images and evaluation procedure consistent across conditions to support a fair comparison.

## Preliminary Observations

In the initial single-image pilot:

- The original image was classified as *Samoyed* with 88.46% confidence.
- Gaussian noise increased the reported confidence to 90.75%.
- Gaussian blur reduced the reported confidence to 74.05%.
- Reduced brightness increased the reported confidence to 92.79%.

These observations apply only to the tested image. Confidence is not equivalent to accuracy, and the pilot does not establish how the model performs across a wider range of images.

## Technology Stack

- **Python:** Experiment implementation.
- **PyTorch:** Model inference and evaluation.
- **Torchvision:** Pretrained models, image transformations, and datasets.
- **NumPy:** Numerical operations and noise generation.
- **Pillow:** Image manipulation and distortions.
- **Matplotlib:** Visualisation of images and results.
- **Google Colab:** Development and GPU-enabled experiments.
- **Git and GitHub:** Version control and project documentation.

## Repository Structure

```text
image-classifier-robustness/
├── README.md
├── notebooks/
│   ├── 01_pretrained_model_pilot.ipynb
│   └── 02_cifar10_robustness.ipynb
├── src/
│   └── distortions.py
├── results/
├── figures/
└── report/
    └── research_report.md
```

*This structure describes the intended organisation; folders and files will be added as the project develops.*

## Current Limitations

- The initial pilot uses only one image.
- The pilot measures prediction confidence rather than dataset-level accuracy.
- A single image cannot represent the variety of real-world inputs.
- Further experiments are needed to assess whether the observed patterns generalise.

## Future Work

- Complete the CIFAR-10 dataset-based evaluation.
- Compare clean-image accuracy with accuracy under controlled distortions.
- Investigate the effect of increasing distortion severity.
- Visualise and interpret the results.
- Document the experimental setup, findings, and limitations in a research report.

## Reproducibility

The notebooks will document the experimental steps and code as the project develops. Random seeds will be used where appropriate to improve reproducibility.

The project is ongoing. Results and conclusions will be updated as additional experiments are completed.

## Author

**Berenice Olawale**  
Computer Science Engineering student  
John von Neumann University, Hungary

Interested in machine learning, computer vision, and software engineering.
