# Image Classifier Robustness Under Common Image Distortions

**Status:** Experiments completed; further analysis in progress  
**Research area:** Computer Vision · Machine Learning · Model Robustness  
**Tools:** Python, PyTorch, Torchvision, NumPy, Pillow, Matplotlib, Google Colab

## Overview

This independent research project investigates how common image distortions affect image classification models.

Image classifiers are often evaluated using clean images, but real-world images may contain noise, blur, or changes in lighting. Understanding how these changes affect classification performance is an important step towards evaluating model robustness.

The project began with a pilot experiment using an ImageNet-pretrained ResNet-18 model. It then progressed to a dataset-based evaluation using a convolutional neural network (CNN) trained on CIFAR-10.

The main experiment evaluates the model on 10,000 labelled test images under clean conditions and three types of image distortion: Gaussian noise, Gaussian blur, and reduced brightness.

## Research Questions

1. How do Gaussian noise, Gaussian blur, and reduced brightness affect image classification accuracy?
2. How does increasing distortion severity affect the performance of a trained CNN?
3. How does classification accuracy under distorted images compare with accuracy on clean test images?

## Project Objectives

- Understand the inference process of a pretrained image classification model.
- Implement common image distortions using Python.
- Compare model predictions and confidence scores under different image conditions.
- Train and evaluate a CNN on a labelled image dataset.
- Measure classification accuracy under controlled distortions.
- Analyse results, identify limitations, and propose further experiments.

## Methodology

### Stage 1: Pilot Experiment

An ImageNet-pretrained ResNet-18 model was used to classify an example image.

The original image was compared with modified versions created using:

- **Gaussian noise:** Adds random variations to pixel values.
- **Gaussian blur:** Reduces image sharpness and fine detail.
- **Reduced brightness:** Simulates darker lighting conditions.

The pilot recorded predicted categories and confidence scores. Since it used only one image, it was exploratory and could not establish dataset-level robustness.

### Stage 2: CIFAR-10 Dataset-Based Evaluation

A CNN was trained to classify images from CIFAR-10, a dataset containing 10 image categories.

The dataset contains 50,000 training images and 10,000 test images. The model was trained for five epochs and evaluated on the test set.

The clean test-set accuracy was used as the baseline. The same test images and trained model were then evaluated under different distortion conditions.

The experiments covered three distortion types:

- **Gaussian noise:** Standard deviations of 0.05, 0.10, and 0.20.
- **Gaussian blur:** Sigma values of 0.5, 1.0, and 2.0.
- **Brightness reduction:** Brightness factors of 0.75, 0.50, and 0.25.

Accuracy was calculated as the percentage of test images classified correctly. The change in accuracy was measured in percentage points relative to the clean baseline.

## Experimental Results

### Clean Baseline

The trained CNN achieved **67.59% accuracy** on the clean CIFAR-10 test set, correctly classifying 6,759 of the 10,000 test images.

### Gaussian Noise

| Noise standard deviation | Accuracy | Drop from baseline |
|---:|---:|---:|
| 0.05 | 63.00% | 4.59 percentage points |
| 0.10 | 50.07% | 17.52 percentage points |
| 0.20 | 27.49% | 40.10 percentage points |

Accuracy decreased as the noise standard deviation increased, indicating that stronger pixel-level noise substantially affected this model's classification performance.

### Gaussian Blur

| Blur sigma | Accuracy | Drop from baseline |
|---:|---:|---:|
| 0.5 | 66.46% | 1.13 percentage points |
| 1.0 | 53.23% | 14.36 percentage points |
| 2.0 | 35.78% | 31.81 percentage points |

Mild blur caused a relatively small decrease in accuracy, while stronger blur resulted in a more substantial decline.

### Reduced Brightness

| Brightness factor | Accuracy | Drop from baseline |
|---:|---:|---:|
| 0.75 | 67.24% | 0.35 percentage points |
| 0.50 | 57.50% | 10.09 percentage points |
| 0.25 | 17.10% | 50.49 percentage points |

Moderate brightness reduction had a limited effect compared with the strongest setting. At a brightness factor of 0.25, accuracy decreased substantially.

### Preliminary Interpretation

Across the tested settings, stronger distortions were associated with lower classification accuracy.

The largest observed decrease was under the strongest brightness reduction, followed by strong Gaussian noise and strong Gaussian blur. However, the distortion parameters use different scales and do not represent equivalent perceptual severity. Therefore, these results do not establish that one distortion type is universally more damaging than another.

The findings apply to this CNN, its training setup, the CIFAR-10 test set, and the distortion settings used in these experiments.

## Pilot Experiment Observations

In the initial single-image pilot using the pretrained ResNet-18 model:

- The original image was classified as *Samoyed* with 88.46% confidence.
- Gaussian noise increased the reported confidence to 90.75%.
- Gaussian blur reduced the reported confidence to 74.05%.
- Reduced brightness increased the reported confidence to 92.79%.

These observations illustrate that a distortion does not necessarily reduce the confidence of an individual prediction. Confidence is not equivalent to accuracy, and a confident prediction can still be incorrect.

## Technology Stack

- **Python:** Experiment implementation.
- **PyTorch:** CNN training, inference, and evaluation.
- **Torchvision:** Pretrained models, image transformations, and CIFAR-10.
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

The notebooks document the pilot experiment and the main CIFAR-10 evaluation. Supporting code, results, visualisations, and the research report are organised in the corresponding directories.

## Current Limitations

- The pilot experiment uses only one image and cannot establish general robustness.
- The CIFAR-10 CNN was trained for five epochs and achieved 67.59% clean test accuracy; results may differ with a different training setup.
- The experiments evaluate one model on one dataset.
- Gaussian noise is stochastic, so results may vary between runs unless random seeds and noise samples are controlled.
- Distortion strengths are not calibrated to equivalent levels of perceptual severity.
- The current results do not include a detailed per-class analysis or repeated training runs across multiple random seeds.

## Future Work

- Improve reproducibility by controlling random seeds and reusing fixed distortion samples.
- Repeat experiments to assess the stability of the results.
- Analyse accuracy changes for individual CIFAR-10 classes.
- Investigate whether model architecture or training choices affect robustness.
- Explore robustness improvements, such as data augmentation during training.
- Complete a research report discussing methodology, results, limitations, and potential extensions.

## Reproducibility

The notebooks document the experimental workflow, model implementation, distortion procedures, and evaluation results.

To reproduce the experiments, use a compatible Python environment with PyTorch, Torchvision, NumPy, Pillow, and Matplotlib. Follow the notebook cells in order. Because training and random noise generation can introduce variation, exact numerical reproduction may require controlling random seeds and using the same model checkpoint and distortion samples.

## Author

**Berenice Olawale**  
Computer Science Engineering student  
John von Neumann University, Hungary

Interested in machine learning, computer vision, and software engineering.
