# PhytNet_Cocoa

This repo provides code to train Convolutional Neural Networks (CNN) such as PhytNet or ResNet for the classification of cocoa diseases using semi-supervised learning. 
This repository contains the code and resources needed to train the model progressively to learn and classify more difficult cases of cocoa disease.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Directory Structure](#directory-structure)
- [License](#license)

## Overview
Here we employ a semi-supervised learning approach, allowing the CNN to improve its classification accuracy on more challenging cases of cocoa diseases over time. 
The model begins with easy labelled data and iteratively incorporates more difficult cases of cocoa disease with labels, refining its ability to identify and categorise various diseases affecting 
cocoa plants.

## Features
- **Semi-supervised learning**: Leverages both labelled and unlabeled data.
- **Progressive learning**: Enhances classification accuracy on difficult cases.
- **CNN architecture**: Utilizes Convolutional Neural Networks for image classification.
- **Cross-validation**: Includes mechanisms for validating model performance.

## Installation
To install and set up PhytNet_Cocoa, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/jrsykes/CocoaReader.git
   cd CocoaReader/CocoaNet/PhytNet_Cocoa
   ```

## Directory Structure
- `Sweep/`: Hyperparameter sweep scripts with Weights and Biases.
- `FinalTrain/`: Scripts and configurations for final model training.
- `cross-validation/`: Scripts for cross-validation.


## License
This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for more details.

---

Feel free to customise the sections further based on your specific requirements and additional details about the project.
