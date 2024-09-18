
# Master’s Thesis Project

### Achieving Sketch-to-3D Shape Transformation using the Sketch-A-Shape Architecture

**Liulihan Kuang**

**AU-ID: AU636049**

**Student number: 201906612**

**Department of Computer Engineering, Aarhus University**  

**September 2024**

Welcome to the GitHub repository for Liulihan Kuang's master’s thesis project. This repository contains the source code and resources for the project, structured into two main stages of training and inference. 

## Project Overview

This project focuses on transforming 2D sketches into 3D shapes using the Sketch-A-Shape architecture. The implementation includes two key training stages, as well as inference capabilities.

### Directory Structure

1. **Training Stage 1 (VQ-VAE Training)**  
   This directory contains the implementation of the VQ-VAE (Vector Quantized Variational AutoEncoder) architecture, including:
   - Network architecture files
   - Training scripts
   - Testing scripts
   - Plotting scripts for visualizing results

2. **Training Stage 2 and Inference (Transformer-based Model)**  
   This directory contains files for the transformer-based model used in the second training stage and inference, including:
   - Transformer architecture
   - Training scripts for Stage 2
   - Pretrained VQ-VAE architecture from Stage 1
   - Inference scripts
   - Input sketch samples and output 3D shapes
   - Plotting scripts for evaluating model performance

## Setup Instructions

To replicate the training and inference processes, follow these steps:

1. **Dataset Download**  
   Download the dataset required for both training stages using the following command:

   ```bash
   wget https://clip-forge-pretrained.s3.us-west-2.amazonaws.com/exps.zip
   unzip exps.zip
   ```

2. **Environment Setup**  
   Create and activate the conda environment for this project:

   ```bash
   conda env create -f environment.yml
   conda activate master_project
   ```

## Running the Code

You can execute the training and inference stages with the following commands:

- **Stage 1 (VQ-VAE Training):**
  ```bash
  python train.py --dataset_path /path/to/dataset/
  ```

- **Stage 2 (Transformer-based Model Training):**
  ```bash
  python training_stage2.py --dataset_path /path/to/dataset/
  ```

- **Inference (Generate 3D Shape from Sketch):**
  
  ```bash
  python inference.py --save_path /path/to/output/
  ```

> **Note:** Both training and inference stages require a significant amount of GPU VRAM. Ensure that your system meets the necessary hardware requirements.
>
> Sorry to say that i actually don't know what the minimum hardware requirements are, because I just tried to adjust the transformer architectures to run it,it was trained and inferenced on a NVIDA A40 which has 48GB of VRAM, and it used up to 46GB VRAM to both train and inference.
>
> 
