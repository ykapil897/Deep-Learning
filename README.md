# Deep Learning Course and Project

This repository contains materials for a Deep Learning course, along with an implementation of an Image Super-Resolution project for Satellite Imagery.

## Course Materials

### Lecture Notes
- Deep Learning Fundamentals
- Neural Network Architectures
- Convolutional Neural Networks
- Recurrent Neural Networks
- Generative Models
- Transformers and Advanced Topics

### Labs
1. **Introduction to PyTorch**: Basics of PyTorch
2. **Autoencoders**: Implementation of different autoencoder architectures including VAE
3. **Advanced CNN**: Implementation of advanced CNN architectures
4. **Sequence Models**: Implementation of GRU and other sequence models
5. **Object Detection and Segmentation**: R-CNN, PSPNet implementations
6. **Generative Models**: 
   - GAN implementations
   - Conditional GAN
   - CycleGAN
   - StyleGAN
   - Diffusion Models (DDPM)
7. **Natural Language Processing**:
   - Transformer models
   - Image captioning using CNN and LSTM

### Examinations
- Major Examination
- Minor Examination
- Quiz 2
- Quiz 3

## Project: Image Super-Resolution for Satellite Imagery

This project implements different deep learning models for image super-resolution applied to satellite imagery. The models are organized into subdirectories and connected into a single pipeline.

### Project Structure

- **Super-Resolution Model/**: Contains the super-resolution model and its associated files.
- **Denoising Model/**: Contains the deep learning model for denoising and its associated files.
- **SRGAN-for-Satellite-Image-Super-Resolution/**: Implementation of SRGAN model specifically for satellite imagery enhancement.

### Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### SRGAN for Satellite Image Super-Resolution

The SRGAN-for-Satellite-Image-Super-Resolution folder contains an implementation of Super-Resolution Generative Adversarial Network (SRGAN) specifically optimized for satellite imagery. This project was developed in collaboration with [viveksapkal2793](https://github.com/viveksapkal2793) and adopts a different approach to the super-resolution problem.

Alternate project link: [https://github.com/viveksapkal2793/Image-Super-Resolution-for-Satellite-Imagery](https://github.com/viveksapkal2793/Image-Super-Resolution-for-Satellite-Imagery)

#### SRGAN Project Features:
- Implementation of SRGAN architecture for 4x upscaling
- WGAN-GP variant for improved training stability
- Pytorch implementation with SSIM loss
- Visualization tools for qualitative comparison
- Pre-trained models available in the cp/ directory

#### Usage:
- Training: Use `train.py` or `train-wgangp.py` for different training approaches
- Inference: Use `sr.py` for single image super-resolution
- Evaluation: `eval.py` for quantitative metrics and `eval-compare.py` for comparison

### Usage

Usage of both Models is described in their respective directories.

## Repository Structure

```
.
├── exam/
│   ├── major.pdf
│   ├── minor.pdf
│   ├── quiz_2.pdf
│   └── quiz_3.pdf
├── labs/
│   ├── CGAN CycleGAN StyleGAN.pdf
│   ├── CNN Autoencoder.ipynb
│   ├── conditional-GAN-generating-fashion-mnist.ipynb
│   ├── DDPM_lab11.ipynb
│   ├── GAN.ipynb
│   ├── GRU.ipynb
│   ├── image-caption-generator-using-cnn-and-lstm.ipynb
│   ├── Lab 1 - Introduction_ Basics of PyTorch.ipynb
│   ├── Lab 2_ AutoEncoder _ CSL4020_ Deep Learning.ipynb
│   ├── Lab 4 _ Object Detection and Segmentation.pdf
│   ├── Lab_4__Sequence_Models.ipynb
│   ├── lab_4_temperature_series.pdf
│   ├── Lab3_Advanced_CNN_Code.ipynb
│   ├── PSPNet Implementation.ipynb
│   ├── RCNN Implementation.ipynb
│   ├── stylegan-mnist.ipynb
│   ├── stylegan.ipynb
│   ├── transfer_learning_tutorial.ipynb
│   ├── transformer_NLP.ipynb
│   ├── VAE.ipynb
│   └── ViT.ipynb
└── lecture_notes/
    ├── Deep Learning.pdf
    └── Various numbered lecture PDFs
```