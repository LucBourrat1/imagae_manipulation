# Presentation
This is an implementation of the following image manipulation models:
- vanilla autoencoder
- convolutional autoencoder
- variational autoencoder
- GAN
- DCGAN
- conditional GAN
- Pix2Pix GAN
- cycle GAN
- style GAN
- super-resolution GAN

# Installation
Needs to clone the repo, and prepare a conda environment with latest pytorch.
```bash
git clone https://github.com/LucBourrat1/image_manipulation
conda create -n image_manip python=3.11 -y
conda activate image_manip
pip3 install torch torchvision torchaudio
pip install ./image_manipulation
```