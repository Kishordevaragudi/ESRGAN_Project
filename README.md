# ESRGAN_Project

This repository contains an op-for-op PyTorch reimplementation of ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

# Abstract
The Super-Resolution Generative Adversarial Network (SRGAN) is a seminal work that is capable of generating realistic textures during single image super-resolution. However, the hallucinated details are often accompanied with unpleasant artifacts. To further enhance the visual quality, we thoroughly study three key components of SRGAN - network architecture, adversarial loss and perceptual loss, and improve each of them to derive an Enhanced SRGAN (ESRGAN). In particular, we introduce the Residual-in-Residual Dense Block (RRDB) without batch normalization as the basic network building unit. Moreover, we borrow the idea from relativistic GAN to let the discriminator predict relative realness instead of the absolute value. Finally, we improve the perceptual loss by using the features before activation, which could provide stronger supervision for brightness consistency and texture recovery. Benefiting from these improvements, the proposed ESRGAN achieves consistently better visual quality with more realistic and natural textures than SRGAN.

# How to run the Project 
1. git clone : https://github.com/Kishordevaragudi/ESRGAN_Project.git
2. conda create -n env_name python==3.8 -y
3. pip install -r requirements.txt
4. streamlit run app.py
   

