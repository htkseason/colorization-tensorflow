# Tensorflow Colorization  
A very rough colorization network based on CNN and tensorflow.  
  
  
### Result on Celeba  
<img src="https://github.com/htkseason/Colorization-Tensorflow/blob/master/demo-celeba.png" width="75%" alt="demo-celeba" />  
  
---  
  
### Network Structure  
Gray Image Input --> 64×64×1  
Feature Extract Layer 1 --> 64×64×32  
Feature Extract Layer 2 --> 32×32×64  
Feature Extract Layer 3 --> 16×16×128  
Feature Extract Layer 4 --> 8×8×256  
Feature Extract Layer 5 --> 4×4×512  
Deconvolution Layer 1 --> 8×8×256  
Deconvolution Layer 2 --> 16×16×128  
Deconvolution Layer 3 --> 32×32×64  
Deconvolution Layer 4 --> 64×64×32  
Concat Deconvolution Layer 4 with Feature Extract Layer 1 --> 64×64×64  
Deconvolution Layer 5 (output) --> 64×64×3 (RGB)  
  
loss = mean(square(output-ground_truth))  