---
layout: post
title: "Dog Vision – Deep Learning Image Classifier"
permalink: /dog-vision/
---

## Overview
This project applies **transfer learning** to build a convolutional neural network that classifies dog breeds from images.  
Inspired by the Stanford Dogs dataset challenge.

## Key Skills
- TensorFlow / Keras  
- Transfer learning  
- Data augmentation  
- Model optimization  

## Approach
1. Loaded dataset of dog images  
2. Applied augmentation (random flip, rotation, zoom)  
3. Used **EfficientNet** as feature extractor  
4. Trained final classifier head  
5. Evaluated accuracy, precision, and confusion matrix  

## Results
- Fast convergence using pretrained weights  
- Clear improvement vs. training from scratch  
- Visualized sample predictions  

## Code Snippet
```python
model = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=(224,224,3))
