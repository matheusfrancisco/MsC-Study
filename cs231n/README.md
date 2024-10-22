# Stanford CS231n Summary


After watching all the videos of the famous Standford's CS231n http://cs231n.stanford.edu/
course that took place in 2017, i decided to take summary of the whole course 
to help me to remember and to anyone who would like to know about it.


## Table of Contents

* [Standford CS231n 2017 Summary](#standford-cs231n-2017-summary)
   * [Table of contents](#table-of-contents)
   * [Course Info](#course-info)
   * [01. Introduction to CNN for visual recognition](#01-introduction-to-cnn-for-visual-recognition)
   * [02. Image classification](#02-image-classification)
   * [03. Loss function and optimization](#03-loss-function-and-optimization)
   * [04. Introduction to Neural network](#04-introduction-to-neural-network)
   * [05. Convolutional neural networks (CNNs)](#05-convolutional-neural-networks-cnns)
   * [06. Training neural networks I](#06-training-neural-networks-i)
   * [07. Training neural networks II](#07-training-neural-networks-ii)
   * [08. Deep learning software](#08-deep-learning-software)
   * [09. CNN architectures](#09-cnn-architectures)
   * [10. Recurrent Neural networks](#10-recurrent-neural-networks)
   * [11. Detection and Segmentation](#11-detection-and-segmentation)
   * [12. Visualizing and Understanding](#12-visualizing-and-understanding)
   * [13. Generative models](#13-generative-models)
   * [14. Deep reinforcement learning](#14-deep-reinforcement-learning)
   * [15. Efficient Methods and Hardware for Deep Learning](#15-efficient-methods-and-hardware-for-deep-learning)
   * [16. Adversarial Examples and Adversarial Training](#16-adversarial-examples-and-adversarial-training)

# Course Info

- Website: http://cs231n.stanford.edu/

- Lectures link: https://www.youtube.com/playlist?list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk

- Full syllabus link: http://cs231n.stanford.edu/syllabus.html

## 01. Introduction to CNN for visual recognition

## 02. Image classification
- Image classification problem has a lot challenges like illumination and viewpoints.
- An image classification algorithm can be solved with K nearest neighbordhood (KNN)
    - Hyperparameters: K, distance metric, etc.
    - K is the number of neighbors we are comparing to
    - Distance measures: L2 distance, L1 distance
        - L1 distance: sum of absolute differences (manhattan distance)
        - L2 distance: sum of squared differences (euclidean distance)
        L1 = d(p,q) = sqrt((p1-q1)^2 + (p2-q2)^2 + ... + (pn-qn)^2) 
            - it just a pitaqgorean theorem in n dimensions

