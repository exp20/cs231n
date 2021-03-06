

### Goals

In this assignment you will practice putting together a simple image classification pipeline based on the k-Nearest Neighbor or the SVM/Softmax classifier. The goals of this assignment are as follows:

- Understand the basic **Image Classification pipeline** and the data-driven approach (train/predict stages)
- Understand the train/val/test **splits** and the use of validation data for **hyperparameter tuning**.
- Develop proficiency in writing efficient **vectorized** code with numpy
- Implement and apply a k-Nearest Neighbor (**kNN**) classifier
- Implement and apply a Multiclass Support Vector Machine (**SVM**) classifier
- Implement and apply a **Softmax** classifier
- Implement and apply a **Two layer neural network** classifier
- Understand the differences and tradeoffs between these classifiers
- Get a basic understanding of performance improvements from using **higher-level representations** as opposed to raw pixels, e.g. color histograms, Histogram of Gradient (HOG) features, etc.


### Q1: k-Nearest Neighbor classifier (20 points)
### Q2: Training a Support Vector Machine (25 points)
### Q3: Implement a Softmax classifier (20 points)

Работа с классификаторами kNN, SVM, Softmax выполнена в **lab1.ipynb**. А реализованы в **"./scripts/classifiers/"**.
Далее все файлы заданий (assignment1,2,3) расположены согласно оригинальной структуре курсов cs231n (2020).

### Q4: Two-Layer Neural Network (25 points)

The notebook **two\_layer\_net.ipynb** will walk you through the implementation of a two-layer neural network classifier.

### Q5: Higher Level Representations: Image Features (10 points)

The notebook **features.ipynb** will examine the improvements gained by using higher-level representations
as opposed to using raw pixel values.
