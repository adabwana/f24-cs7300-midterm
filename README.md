# F24-CS7300-MIDTERM

### Jaryt Salvo
**Date:** 10-26-2024

**Fall 2024 | CS7300 Unsupervised Learning**

*************

## **[Please click HERE to view the workbook](https://adabwana.github.io/f24-cs7300-midterm/).**

This project focuses on implementing and comparing different unsupervised learning techniques, specifically autoencoders and dimensionality reduction methods. The project is divided into two main questions, each addressing a specific aspect of unsupervised learning.

### Question 1: Super-resolution with Autoencoders (50 Points)

In this part, we implement a super-resolution task using ***autoencoders*** on the `MNIST` dataset. The main steps include:

1. **Data Preprocessing:** Loading the `MNIST` dataset and adding noise to the original images.
2. **Autoencoder Design:** Implementing a convolutional autoencoder using PyTorch.
3. **Model Training:** Training the autoencoder to reconstruct clean images from noisy inputs.
4. **Evaluation:** Plotting training curves and visualizing results on test data.

Key features of the implementation:
                            
- Custom `MNIST` dataset loader
- Convolutional autoencoder architecture
- Noise addition function for data augmentation
- Training and evaluation loops
- Visualization of original, noisy, and reconstructed images

### Question 2: Dimensionality Reduction (50 Points)

This section compares two dimensionality reduction techniques: ***Principal Component Analysis (PCA)*** and ***Autoencoders***, using the `Fashion MNIST` dataset. The main steps include:

1. **Data Preprocessing:** Loading the `Fashion MNIST` dataset.
2. **PCA Implementation:** Using scikit-learn's PCA for baseline dimensionality reduction.
3. **Autoencoder Implementation:** Designing and training an autoencoder for dimensionality reduction.
4. **Comparison:** Evaluating both methods for reduced dimensionalities of 1, 2, and 3.
5. **Visualization:** Displaying reconstructed images and training curves.

Key features of the implementation:

- Custom `Fashion MNIST` dataset loader
- Flexible autoencoder architecture for different encoded dimensions
- PCA implementation using scikit-learn
- Comparative analysis of PCA and autoencoder performance
- Visualization of reconstructed images for both methods

### Technologies and Libraries Used:

- Python as the primary programming language
- `PyTorch` for neural network implementation
- `NumPy` for numerical computations
- `Matplotlib` for data visualization
- `scikit-learn` for PCA implementation

This project demonstrates the application of ***autoencoders*** in both super-resolution tasks and dimensionality reduction, showcasing their versatility in unsupervised learning. It also provides a comparative analysis between traditional (***PCA***) and modern (***autoencoder***) dimensionality reduction techniques, offering insights into their respective strengths and use cases.

The code for both questions is implemented in separate Python scripts (`question_1.py` and `question_2.py`), allowing for modular development and easy comparison of results."
