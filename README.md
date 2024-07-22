# Advanced Classification of Animal Sounds using Spectrogram Analysis.-

## Project Overview
Our project focuses on the development of a machine-learning model to classify animal sounds from spectrograms. Utilizing advanced audio processing techniques and a Convolutional Neural Network (CNN) built with TensorFlow and Keras, the model aims to distinguish between various categories of animal sounds, including household pets, farm animals, and wild animals.

## Dataset
The dataset comprises over 150 audio recordings of animal sounds, sourced from platforms such as YouTube and Google's AudioSet. The sounds are categorized into:

- Household Pets (Dogs, Cats)
- Farm Animals (Sheep, Goats)
- Wild Animals (Big Cats, Wild Dogs)

Each audio file is processed to extract Mel Spectrograms and Mel Frequency Cepstral Coefficients (MFCCs), which serve as the input features for the machine learning model.

## Technologies Used
Python: Programming language
TensorFlow & Keras: Machine learning framework for building and training the CNN model
Librosa: Library for audio and music analysis
Matplotlib & Seaborn: Libraries for data visualization
Scikit-learn: Library for model evaluation and utilities

## Model Architecture
The Convolutional Neural Network (CNN) used in this project includes:

- Convolutional layers with ReLU activation
- MaxPooling layers for downsampling
- Dense layers with Dropout for regularization
- Softmax output layer for classification

## Results
Training Accuracy: 98.94%

Test Accuracy: 87.5%

**Confusion Matrix Analysis:** The model performed well in classifying 'Big Cats' and 'Wild Dogs', while there was room for improvement in distinguishing 'Cats' and 'Sheep'.

## Future Work
- Implement more sophisticated data augmentation and regularization techniques to further reduce overfitting.
- Experiment with different machine learning and deep learning models.
- Expand the dataset to include more diverse and challenging animal sounds.


