# RNN Implementation

This repository contains Jupyter notebooks and Python scripts demonstrating the implementation of Recurrent Neural Networks (RNNs) for sequence data analysis. The project includes a sentiment analysis model trained on the IMDB movie review dataset and an interactive Streamlit application for classifying movie reviews as positive or negative.

### Dataset Source:

The dataset used in this project is the IMDB movie review dataset, which provides a collection of movie reviews labeled as either positive or negative. It is accessible directly through TensorFlow/Keras datasets.

### Key Features:

- **Embedding Layer Demonstration**: Showcases how embedding layers are used to map words into continuous vector spaces for sentiment analysis.
- **Simple RNN Implementation**: Implements a basic RNN model for binary sentiment classification.
- **Prediction Notebook**: A dedicated notebook for loading the trained model and predicting sentiments of sample reviews.
- **Streamlit Integration**: Provides an interactive application for real-time sentiment prediction of user-submitted reviews.

### Technologies Used:

- **Python**: Core programming language used throughout the project.
- **TensorFlow & Keras**: Frameworks for building and training the RNN model.
- **Pandas**: Used for dataset analysis.
- **Streamlit**: Creates an interactive dashboard for sentiment classification.
- **Matplotlib & Seaborn**: Utilized for visualizations in other notebooks.

### Simple RNN Model:

The model is implemented in `simplernn.ipynb`:
- Trained on the IMDB dataset using a ReLU activation function.
- Embedding layers are used to preprocess the text into numerical vectors.
- The model predicts whether a given review is positive or negative based on its text.

### Prediction Notebook:

The `prediction.ipynb` notebook demonstrates:
- Loading the pre-trained RNN model from `simple_rnn_imdb.h5`.
- Preprocessing and encoding user-provided text for prediction.
- Decoding text reviews for visualization and analysis.
- Predicting sentiment (positive or negative) along with a confidence score.

#### Streamlit Output:
![Output](https://github.com/AashishSaini16/RNN-Implementation/blob/main/output.JPG)
