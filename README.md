# Human Activity Recognition using Machine Learning

This repository contains code for a machine learning project that aims to recognize human activities based on sensor data. The dataset used in this project consists of sensor measurements related to various physical activities performed by individuals, such as walking, standing, sitting, and more.

## Getting Started

### Prerequisites

Make sure you have the following Python libraries installed:

- [pyforest](https://pypi.org/project/pyforest/): A library that imports commonly used data science libraries.
- [skimpy](https://github.com/ysraell/skimpy): A tool for summarizing and visualizing data quickly.
- [scikit-learn](https://scikit-learn.org/stable/): A machine learning library for Python.
- [plotly](https://plotly.com/python/): A library for creating interactive plots.
- [pandas](https://pandas.pydata.org/): A powerful data manipulation and analysis library.
- [matplotlib](https://matplotlib.org/): A library for creating static, animated, and interactive visualizations in Python.
- [seaborn](https://seaborn.pydata.org/): A data visualization library based on matplotlib.

You can install these libraries using `pip`:

```bash
pip install pyforest skimpy scikit-learn plotly pandas matplotlib seaborn
```

### Data

This project assumes you have two CSV files, 'train.csv' and 'test.csv', containing the training and testing data, respectively. These files should have the necessary features and labels for activity recognition.

## Project Overview

1. **Data Preprocessing**: Load the training and testing datasets, concatenate them, and perform initial data exploration.

2. **Data Standardization**: Standardize the dataset using `StandardScaler` to ensure that all features have a mean of 0 and a standard deviation of 1.

3. **Dimensionality Reduction**: Apply Principal Component Analysis (PCA) to reduce the dimensionality of the dataset while preserving 90% of the variance.

4. **Ensemble Learning**: Create an ensemble model using the following machine learning algorithms:
   - Logistic Regression
   - Support Vector Machine (SVM)
   - K-Nearest Neighbors (KNN)
   - Naive Bayes

5. **Model Evaluation**: Evaluate the performance of individual algorithms and the ensemble model using metrics such as accuracy, classification report, and confusion matrix.

6. **Visualization**: Visualize the confusion matrix using a heatmap and create a Sankey diagram to illustrate the flow of predictions.

## Usage

You can run the code by executing the provided Python script. Make sure to have the required dataset files ('train.csv' and 'test.csv') in the same directory as the script.

## Results

The project demonstrates the following key findings:

- The SVM algorithm achieves the highest accuracy of 96% for human activity recognition.
- The ensemble model also selects SVM as the best-performing algorithm for this use case.

## Contacts

For questions or inquiries about this project, please contact:

- [Harshit Sharma](mailto:harshit2531937@gmail.com)
