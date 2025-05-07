# BASIC MACHINE LEARNING

## Table of Contents
1. [What is Machine Learning?](#1-what-is-machine-learning)
2. [Machine Learning Workflow](#2-machine-learning-workflow)
3. [Types of Machine Learning](#3-types-of-machine-learning)
4. [Data Preparation Techniques](#4-data-preparation-techniques)
5. [Model Evaluation Metrics](#5-model-evaluation-metrics)
6. [Core ML Architectures Overview](#6-core-ml-architectures-overview)
7. [Advanced Machine Learning Concepts](#7-advanced-machine-learning-concepts)
8. [Visualization Guide](#8-visualization-guide)

## 1. What is Machine Learning?

Machine learning is a branch of artificial intelligence (AI) that focuses on building systems that can learn from data, identify patterns, and make decisions with minimal human intervention.

## 2. Machine Learning Workflow

| Step | Description | Key Activities | Why It Matters |
|------|-------------|---------------|----------------|
| 1. Define the Problem | Set clear objectives | Identify specific questions or tasks for the machine to perform | A well-defined problem leads to appropriate model selection and evaluation criteria |
| 2. Collect Data | Gather relevant information | Source data from databases, APIs, web scraping, sensors, etc. | Your model is only as good as the data it learns from |
| 3. Data Cleaning & Preparation | Process raw data | Handle missing values, outliers, normalize/standardize data | Garbage in = garbage out; clean data is essential for accurate models |
| 4. Exploratory Data Analysis | Understand the data | Visualize patterns, anomalies, and relationships using charts and statistics | Reveals insights that guide feature selection and modeling decisions |
| 5. Feature Engineering | Create informative features | Transform existing data into more useful inputs (e.g., extracting day of week from dates) | Better features often lead to better models than complex algorithms alone |
| 6. Select a Model | Choose appropriate algorithm | Match algorithm to problem type (regression, classification) based on data characteristics | Different algorithms have different strengths and weaknesses |
| 7. Train the Model | Learn from patterns | Feed prepared data to the selected algorithm | This is where the actual "learning" happens - the model finds patterns in data |
| 8. Evaluate Performance | Assess accuracy | Test model on unseen data using appropriate metrics | Ensures the model generalizes well to new data, not just memorizing training data |
| 9. Tune Parameters | Optimize settings | Adjust hyperparameters to improve performance | Fine-tuning can significantly improve model performance |
| 10. Deploy & Monitor | Put into production | Implement model and track ongoing performance | Models can degrade over time as data patterns change |

## 3. Types of Machine Learning

### 3.1 Supervised Learning

| Type | Description | Common Algorithms | Real-world Applications | How It Works |
|------|-------------|-------------------|-------------------------|-------------|
| **Regression** | Predicts continuous values (numbers) | Linear Regression, Decision Trees, Random Forest, SVR | Price prediction, weather forecasting, sales projections | The model learns to predict a numeric value based on input features. For example, predicting house prices based on size, location, and amenities. |
| **Classification** | Assigns items to categories | Logistic Regression, Decision Trees, Random Forest, SVM, KNN | Email spam detection, medical diagnosis, image classification | The model learns to place inputs into distinct categories. For example, determining if an email is spam or not based on its content and sender information. |

### 3.2 Unsupervised Learning

| Type | Description | Common Algorithms | Real-world Applications | How It Works |
|------|-------------|-------------------|-------------------------|-------------|
| **Clustering** | Finds natural groupings | K-Means, Hierarchical, DBSCAN | Customer segmentation, anomaly detection, document grouping | The model discovers natural groups in data without labels. For example, grouping customers with similar buying behaviors. |
| **Dimensionality Reduction** | Reduces feature count | PCA, t-SNE, UMAP | Data visualization, noise reduction, feature compression | The model combines or transforms features to create a simpler representation while preserving important information. This helps with visualization and reducing computation time. |

### 3.3 Reinforcement Learning

| Algorithm | Description | Applications | How It Works |
|-----------|-------------|--------------|-------------|
| Q-Learning | Learns action-value function | Game playing, resource management | Builds a table of state-action pairs and their expected rewards, then chooses actions that maximize reward |
| DQN | Deep Q-Networks using neural networks | Complex games, robotic control | Uses neural networks to approximate the Q-function, allowing it to handle complex state spaces |
| Policy Gradients | Directly optimizes policy | Continuous control problems | Learns the best policy (mapping from states to actions) directly rather than through a value function |

## 4. Data Preparation Techniques

### 4.1 Handling Missing Data

| Method | Description | When to Use | Practical Example | When Not to Use |
|--------|-------------|-------------|-------------------|-----------------|
| Removal | Delete rows/columns with missing values | When missing data is random and limited | Removing a few customers with incomplete survey responses | When missing data is non-random or when data is scarce |
| Mean/Median Imputation | Replace with statistical measures | For numerical features with normal distribution | Filling in missing age values with the average age | When data is not normally distributed or when missingness carries meaning |
| Mode Imputation | Replace with most frequent value | For categorical features | Filling in missing gender with the most common gender | When the mode is not representative or when missing values have patterns |
| KNN Imputation | Estimate based on similar instances | When relationships between features exist | Estimating missing income based on similar people's income | When computational resources are limited or data is too sparse |
| Model-based Imputation | Use predictive models to fill gaps | For complex datasets with clear patterns | Predicting missing blood pressure values based on other health metrics | When the model could introduce bias or when features are weakly correlated |

### 4.2 Dealing with Outliers

| Detection Method | Description | Handling Approach | Why It Matters | When Not to Use |
|------------------|-------------|-------------------|----------------|-----------------|
| Visual Inspection | Box plots, scatter plots | Determine if genuine or errors | Outliers can significantly skew models, especially linear ones | For very large datasets where visualization is impractical |
| IQR Method | Values > 1.5×IQR from Q1/Q3 | Remove, transform, or cap values | Provides a statistical basis for identifying extreme values | For multimodal distributions where IQR may flag valid data points |
| Z-score | Values > 3 std dev from mean | Remove or transform based on domain knowledge | Works well for normally distributed data | For non-normal distributions where z-scores are misleading |
| DBSCAN | Density-based clustering | Identify clusters vs. outliers | Effective for multi-dimensional outlier detection | When computational resources are limited or when data is uniformly distributed |

### 4.3 Data Scaling

| Method | Formula | Result | Best For | Practical Example | When Not to Use |
|--------|---------|--------|----------|-------------------|-----------------|
| Min-Max Scaling | (X - min(X)) / (max(X) - min(X)) | Values between 0-1 | When bounded range is needed | Normalizing pixel values for image processing | When outliers are present (they'll compress most data into a small range) |
| Z-score Standardization | (X - mean(X)) / std(X) | Mean=0, SD=1 | Algorithms sensitive to scale (SVM, KNN) | Standardizing features for principal component analysis | When data is not normally distributed or when absolute values matter |
| Robust Scaling | (X - median(X)) / IQR | Robust to outliers | When outliers are present | Scaling financial data with extreme values | When complete range of data is important to preserve |
| Log Transform | log(X) | Reduces skewness | Highly skewed data | Transforming salary data where a few people earn much more | For data with zeros or negative values (without special handling) |

## 5. Model Evaluation Metrics

### 5.1 Classification Metrics

| Metric | Formula | Description | When to Use | Practical Example | When Not to Use |
|--------|---------|-------------|-------------|-------------------|-----------------|
| Accuracy | (TP + TN) / (TP + TN + FP + FN) | Overall correctness | Balanced classes | General performance in equal classes, like cat vs. dog classifier | When classes are imbalanced (e.g., rare disease detection) |
| Precision | TP / (TP + FP) | Exactness of positive predictions | When false positives are costly | Medical tests where false positives cause unnecessary anxiety | When recall is much more important than precision |
| Recall (Sensitivity) | TP / (TP + FN) | Completeness of positive predictions | When false negatives are costly | Cancer detection where missing a case is dangerous | When precision is much more important than recall |
| F1 Score | 2 * (Precision * Recall) / (Precision + Recall) | Harmonic mean of precision & recall | When balance between precision & recall is needed | Spam detection where both false positives and negatives matter | When costs of false positives and false negatives are very different |
| AUC-ROC | Area under ROC curve | Ranking quality across thresholds | Model comparison, threshold selection | Comparing different fraud detection models | When specific operating points (precision/recall) matter more than overall ranking |

### 5.2 Regression Metrics

| Metric | Formula | Description | When to Use | Practical Example | When Not to Use |
|--------|---------|-------------|-------------|-------------------|-----------------|
| MAE | (1/n) * Σ\|actual - predicted\| | Average absolute error | When all errors equally important | Predicting daily temperature | When larger errors should be penalized more heavily |
| MSE | (1/n) * Σ(actual - predicted)² | Penalizes larger errors | When outliers should be penalized | Stock price prediction where large errors are more problematic | When units matter or when interpretability is important |
| RMSE | √MSE | Error in original units | Standard metric for most regression problems | Housing price prediction | When resistance to outliers is required |
| MAPE | (1/n) * Σ\|(actual - predicted) / actual\| * 100% | Percentage error | When relative error matters | Sales forecasting where percentage deviation matters | When actual values can be zero or very close to zero |
| R² | 1 - (Residual SS / Total SS) | Variance explained by model | General goodness of fit | Measuring how well your model explains variations in the data | When comparing models across different datasets or when negative values are possible |

## 6. Core ML Architectures Overview

| Architecture | Key Characteristics | Best For | How It Works | Real-World Examples |
|--------------|---------------------|----------|-------------|---------------------|
| **Artificial Neural Networks (ANN)** | Multiple layers of connected neurons that process inputs to produce outputs | Pattern recognition, classification, regression | Processes data through layers of nodes, with each node applying activation functions to weighted inputs | Voice recognition, risk assessment, customer churn prediction |
| **Convolutional Neural Networks (CNN)** | Specialized for processing grid-like data using convolutional layers | Image & video analysis, visual recognition | Uses filters/kernels that slide over input data to detect patterns regardless of position | Facial recognition, medical image analysis, self-driving cars |
| **Recurrent Neural Networks (RNN/LSTM)** | Contains feedback loops to maintain "memory" of previous inputs | Sequential data, time series, text | Processes sequences by maintaining a state that captures information about previous inputs | Speech recognition, language translation, stock prediction |
| **Natural Language Processing (NLP)** | Specialized models for understanding human language | Text analysis, language tasks | Processes text using tokenization, embeddings, and specialized architectures | Chatbots, sentiment analysis, content summarization |
| **Time Series Models** | Focus on temporal patterns and dependencies | Forecasting, trend analysis | Analyzes patterns over time to predict future values | Stock market prediction, weather forecasting, demand planning |
| **Ensemble Methods** | Combine multiple models to improve performance | Reducing errors, increasing stability | Aggregates predictions from multiple models to get a better overall prediction | Kaggle competitions, production systems requiring high accuracy |

## 7. Advanced Machine Learning Concepts

### 7.1 Handling Imbalanced Data

| Technique | Description | Pros | Cons | Example Scenario | When Not to Use |
|-----------|-------------|------|------|-----------------|-----------------|
| Undersampling | Reduce majority class | Fast, addresses imbalance | Information loss | Reducing "normal" transactions in fraud detection | When every majority class example contains valuable information or when data is already limited |
| Oversampling | Duplicate minority class | No information loss | Risk of overfitting | Duplicating rare disease cases in medical diagnosis | When computational resources are limited or when exact duplication creates overfitting |
| SMOTE | Generate synthetic minority examples | Better generalization | Can create unrealistic samples | Creating synthetic examples of fraudulent transactions | When feature spaces have clear boundaries that synthetic samples might violate |
| Class Weights | Penalize misclassification of minority | No data modification | May not work for extreme imbalance | Giving more importance to rare classes in classification | When imbalance is too extreme (10,000:1 or more) or when the algorithm doesn't support weights |
| Focal Loss | Down-weight easy examples | Works well with deep learning | Requires careful tuning | Image classification with rare objects | For simple models or when computational resources are limited |

### 7.2 Cross-Validation Strategies

| Strategy | Description | Best For | Why It Matters | When Not to Use |
|----------|-------------|----------|---------------|-----------------|
| K-Fold | Split data into k equal parts | Standard datasets | Reduces bias from a single train/test split | For time series data or when observations are not independent |
| Stratified K-Fold | Maintain class distribution | Imbalanced datasets | Ensures each fold has similar class distribution | When stratification isn't possible or for regression problems |
| Time-based (Forward Chaining) | Train on past, test on future | Time series data | Respects temporal nature of data | For non-sequential data where time ordering doesn't matter |
| Group K-Fold | Keep related samples together | When data has natural groupings | Prevents data leakage from related samples | When observations are truly independent |
| Nested CV | Inner loop for tuning, outer for evaluation | When hyperparameter tuning | Provides unbiased performance estimates | When computational resources are severely limited |

### 7.3 Hyperparameter Tuning Methods

| Method | Description | Efficiency | When to Use | Practical Example | When Not to Use |
|--------|-------------|------------|-------------|-------------------|-----------------|
| Grid Search | Try all combinations | Low | Small hyperparameter space | Testing 5 learning rates and 5 regularization values | For large parameter spaces where exhaustive search is impractical |
| Random Search | Sample randomly | Medium | Larger spaces, limited compute | Exploring 100+ combinations with limited resources | When the parameter space is small enough for grid search or when reproducibility is critical |
| Bayesian Optimization | Build probabilistic model | High | Complex models, limited compute | Optimizing deep learning hyperparameters | When the objective function evaluation is very fast (grid search may be faster) |
| Genetic Algorithms | Evolutionary approach | Medium-High | Very large parameter spaces | Complex optimization problems with many parameters | When the fitness landscape is simple or when reproducibility is required |
| Population-Based Training | Parallel training with evolution | High | Deep learning, if resources available | Training multiple models in parallel, adopting best practices | When computational resources are limited or the problem is simple |
## 8. Visualization Guide

| Plot Type | Purpose | Data Requirements | Example Use Case | What It Shows You |
|-----------|---------|-------------------|------------------|------------------|
| Histogram | Show distribution | Single numeric variable | Age distribution of customers | Shows frequency of values within ranges |
| Box Plot | Show statistics & outliers | Single numeric variable | Salary distribution by department | Shows median, quartiles, and outliers at a glance |
| Scatter Plot | Show relationships | Two numeric variables | Height vs. Weight correlation | Shows correlation and patterns between two variables |
| Line Chart | Show trends over time | Time series data | Stock price movement | Shows how values change over time |
| Heatmap | Show correlations | Numeric matrix | Feature correlation matrix | Shows strength of relationships between multiple variables |
| Bar Chart | Compare categories | Categorical data | Sales by region | Shows comparison between discrete categories |
| Violin Plot | Distribution & density | Numeric by category | Test scores across schools | Shows full distribution shape alongside summary statistics |
| Pair Plot | Multiple relationships | Multiple numeric variables | Relationships between all features | Shows correlations between multiple variable pairs |

---