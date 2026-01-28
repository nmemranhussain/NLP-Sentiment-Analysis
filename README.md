# NLP-Sentiment-Analysis

Analyze Titanic passenger data with Logistic Regression &amp; Random Forest. Identify key factors influencing survival.

## Basic Information
**Names:** N M Emran Hussain  
**Email:** nmemranhussain2023@gmail.com  
**Date:** October 2025  
**Model Version:** 1.0.0  
**License:** [Apache License Version 2.0,](LICENSE)

## Intended Use
**Purpose:** The primary purpose of this project is to develop and evaluate a high-performance Sentiment Analysis Pipeline utilizing the FastText supervised learning library.  
**Intended Users:** The intended users of this project are Data Scientists, Machine Learning Engineers, and NLP Researchers, hiring managers or technical recruiters looking for an efficient, scalable template for text classification.  
**Out-of-scope Uses:** This project is not intended for end-consumers seeking a finished application, entities requiring multi-class sentiment detection, or organizations processing highly informal, short-form text, as the current model is a backend tool specifically optimized for binary classification of structured IMDb reviews.  

## Dataset
**Dataset Name:** [IMDb Large Movie Review Dataset.]([https://www.kaggle.com/c/titanic/data?select=train.csv](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews))
**Number of Samples:** The dataset contains a total of 50,000 highly polar movie reviews.The dataset is perfectly balanced, with 25,000 positive and 25,000 negative reviews.    
**Features Used:** The FastText sentiment analysis pipeline utilizes the raw text content of movie reviews as its primary input feature, paired with binary sentiment labels for classification. To enhance predictive accuracy, the model incorporates engineered features such as word bi-grams to capture contextual sentiment and subword information to better represent rare words or morphological patterns. Furthermore, the project leverages a bag-of-words approach, allowing the model to analyze the presence and frequency of specific terms within the reviews to distinguish between positive and negative sentiments.  
**Training Set (Approx. 90%):** 22,500 examples were used for the actual model training.  
**Validation Set (Approx. 10%):** 2,500 examples were set aside from the original training split to guide the Autotune hyperparameter optimization.  
**Test Set (50% of total):** 25,000 examples remained strictly for the final model evaluation to ensure unbiased performance metrics.  

### Data Dictionary

| Column Name     | Modeling Role  | Measurement Level | Description                            |
|-----------------|----------------|-------------------|----------------------------------------|
|Label	| Target (Dependent Variable)|	Nominal (Binary)	|The sentiment classification of the review. In the raw data, it is numeric (0 or 1). In the FastText format, it is converted to __label__negative or __label__positive.|  
|Text	| Feature (Independent Variable)	| Text / String	|The full string content of the movie review used for training and inference. This undergoes tokenization and N-gram processing during modeling.|

### Differences Between Training and Test Data
- The training data includes the target variable (Survived), allowing us to train and evaluate the model, while the test data lacks this target, so it’s used solely for generating predictions to assess model performance on unseen data.
- All other feature columns are the same between the training and test datasets.

## Model Details
### Architecture  
- This model card utilizes linear model such as **Logistic Regression**. As an alternative model **Random Forest** is used.  

### Evaluation Metrics  
- AUC (Area Under the ROC Curve): Measures the model's ability to distinguish between positive and negative classes.

### Final Values of Metrics for All Data using 'logistic regression' model:

| Dataset     | AUC   | 
|-------------|-------|
| Training    | 0.78  | 
| Validation  | 0.80  |
| Test        | 0.76  | 

### Columns Used as Inputs in the Final Model
The following columns were used as inputs (features) in the final model:
- Pclass
- Sex
- Age
- SibSp
- Parch
- Fare
- Embarked

### Column(s) Used as Target(s) in the Final Model
- **Target Column:** Survived

### Type of Models
* **[Logistic Regression Classifier](https://github.com/nmemranhussain/titanic-ml-models/blob/main/Titanic_logistic%20(1).ipynb)**
* **[Random Forest Classifier](https://github.com/nmemranhussain/titanic-ml-models/blob/main/Titanic_RF.ipynb)**

### Software Used to Implement the Model
- **Software:** Python (with libraries such as Pandas, Scikit-learn, seaborn & matplotlib)

### Version of the Modeling Software: 
- **'pandas'**: '2.2.2',
- **'scikit-learn'**: '1.4.2',
- **'seaborn'**: '0.13.2',
- **'matplotlib'**: '3.8.4**

### Hyperparameters or Other Settings of the Model
The following hyperparameters were used for the 'logistic regression' model:
- **Solver:** lbfgs
- **Maximum Iterations:** 100
- **Regularization (C):** 1.0
- **Features used in the model**: ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
- **Target column**: Survived
- **Model type**: Logistic Regression
- **Hyperparameters**: Solver = lbfgs, Max iterations = 500, C = 1.0
- **Software used**: scikit-learn sklearn.linear_model._logistic

The following hyperparameters were used for the 'random forest' as an alternative model:
- **Columns used as inputs**: ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'], 
- **Target column**: 'Survived',
- **Type of model**: 'Random Forest Classifier',
- **Software used**: 'scikit-learn',

## Quantitative Analysis

### Plots Related to Data or Final Model
 
![Plot of Survival Rate Vs. Passenger Class](SR_by_Class.png) 

**Description**: Passengers in 1st class had the highest survival rate, followed by those in 2nd class. 3rd class passengers had the lowest survival rate.

![Plot of Survival Rate Vs. Passenger Gender](SR_by_Gender.png) 

**Description**: Females had a significantly higher survival rate than males, aligning with the negative coefficient for the "Sex" feature in the logistic regression model.

![Plot of Survival Rate Vs. Passenger Age](SR_by_Age.png) 

**Description**: Children (ages 0-12) had the highest survival rate, while seniors (ages 50-80) had the lowest. Young adults and adults had relatively similar survival rates, though slightly lower than children.

## Potential Impacts, Risks, and Uncertainties using Logistic Regression & Random Forest Model ##
Logistic regression offers a powerful tool for classification tasks. However, it is crucial to acknowledge its limitations. The model assumes a linear relationship between features and the outcome, which could overlook complex patterns in the data. This can lead to biased predictions, particularly when dealing with sensitive attributes like gender or class. Additionally, the probabilistic nature of the output can be misinterpreted as deterministic, potentially leading to misinformed decisions. To mitigate these risks and promote responsible AI practices, this model development employed several strategies. First, the training data was thoroughly examined for potential disparities related to gender and class. Second, interpretability tools from libraries like PiML were used to analyze the model's decision-making process and its impact on different groups. By incorporating these responsible AI practices, we aimed to ensure fairer and more transparent outcomes from the logistic regression model.

While random forests boast strong performance in classification tasks, they also present challenges. Their complex structure can be difficult to interpret, hindering explainability. Despite resilience to noise, random forests can still be susceptible to overfitting if not carefully tuned. Furthermore, biased training data can lead to unfair predictions. Additionally, their reliance on multiple decision trees can obscure the true influence of individual features, and their performance is sensitive to data quality and hyperparameter tuning. This can lead to unexpected patterns with potentially positive or negative consequences. Similar to the logistic regression model, responsible AI practices were prioritized during development. The training data was rigorously scrutinized for biases, particularly regarding gender and class. Tools from InterpretML were utilized to understand the model's behavior and its potential impact on protected groups. By fostering responsible AI throughout the development process, we aimed to ensure fairer and more interpretable predictions from the random forest model.
