# Predictive Loan Default Analysis: Leveraging Machine Learning for Risk Assessment

## Overview
This project aims to predict loan default using machine learning techniques. It utilizes various classification models including Decision Tree Classifier, Logistic Regression, and Random Forest Classifier to analyze and predict loan default risks. The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/datasets/nikhil1e9/loan-default) and consists of several features such as age, income, credit score, loan amount, employment details, and more.

## Steps Involved
1. **Importing Required Libraries:** Libraries such as NumPy, Pandas, Seaborn, Matplotlib, and Scikit-learn are imported for data manipulation, visualization, preprocessing, model building, and evaluation.
2. **Load Dataset:** The dataset containing information about loan applicants is loaded into the environment.
3. **Data Visualization:** Visualizations are created to gain insights into the distribution and relationships among different variables in the dataset.
4. **Encoding Categorical Variables:** Categorical variables are encoded to convert them into numerical format, suitable for machine learning algorithms.
5. **Feature Scaling:** Features are scaled using StandardScaler to standardize the range of independent variables.
6. **Train Test Split:** The dataset is split into training and testing sets to train and evaluate the machine learning models.
7. **Addressing Class Imbalance with SMOTE:** Synthetic Minority Over-sampling Technique (SMOTE) is used to address class imbalance issues.
8. **Importing Machine Learning Models and Evaluation Metrics:** Decision Tree Classifier, Logistic Regression, and Random Forest Classifier models are imported along with evaluation metrics to assess model performance.
9. **Model Training:** The models are trained on the training data.
10. **Model Evaluation:** The accuracy of different classification models is evaluated using testing data.
11. **Calculating AUC Scores for Models:** Area Under the Curve (AUC) scores are calculated to measure the performance of the models.
12. **Plotting ROC Curves for Models:** ROC curves are plotted to visualize the true positive rate against the false positive rate for different classification models.
13. **Conclusion:** The project concludes by summarizing the findings and highlighting the accuracy achieved by each model. Logistic Regression achieved the highest accuracy of 88.34%, followed by Random Forest Classifier with an accuracy of 88.33%, and Decision Tree Classifier with an accuracy of 80.33%.

## Usage
1. Clone the repository.
2. Install the required libraries mentioned in `requirements.txt`.
3. Install Jupyter Notebook using the following command:pip install notebook
4. Run the `loan_default_prediction.ipynb` notebook to execute the project.





