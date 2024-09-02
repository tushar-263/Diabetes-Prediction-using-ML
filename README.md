# Diabetes-Prediction-using-Machine-Learning

This Python code implements a K-Nearest Neighbour model to predict the likelihood of diabetes prediction based on various features. It uses the scikit-learn library for data preprocessing, model training, and evaluation.

## Data

The code uses the `diabetes.csv` dataset, which contains the following features:


- `Pregnancies`: Number of times pregnant
- `glucose`: Plasma glucose
- `blood_pressure`: Diastolic blood pressure (mm Hg)
- `skin_thickness`: Triceps skin fold thickness (mm)
- `insulin`: 2-Hour serum insulin (mu U/ml)
- `bmi`:Body mass index (weight in kg/(height in m)^2)
- `age`: Age of the patient(years)

## Preprocessing

The code separates the features (`x`) and the target variable (`y`) from the dataset. It then splits the data into training and testing sets using `train_test_split` from `sklearn.model_selection`.

## Model Training

The logistic regression model is initialized using `LogisticRegression` from `sklearn.linear_model`. The model is trained on the training data using the `.fit()` method.

## Model Evaluation

The trained model's accuracy is evaluated on the test data using the `.predict()` method and `accuracy_score` from `sklearn.metrics`.

## Prediction System

The code provides an example of how to use the trained model for prediction. It takes a new set of input data and passes it to the `.predict()` method to obtain the prediction (0 or 1). Based on the prediction, it prints either "Diabetic" or "Non-Diabetic".

## Usage

1. Ensure you have the required Python libraries installed (`numpy`, `pandas`, `scikit-learn`).
2. Download the `diabetes.csv` dataset and place it in the same directory as the code.
3. Run the Python script.

Note: The code includes a `ConvergenceWarning` from scikit-learn, indicating that the logistic regression model did not converge within the maximum number of iterations. This warning can be addressed by increasing the number of iterations or scaling the data as suggested in the warning message.