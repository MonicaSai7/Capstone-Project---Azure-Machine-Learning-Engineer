# Machine Learning Engineer with Microsoft Azure - Capstone Project

This project is part of the Machine Learning Engineer with Microsoft Azure Nanodegree Program by Udacity and Microsoft.

## Dataset
The dataset used in this project is the Wisconsin Breast Cancer dataset from Kaggle. Two different models are developed, one model trained using Automated ML (AutoML) and the other model trained and tuned with HyperDrive. The performance of both the models are compared and the best model is deployed. The model is then consumed from the generated REST endpoint.

### Overview
The dataset is obtained is the Wisconsin Breast Cancer dataset from Kaggle. The features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. The class distribution is 357 benign, 212 malignant.

Attribute Information:

1) ID number<br>
2) Diagnosis (M = malignant, B = benign)<br>
3-32)

Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from center to points on the perimeter)<br>
b) texture (standard deviation of gray-scale values)<br>
c) perimeter<br>
d) area<br>
e) smoothness (local variation in radius lengths)<br>
f) compactness (perimeter^2 / area - 1.0)<br>
g) concavity (severity of concave portions of the contour)<br>
h) concave points (number of concave portions of the contour)<br>
i) symmetry<br>
j) fractal dimension ("coastline approximation" - 1)

### Task
The dataset provides us with a binary classification task that requires us to classify the the given details of the FNA image into two classes: Malignant and Benign. All the attributes excluding the *ID Number* are used for training the model. The column *diagnosis* is the target variable.

### Access
The dataset is uploaded into this github repository which then is associated with raw github content URL. This URL is used to access the dataset from the workspace.<br>
https://raw.githubusercontent.com/MonicaSai7/Capstone-Project---Azure-Machine-Learning-Engineer/main/BreastCancerWisconsinDataset.csv

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
