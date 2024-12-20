!pip install pgmpy

import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

# Load the dataset
data = pd.read_csv("heart disease.csv")
heart_disease = pd.DataFrame(data)
print(heart_disease)

# Define the Bayesian Model structure
model = BayesianNetwork([
    ('age', 'Lifestyle'), ('Gender', 'Lifestyle'),
    ('Family', 'heartdisease'),
    ('diet', 'cholestrol'),
    ('Lifestyle', 'diet'),
    ('cholestrol', 'heartdisease')
])

# Fit the model using Maximum Likelihood Estimator
model.fit(heart_disease, estimator=MaximumLikelihoodEstimator)

# Perform inference
heart_disease_infer = VariableElimination(model)

# Mapping descriptions
descriptions = {
    'Age': 'SuperSeniorCitizen:0, SeniorCitizen:1, MiddleAged:2, Youth:3, Teen:4',
    'Gender': 'Male:0, Female:1',
    'Family History': 'Yes:1, No:0',
    'Diet': 'High:0, Medium:1',
    'Lifestyle': 'Athlete:0, Active:1, Moderate:2, Sedentary:3',
    'Cholesterol': 'High:0, BorderLine:1, Normal:2'
}

# Display input descriptions
for category, mapping in descriptions.items():
    print(f'For {category} enter {mapping}')

# Get user inputs for evidence
evidence = {
    'age': int(input('Enter Age: ')),
    'Gender': int(input('Enter Gender: ')),
    'Family': int(input('Enter Family History: ')),
    'diet': int(input('Enter Diet: ')),
    'Lifestyle': int(input('Enter Lifestyle: ')),
    'cholestrol': int(input('Enter Cholesterol: '))
}

# Query the model
result = heart_disease_infer.query(variables=['heartdisease'], evidence=evidence)

# Display the result
print(result)
