import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the dataset
spam_df = pd.read_csv("spam.csv")

# Encode 'spam' as 1 and 'ham' as 0
spam_df['spam'] = spam_df['Category'].map({'spam': 1, 'ham': 0})

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(spam_df['Message'], spam_df['spam'], test_size=0.25)

# Vectorize the text data
cv = CountVectorizer()
x_train_count = cv.fit_transform(x_train)

# Train the model
model = MultinomialNB()
model.fit(x_train_count, y_train)

# Test the model with sample emails
print("Prediction for ham email:", model.predict(cv.transform(["Hey how are you?"])))
print("Prediction for spam email:", model.predict(cv.transform(["gift"])))

# Evaluate the model's performance
print("Model Accuracy:", model.score(cv.transform(x_test), y_test))
