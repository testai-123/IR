import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("spam.csv")

encoder = LabelEncoder()
df['Category'] = encoder.fit_transform(df['Category'])

x_train, x_test, y_train, y_test = train_test_split(df['Message'],df['Category'],test_size=0.25)

cv = CountVectorizer()
x_train_count = cv.fit_transform(x_train)


model = MultinomialNB()
model.fit(x_train_count, y_train)

print("Prediction for ham email:", model.predict(cv.transform(["Hey how are you?"])))
print("Prediction for spam email:", model.predict(cv.transform(["gift"])))
print("Model Accuracy:", model.score(cv.transform(x_test), y_test))
