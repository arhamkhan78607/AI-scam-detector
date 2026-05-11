import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Sample dataset
data = {
    'label': ['spam', 'ham', 'spam', 'ham'],
    'message': [
        'Win money now',
        'Hello how are you',
        'Claim your free prize',
        'Lets meet tomorrow'
    ]
}

df = pd.DataFrame(data)

# Convert text into numbers
cv = CountVectorizer()
X = cv.fit_transform(df['message'])

y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test message
msg = ["Congratulations! You won free money"]

msg_vector = cv.transform(msg)

result = model.predict(msg_vector)

print("Prediction:", result[0])