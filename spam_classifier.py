import pandas as pd


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset
data = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Step 2: Separate input and output
X = data["message"]
y = data["label"]

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 4: Convert text into numerical features
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 5: Train the machine learning model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test_tfidf)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("Model Training Completed!")
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))








# Step 2: Test custom messages

print("\n--- Spam Message Checker ---")

while True:
    message = input("\nEnter a message (or type 'exit' to stop): ")

    if message.lower() == "exit":
        print("Thank you for using Spam Message Checker!")
        break

    # Convert the message into numerical features
    message_tfidf = vectorizer.transform([message])

    # Predict whether the message is spam or ham
    prediction = model.predict(message_tfidf)[0]

    if prediction == "spam":
        print("Result: SPAM MESSAGE!")
    else:
        print("Result: NOT SPAM (HAM)")
