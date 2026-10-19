import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Spam Message Classifier",
    page_icon="📩"
)

st.title("📩 Spam Message Classifier")
st.write("Check whether a message is Spam or Not Spam using Machine Learning.")

@st.cache_resource
def train_model():
    data = pd.read_csv(
        "SMSSpamCollection",
        sep="\t",
        header=None,
        names=["label", "message"]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        data["message"],
        data["label"],
        test_size=0.2,
        random_state=42,
        stratify=data["label"]
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, predictions)

    return model, vectorizer, accuracy

try:
    model, vectorizer, accuracy = train_model()

    st.success(f"Model ready! Test accuracy: {accuracy:.2%}")

    message = st.text_area(
        "Enter your message below:",
        placeholder="Example: Congratulations! You won a free prize."
    )

    if st.button("Check Message", type="primary"):
        if message.strip():
            message_tfidf = vectorizer.transform([message])
            prediction = model.predict(message_tfidf)[0]

            if prediction == "spam":
                st.error("🚨 This message is likely SPAM!")
            else:
                st.success("✅ This message is likely NOT SPAM.")

        else:
            st.warning("Please enter a message first.")

except FileNotFoundError:
    st.error(
        "Dataset not found. Keep SMSSpamCollection "
        "in the same folder as app.py."
    )