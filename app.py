import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# App title
st.title("📧 Spam Email Detection App")

st.write("Enter a message and check whether it is Spam or Not Spam")

# Input box
msg = st.text_area("Enter your message here")

# Button
if st.button("Predict"):

    if msg.strip() == "":
        st.warning("⚠️ Please enter a message")
    else:
        # Convert text to numbers
        data = vectorizer.transform([msg])

        # Prediction
        prediction = model.predict(data)

        # Probability
        proba = model.predict_proba(data)

        # Show probability
        st.write(f"📊 Spam Probability: {proba[0][1]:.2f}")

        # Result
        if prediction[0] == 1:
            st.error("🚨 This is SPAM")
        else:
            st.success("✅ This is NOT SPAM")