import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# Page Config
st.set_page_config(page_title="AI Spam Detector", page_icon="📩", layout="centered")

st.title("📩 AI Email & SMS Spam Detector")
st.write("Built using **Multinomial Naïve Bayes**")

# 1. Built-in Dataset (No external files needed)
@st.cache_resource
def train_spam_model():
    data = {
        'text': [
            "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.",
            "FreeMsg Hey there darling it's been 3 week's now and no word back!",
            "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward!",
            "Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles for Free!",
            "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot!",
            "Congratulations! You won a $1000 Walmart gift card. Click here to claim now.",
            "Hey, are we still meeting for lunch today at 1 PM?",
            "Can you please send me the BCA assignment PDF by evening?",
            "Nah I don't think he goes to usf, he lives around here though.",
            "Ok lor... Joining you guys for dinner later.",
            "Will call you back in 10 minutes, driving right now.",
            "Don't forget to submit your lab record before Friday."
        ],
        'label': [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  # 1 = Spam, 0 = Ham (Legitimate)
    }
    
    df = pd.DataFrame(data)
    
    # Text Processing & Feature Extraction
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['text'])
    y = df['label']
    
    # Train Naïve Bayes Model
    model = MultinomialNB()
    model.fit(X, y)
    
    return model, vectorizer

model, vectorizer = train_spam_model()

st.markdown("---")

# 2. User Input
user_input = st.text_area("Enter Email / Message Content Below:", height=120, 
                          placeholder="e.g., Congratulations! You have won a free gift card. Click here...")

if st.button("Analyze Message"):
    if user_input.strip() == "":
        st.warning("Please type or paste a message to analyze.")
    else:
        # Preprocess & Predict
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        probabilities = model.predict_proba(input_vector)[0]
        
        confidence = probabilities[prediction] * 100
        
        st.markdown("### 🔍 Detection Result")
        
        if prediction == 1:
            st.error(f"🚨 **SPAM DETECTED!** (Confidence: {confidence:.1f}%)")
            st.write("⚠️ **Reasoning:** This message contains patterns, keywords, or phishing structures commonly found in unsolicited spam.")
        else:
            st.success(f"✅ **LEGITIMATE MESSAGE (HAM)** (Confidence: {confidence:.1f}%)")
            st.write("👍 **Reasoning:** Natural language pattern detected without common spam keywords.")

st.caption("⚡ Model: Naïve Bayes Classifier | Framework: Scikit-Learn & Streamlit")