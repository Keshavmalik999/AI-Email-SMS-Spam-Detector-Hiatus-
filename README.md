# 📩 AI Email & SMS Spam Detector

A web application that classifies incoming emails or SMS messages as **Spam** or **Ham (Legitimate)** using Machine Learning.

## 🚀 Overview
This project uses a **Multinomial Naïve Bayes** classifier trained on message text to identify spam patterns. The user interface is built with **Streamlit**, providing a fast, clean, and interactive way to test messages in real-time.

## ✨ Features
* **Real-time Analysis:** Paste any text message or email and get an instant prediction.
* **Confidence Scoring:** Shows the probability percentage of how confident the model is in its prediction.
* **Cached Model Training:** Uses Streamlit's resource caching to ensure the machine learning model trains instantly on startup and stays ready in memory.
* **Responsive UI:** Clean, centered web interface optimized with visual alerts (red for spam, green for legitimate messages).

## 🛠️ Tech Stack
* **Frontend/Framework:** Streamlit
* **Machine Learning Library:** Scikit-Learn
* **Data Handling:** Pandas
* **Algorithm:** Multinomial Naïve Bayes (`MultinomialNB`)
* **Text Processing:** Count Vectorizer (Bag of Words)
