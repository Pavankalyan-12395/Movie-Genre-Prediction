# 🎬 Movie Genre Prediction using Machine Learning

- This project predicts the **genre of a movie** based on its **plot/overview text** using **Natural Language Processing (NLP)** and **Machine Learning**.  
A simple and interactive **Streamlit web app** is used for prediction.

---

## 🚀 Features
- Text preprocessing using NLP techniques
- TF-IDF vectorization
- Logistic Regression classifier
- Interactive Streamlit web application
- Real-time genre prediction

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- NLTK
- Streamlit

---

## 📂 Project Structure
Movie Genre Prediction/

├── Movie Dataset.csv

├── train_movie_genre.py

├── app_movie_genre.py

├── README.md

---

## ⚙️ How to Run the Project

1️⃣ Install required libraries

     pip install streamlit pandas numpy scikit-learn nltk joblib
    
2️⃣ Train the model

     python train_movie_genre.py
    
3️⃣ Run the Streamlit app

     streamlit run app_movie_genre.py
    
🧪 Example Input

    A skilled thief is given a chance at redemption if he can perform an impossible task by planting an idea into someone's subconscious.
✅ Output

Predicted Genre: Drama

📌 Note

Model file (.pkl) is uploaded to GitHub but go through with the data that you are working on.

Predictions may vary due to dataset imbalance.
