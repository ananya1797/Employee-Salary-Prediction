# 💼 Employee Salary Prediction

This project predicts whether an employee earns more than $50K per year using machine learning techniques. It is built with Python, scikit-learn, and Streamlit to support both individual and batch predictions through a clean, interactive web interface.

---

## 🚀 Features

- Predict employee salary class (`>50K` or `<=50K`)
- Web-based UI using **Streamlit**
- Supports single input prediction and **batch CSV upload**
- Trained using models like **Logistic Regression** and **MLPClassifier**
- Includes preprocessing pipeline with **Label Encoding** and **MinMax Scaling**

---

## 📁 Project Structure

app.py # Streamlit app for prediction
├── income_prediction_model.pkl # Serialized trained model (joblib)
├── dataset.csv # (Optional) Sample dataset
├── README.md # Project documentation
└── requirements.txt # All required Python libraries
## 📊 Dataset

- Based on the Adult Income dataset (UCI Machine Learning Repository)
- Features include: age, education, occupation, hours-per-week, experience, etc.
- Target: Income (binary classification — `>50K` or `<=50K`)

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/employee-salary-prediction.git
   cd employee-salary-prediction
   Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the Streamlit app

bash
Copy
Edit
streamlit run app.py
🧠 Model Training (optional)
You can retrain the model using the provided code:

python
Copy
Edit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
pipe.fit(X, y)
joblib.dump(pipe, 'income_prediction_model.pkl')
✅ Requirements
Python 3.7+

pandas

scikit-learn

streamlit

joblib

List of dependencies is provided in requirements.txt.

📈 Accuracy & Results
Logistic Regression: ~84% accuracy

MLP Classifier: ~86% accuracy

Models were evaluated using accuracy_score on the test dataset.

❗ Challenges Faced
Handling missing and inconsistent categorical data

Tuning neural network layers to avoid underfitting

Ensuring model generalizes across real-world employee data

🔮 Future Enhancements
Integrate more advanced models like XGBoost or LightGBM

Improve UI with real-time visual feedback

Deploy to cloud platform (e.g., Heroku, AWS)

Add model explainability (e.g., SHAP or LIME)
