feat: Implemented SMS Spam Classifier using TF-IDF and ML Models

- Loaded and cleaned the SMS dataset.
- Applied TF-IDF vectorization for feature extraction.
- Trained and evaluated three models: Naive Bayes, Logistic Regression, and SVM.
- Included evaluation metrics: accuracy, classification report, and confusion matrix.
----------------------------------------------------------------------------------------------------------------------------------------
📧 SMS Spam Classifier using Machine Learning
This project builds an AI model to classify SMS messages as spam or legitimate (ham). The classification is performed using popular techniques like TF-IDF (Term Frequency-Inverse Document Frequency) for text vectorization and machine learning algorithms like Naive Bayes, Logistic Regression, and Support Vector Machines (SVM).

🔧 Project Features:
Data Loading & Preprocessing: Efficient data cleaning and preparation for effective modeling.
Feature Extraction: Applied TF-IDF vectorization to convert text data into numerical features.
Modeling: Implemented and compared three models:
Naive Bayes - Suitable for text data.
Logistic Regression - Effective for binary classification.
Support Vector Machine (SVM) - Robust for text classification.
Evaluation Metrics: Assessed models using:
Accuracy
Precision, Recall, F1-Score
Confusion Matrix
📂 Project Structure:
bash
Copy
Edit
├── spam.csv                  # SMS Dataset
├── Spam_SMS_Classifier.py    # Model training and evaluation code
└── README.md                 # Project documentation
🚀 Getting Started:
Clone the repository:
bash
Copy
Edit
git clone <repository-url>
Install dependencies:
bash
Copy
Edit
pip install pandas scikit-learn
Run the model:
bash
Copy
Edit
python Spam_SMS_Classifier.py
🔍 Results:
Evaluated all three models and compared their performance.
The best model can be selected based on evaluation metrics.
💡 Future Improvements:
Explore advanced techniques like Word Embeddings and Deep Learning.
Integrate a user-friendly interface for practical use.
If this project helps you, please give it a ⭐ and consider contributing to make it better! 😊
