# Spam Mail Detection

## Project Overview
This project is a machine learning-based spam mail detection system. It uses natural language processing (NLP) techniques and a Naive Bayes classifier to distinguish between spam and non-spam (ham) emails. The project provides both a command-line interface and a simple web app built with Streamlit for easy interaction.

## Features
- Preprocessing and cleaning of email text
- Feature extraction using TF-IDF vectorization
- Spam detection using a Multinomial Naive Bayes model
- Streamlit web app for user-friendly predictions
- Example dataset and scripts for training and testing

## Installation
1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd spam_mail_detection
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### 1. Streamlit Web App
To launch the web app:
```bash
streamlit run app/app.py
```
Enter the email content in the text area and click "Predict" to see if it is spam or not.

### 2. Command-Line Prediction
You can use the prediction function in your own scripts. Example:
```python
from src.predict import predict_email

email = "Congratulations! You've won a free ticket. Click here!"
result = predict_email(email)
print("Result:", result)  # Output: Spam or Ham
```

### 3. Testing
Run the provided test scripts in the `tests/` directory to verify preprocessing, feature extraction, and prediction:
```bash
python tests/test_preprocessing.py
python tests/test_feature_extraction.py
python tests/test_predict.py
```

## Model Training
To retrain the model with your own data:
1. Place your dataset in `data/spam.csv` with columns `label` ("ham" or "spam") and `text`.
2. Run the training script:
   ```bash
   python train.py
   ```
This will preprocess the data, extract features, train the model, and save the trained model and vectorizer to the `models/` directory.

## Project Structure
```
spam_mail_detection/
├── app/
│   └── app.py                # Streamlit web app
├── data/
│   └── spam.csv              # Example dataset
├── models/
│   ├── spam_model.pkl        # Trained model
│   └── vectorizer.pkl        # Trained vectorizer
├── src/
│   ├── feature_extraction.py # TF-IDF vectorization
│   ├── predict.py            # Prediction logic
│   ├── preprocessing.py      # Text cleaning
│   └── __init__.py
├── tests/
│   ├── test_feature_extraction.py
│   ├── test_predict.py
│   └── test_preprocessing.py
├── train.py                  # Model training script
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```


