# Spam Mail Detection

## Project Overview
This project implements a spam mail detection system using two different approaches:
1. A Naive Bayes classifier with traditional NLP preprocessing
2. A modern vector database approach using transformer embeddings and KNN classification

The project provides both approaches for comparison and research purposes, along with a Streamlit web application for easy interaction.

## Features
- Two different spam detection approaches:
  - Traditional: Naive Bayes with text preprocessing and bag-of-words features
  - Modern: Transformer embeddings (MiniLM) with KNN classification using FAISS
- Text preprocessing pipeline with NLTK
- Streamlit web app for user-friendly predictions
- Example dataset and scripts for training both models
- Pickle files for model persistence

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

### 2. Training the Models

#### Naive Bayes Classifier
To train the Naive Bayes model:
```bash
python nb_classifier/train.py
```
This will:
- Load and preprocess the data from `data/spam.csv`
- Create a bag-of-words dictionary
- Train a Gaussian Naive Bayes model
- Save the model, dictionary, and label encoder to the `pickle/` directory

#### Vector Database Classifier
To train the vector database model:
```bash
python vector_db/pipeline.py
```
This will:
- Load data from `data/spam.csv`
- Generate embeddings using the MiniLM transformer model
- Create a FAISS index for fast similarity search
- Save the index and labels for later use

### 3. Making Predictions
You can use either model for predictions:

```python
# Using Naive Bayes
from nb_classifier.predict import predict
emails = ["Free prize waiting!", "Meeting at 3pm"]
results = predict(emails)

# Using Vector DB
from vector_db.pipeline import predict_pipeline
results = predict_pipeline(emails, k=3)  # k is the number of neighbors
```

## Project Structure
```
spam_mail_detection/
├── app/
│   └── app.py                # Streamlit web application
├── data/
│   └── spam.csv             # Dataset for training
├── nb_classifier/           # Naive Bayes implementation
│   ├── predict.py           # Prediction logic
│   ├── preprocess.py        # Text cleaning and tokenization
│   ├── train.py            # Model training
│   └── utils.py            # Helper functions
├── pickle/                 # Saved Naive Bayes model files
│   ├── dictionary.pkl      # Vocabulary dictionary
│   ├── label_encoder.pkl   # Label encoder
│   └── model.pkl          # Trained Naive Bayes model
├── vector_db/             # Vector database implementation
│   ├── embedding_model.py  # Transformer embedding generation
│   ├── knn_classifier.py   # KNN classification logic
│   ├── pipeline.py        # Training and prediction pipeline
│   └── vector_index.py    # FAISS index operations
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## Dependencies
- scikit-learn: Machine learning algorithms
- pandas: Data manipulation
- numpy: Numerical operations
- nltk: Natural language processing
- transformers: Transformer models for embeddings
- faiss-cpu: Vector similarity search
- streamlit: Web application framework

## Dataset
The project expects a CSV file (`data/spam.csv`) with the following columns:
- `Message`: The email text content
- `Label`: The classification label ("spam" or "ham")

## Contributing
Feel free to open issues or submit pull requests with improvements.

## Source
- Dataset: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
- Document: https://drive.google.com/file/d/1wYuI75A2TwQaAx6gwfE_J-4ia6PbbwJh/view?fbclid=IwY2xjawLquiNleHRuA2FlbQIxMABicmlkETFhVktVcnBIZUFVeEZ3UnFUAR7izrLJU4DLJzqAFjgsS9XJJDJ_2DzIFcL1F4JFV1KIKL04F9SjKKzhRTVZ-w_aem_B3xj3xt44u2XEro1wqHVtQ

## License
[Add your chosen license here]


