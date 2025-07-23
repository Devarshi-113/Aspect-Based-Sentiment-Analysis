Aspect-Based Sentiment Analysis (ABSA) on Cab Booking App Reviews
This project implements a complete pipeline for Aspect-Based Sentiment Analysis using deep learning models on real-world customer reviews of a cab booking application. It enables fine-grained opinion mining on various aspects mentioned in the reviews and provides sentiment classification (positive, negative, neutral) per aspect.

The project also includes a Streamlit-based frontend to visualize and interact with the results.

🔍 Project Overview
Traditional sentiment analysis provides only a single polarity (positive/negative) per review. However, a review may contain multiple aspects (e.g., driver behavior, app interface, pricing) with different sentiments. This project aims to detect such aspects and determine sentiment polarity for each one, allowing businesses to gain deeper insights from customer feedback.

📊 Dataset
Source: https://www.kaggle.com/datasets/rajatraj0502/ola-vs-uber-reviews
Attributes:

source, review_id, user_name, review_title, review_description, rating, thumbs_up, review_date, developer_response, developer_response_date, appVersion, language_code, country_code

Selected Features Used:
To improve computational efficiency, only essential attributes were retained:

review_description → used as the main document for ABSA

rating → used to derive sentiment labels (1-2: Negative, 3: Neutral, 4-5: Positive)

thumbs_up → used to assign training weights, giving preference to high-quality reviews

🏗️ Project Structure
bash
Copy
Edit
iocl_absa/
├── frontend/                # Streamlit app
│   └── app.py
├── backend/
│   ├── preprocess.py        # Data preprocessing logic
│   ├── sentiment.py         # Sentiment extraction logic
│   └── aspect_extraction.py # Aspect identification
├── models/                  # Trained models saved here
├── data/                    # Input datasets (CSV)
├── absa_training.ipynb      # End-to-end training + testing notebook
├── requirements.txt         # Project dependencies
└── README.md
🛠️ Technologies Covered
Python

Natural Language Processing (NLP)

Tokenization, Lemmatization, Stopword Removal (SpaCy)

Word Embeddings (TF-IDF, GloVe)

Aspect Extraction using TextRank / Topic Modeling

Deep Learning

Bidirectional LSTM for sentiment classification

Attention Mechanisms

Semantic Similarity

Cosine Similarity, Sentence Embeddings

Data Handling: Pandas, NumPy

Frontend: Streamlit

Visualization: Matplotlib, Seaborn, WordCloud

🧠 Training Details
Training Type: Supervised Learning

Labels: Derived from rating column

Weighted Training: Reviews with higher thumbs_up values are given more importance using a custom loss function or sample weighting.

Model: BiLSTM-based sentiment classifier with pretrained embeddings

Epoch-wise Logging: Model performance tracked via accuracy and loss across epochs

🚀 How to Run
Clone the repo

bash
Copy
Edit
git clone https://github.com/your-username/iocl_absa.git
cd iocl_absa
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
python -m spacy download en_core_web_sm
Run training notebook (Optional)

bash
Copy
Edit
jupyter notebook absa_training.ipynb
Launch the frontend

bash
Copy
Edit
streamlit run frontend/app.py
📦 Output
Extracted aspects with classified sentiment per aspect.

Interactive interface to filter and explore reviews.

Color-coded review samples based on sentiment.

Semantic search for aspect-related reviews.

