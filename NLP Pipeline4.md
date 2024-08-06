## NLP Pipeline: Notes with Examples and Explanations: [Source](https://www.youtube.com/watch?v=S3EId9uatxI&list=PLeo1K3hjS3uuvuAXhYjV2lMEShq2UYSwX&index=8)

### 1. Introduction to NLP Pipeline
**Building a real-life NLP application involves several steps:**
1. **Data Acquisition**
2. **Data Cleaning**
3. **Model Building**
4. **Deployment**
5. **Monitoring**

These steps together form what is known as the **NLP pipeline**.

### 2. Use Case: Camtasia Support Ticket System
**Example: Camtasia's Support Ticket System**
- **Camtasia:** A video recording software with a support ticket system.
- **Problem:** Prioritizing tickets manually can be challenging due to high volume.
- **Solution:** An NLP system that analyzes ticket text to determine priority (high, medium, low).

### 3. Step 1: Data Acquisition
- **Data Sources:** Internal databases (e.g., MongoDB) or public datasets.
- **Example:** Camtasia stores support tickets in MongoDB.
- **Challenges:** Sometimes data may not be labeled, requiring human annotators.

**Methods for Data Collection:**
- **Public Datasets:** Google Dataset Search, U.S. Census Bureau.
- **Web Scraping:** Using Python scripts to collect data from the internet.
- **Product Intervention:** Instrumenting products to collect data.
- **Data Augmentation:** Creating more samples from existing ones.

### 4. Step 2: Data Cleaning (Text Extraction & Cleanup)
**Tasks:**
- **Discard Irrelevant Information:** Remove unnecessary fields like creator and timestamp.
- **Example:** Only keep ticket title and description.
- **Spelling Correction:** Fix common typos.
- **Example:** Correct "cras" to "crash".
- **Removing Line Breaks:** Clean up text formatting.

### 5. Step 3: Sentence Segmentation
**Definition:** Splitting text into sentences.
**Challenges:** Simple rules (like splitting by periods) are often insufficient due to nuances in language (e.g., "Dr. Strange").

### 6. Step 4: Word Tokenization
**Definition:** Splitting sentences into words.

### 7. Step 5: Stemming and Lemmatization
**Stemming:**
- **Definition:** Reducing words to their root form using simple rules.
- **Example:** "Eating" becomes "eat" by removing "-ing".

**Lemmatization:**
- **Definition:** Mapping words to their base form using grammar rules.
- **Example:** "Ate" becomes "eat".

### 8. Step 6: Feature Engineering
**Definition:** Converting words into numerical features for machine learning models.
**Techniques:**
- **TF-IDF Vectorizer**
- **One Hot Encoding**
- **Word Embedding**

### 9. Step 7: Model Building
**Goal:** Create a classifier to determine ticket priority.
**Techniques:**
- **Classification Algorithms:** Decision Tree, SVM, Random Forest, Naive Bayes.
- **Example:** Using Naive Bayes to classify ticket priority.

### 10. Model Evaluation
**Techniques:**
- **Grid Search CV:** To find the best model parameters.
- **Confusion Matrix:** To evaluate model performance by comparing predictions to actual values.

### 11. Deployment and Monitoring
**Deployment:**
- **Options:** Cloud platforms like AWS, Azure, Google Cloud.
- **Tools:** FastAPI, Flask for building REST services.

**Monitoring:**
- **Importance:** Ensure the model performs well in production.
- **Adjustments:** Periodically update the model to adapt to new data.

### Summary
- **NLP Pipeline Steps:** Data acquisition, data cleaning, sentence segmentation, word tokenization, stemming/lemmatization, feature engineering, model building, deployment, monitoring.
- **Real-life application:** Example of Camtasia support ticket system to demonstrate the pipeline.

### Additional Resources
- **Book:** Practical NLP (recommended for detailed examples).
- **Videos:** Tutorials on machine learning, feature engineering, and model evaluation (available on YouTube, e.g., Codebasics channel).


**To illustrate the concepts discussed in the video, I'll provide a coding example and diagrams for each step of the NLP pipeline. We'll use Python, the Natural Language Toolkit (NLTK), and Scikit-learn to build a support ticket classification model.**

### Step 1: Data Acquisition
We'll use a sample dataset of support tickets.

```python
import pandas as pd
data = {
    "title": ["App crashes on startup", "Unable to login", "Slow performance when editing"],
    "description": ["The application crashes every time I try to open it.", 
                    "I cannot login to my account using the correct credentials.", 
                    "Editing videos is extremely slow, almost unusable."],
    "severity": ["high", "medium", "low"]
}

df = pd.DataFrame(data)
df.to_csv('support_tickets.csv', index=False)
print(df)
```

### Step 2: Text Extraction & Cleanup
We preprocess the text data by cleaning it.

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])
    lemmatizer = WordNetLemmatizer()
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

df['cleaned_text'] = (df['title'] + " " + df['description']).apply(preprocess_text)
print(df[['cleaned_text', 'severity']])
```

### Step 3: Feature Engineering
We convert text data into numerical features.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['severity']

print(X.toarray())
```

### Step 4: Model Building
We train a Naive Bayes classifier on the dataset.

```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
```

### Diagrams
1. **NLP Pipeline Overview:**
   ![NLP Pipeline](https://i.imgur.com/EkVdQyM.png)
   
2. **Data Acquisition and Preprocessing:**
   ![Data Acquisition and Preprocessing](https://i.imgur.com/jyO4cJK.png)
   
3. **Feature Engineering:**
   ![Feature Engineering](https://i.imgur.com/5yEcbOX.png)
   
4. **Model Training and Evaluation:**
   ![Model Training and Evaluation](https://i.imgur.com/Z5GvM5y.png)
