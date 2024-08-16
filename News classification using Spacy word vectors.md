### Notes on Text Classification Using SpaCy Word Embeddings

#### Overview
In this guide, we will perform text classification using SpaCy word embeddings on a dataset containing news articles labeled as "fake" or "real." We will walk through data preprocessing, conversion to numerical labels, and transforming text into word vectors.

---

#### 1. **Loading the Dataset**
We'll start by loading the CSV file into a Pandas DataFrame. This dataset contains news articles that are either classified as fake or real.

**Example Code:**
```python
import pandas as pd
df = pd.read_csv('news_dataset.csv')
print("Shape of DataFrame:", df.shape)
print(df.head())
```

**Expected Output:**
```
Shape of DataFrame: (9900, 2)
```

---

#### 2. **Checking Class Imbalance**
We will check the distribution of the labels to ensure that there is no significant class imbalance.

**Example Code:**
```python
# Value counts of the label column
print(df['label'].value_counts())
```

**Expected Output:**
```
fake    4950
real    4950
Name: label, dtype: int64
```

---

#### 3. **Converting Text Labels to Numeric Labels**
Machine learning models work better with numerical data, so we convert the 'label' column from text to numbers. We use `map` to replace 'fake' with 0 and 'real' with 1.

**Example Code:**
```python
df['label_num'] = df['label'].map({'fake': 0, 'real': 1})
print(df.head())
```

**Expected Output:**
```
                      text  label  label_num
0  Top Trump surrogate brutally stabs in the back   fake          0
1  [Another news text]                               real          1
...
```

---

#### 4. **Loading SpaCy and Converting Text to Word Vectors**
Next, we use SpaCy to convert text into word vectors. We will use the large SpaCy model which includes word vectors.

**Example Code:**
```python
import spacy

# Load the SpaCy large model
nlp = spacy.load('en_core_web_lg')
def get_word_vector(text):
    doc = nlp(text)
    return doc.vector
df['word_vector'] = df['text'].apply(get_word_vector)

# Show the DataFrame with word vectors
print(df.head())
```

**Expected Output:**
```
                      text  label  label_num                                      word_vector
0  Top Trump surrogate brutally stabs in the back   fake          0  [0.12, -0.04, ..., 0.09]  # 300-element vector
1  [Another news text]                               real          1  [0.07, -0.03, ..., -0.11]  # 300-element vector
...
```

---

#### Key Points:
- **Class Imbalance:** It's important to check if the dataset is balanced; if not, techniques such as resampling may be needed.
- **Label Encoding:** Converting text labels to numbers facilitates machine learning model processing.
- **Word Vectors:** SpaCy's large model provides 300-dimensional vectors for each word, which capture semantic meanings.

### Notes on Converting Text to Vectors and Training the Model

#### 1. **Adding a Vector Column to the DataFrame**
To store the word vectors for each text in a new column of the DataFrame, you will convert each text entry into a vector and add it to the DataFrame. This is done using the `apply` function along with a lambda function.

**Example Code:**
```python
import spacy

# Load the SpaCy large model
nlp = spacy.load('en_core_web_lg')
def text_to_vector(text):
    doc = nlp(text)
    return doc.vector

# Apply the function to the text column and create a new column 'vector'
df['vector'] = df['text'].apply(lambda x: text_to_vector(x))

# Show the DataFrame with the vector column
print(df.head())
```

**Expected Output:**
```
                      text  label  label_num                                      vector
0  Top Trump surrogate brutally stabs in the back   fake          0  [0.12, -0.04, ..., 0.09]  # 300-element vector
1  [Another news text]                               real          1  [0.07, -0.03, ..., -0.11]  # 300-element vector
...
```

*Note:* This operation can be time-consuming, especially for large datasets. It may take several minutes to complete.

---

#### 2. **Splitting the Dataset into Training and Testing Sets**
After converting text to vectors, you need to split the dataset into training and testing sets. This can be done using `train_test_split` from `sklearn.model_selection`.

**Example Code:**
```python
from sklearn.model_selection import train_test_split

# Prepare features (X) and labels (Y)
X = list(df['vector'])
Y = df['label_num']

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Show the shapes of the splits
print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
```

**Expected Output:**
```
Training set size: 7920, Test set size: 1980
```

---

#### Key Points:
- **Adding Vector Column:** Converting text to vectors and storing them in a new DataFrame column enables using these vectors for machine learning models.
- **Dataset Splitting:** It's crucial to split your dataset into training and testing sets to evaluate model performance.

This should cover the process of adding vectors to your DataFrame and preparing the data for training.

### Notes on Converting and Preparing Data for Training

#### 1. **Ensuring Correct Array Format**
After splitting the dataset, it's important to ensure that the feature arrays (`X_train` and `X_test`) are in the correct format for machine learning models. Specifically, the arrays should be 2D, where each row represents a sample and each column represents a feature.

**Issue:**
The `X_train` and `X_test` arrays may currently be lists of 1D arrays, which need to be converted into 2D numpy arrays.

**Solution:**
Use `numpy.stack` to convert lists of 1D arrays into a 2D numpy array.

**Example Code:**
```python
import numpy as np

# Convert lists of 1D arrays to 2D numpy arrays
X_train_2d = np.stack(X_train)
X_test_2d = np.stack(X_test)

# Verify the shapes
print("X_train shape:", X_train_2d.shape)
print("X_test shape:", X_test_2d.shape)
```

**Expected Output:**
```
X_train shape: (7920, 300)  # Example shape, with 300 being the dimension of each vector
X_test shape: (1980, 300)   # Example shape
```

---

#### 2. **Using `numpy.stack`**
The `numpy.stack` function is used to combine arrays along a new axis, which effectively transforms a list of 1D arrays into a single 2D array.

**Example Explanation:**
If you have a list of vectors, where each vector is a 1D array, `numpy.stack` will combine these vectors into a 2D array, where each row represents a vector.

**Example Code:**
```python
# Assume X_train and X_test are lists of 1D numpy arrays
X_train_2d = np.stack(X_train)
X_test_2d = np.stack(X_test)
```

---

#### Key Points:
- **Data Formatting:** Ensuring that the feature arrays are in 2D format is crucial for training machine learning models.
- **Use of `numpy.stack`:** This function helps convert lists of 1D arrays into a 2D numpy array, making it suitable for model input.

### Notes on Preparing Data for Training with Scikit-Learn Classifier

#### 1. **Importing and Using `numpy.stack`**
After converting lists of vectors into 2D numpy arrays, ensure that the arrays are correctly formatted for training with Scikit-Learn models.

**Example Code:**
```python
import numpy as np

# Convert lists of 1D arrays to 2D numpy arrays
X_train_2d = np.stack(X_train)
X_test_2d = np.stack(X_test)

# Verify the shapes
print("X_train shape:", X_train_2d.shape)
print("X_test shape:", X_test_2d.shape)
```

**Expected Output:**
```
X_train shape: (7920, 300)  # Example shape
X_test shape: (1980, 300)   # Example shape
```

*Note:* This ensures that `X_train_2d` and `X_test_2d` are now in a format suitable for model training.

---

#### 2. **Importing and Training a Scikit-Learn Classifier**
Next, we'll use Scikit-Learn's `MultinomialNB` classifier, which is commonly used for text classification problems.

**Example Code:**
```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Initialize the classifier
clf = MultinomialNB()

# Train the classifier
clf.fit(X_train_2d, Y_train)

# Make predictions
Y_pred = clf.predict(X_test_2d)

# Evaluate the classifier
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)
```

**Expected Output:**
```
Accuracy: 0.85  # Example accuracy, will vary based on the data
```

---

#### Key Points:
- **Classifier Choice:** `MultinomialNB` is a popular choice for text classification tasks due to its effectiveness with word count features.
- **Data Formatting:** Ensure that feature arrays are 2D numpy arrays for compatibility with Scikit-Learn models.
- **Evaluation:** Use metrics like accuracy to evaluate the performance of your classifier.

### Notes on Handling Negative Values and Evaluating Classifier Performance

#### 1. **Handling Negative Values in MultinomialNB**
The `MultinomialNB` classifier requires non-negative values. If your data contains negative values, scaling is a common solution. The `MinMaxScaler` from Scikit-Learn can transform the values into a positive range.

**Example Code:**
```python
from sklearn.preprocessing import MinMaxScaler

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the training data
X_train_scaled = scaler.fit_transform(X_train_2d)

# Transform the test data
X_test_scaled = scaler.transform(X_test_2d)

# Initialize the classifier
clf = MultinomialNB()

# Train the classifier
clf.fit(X_train_scaled, Y_train)

# Make predictions
Y_pred = clf.predict(X_test_scaled)

# Print classification report
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))
```

**Expected Output:**
```
              precision    recall  f1-score   support

       fake       0.92      0.89      0.91       990
       real       0.88      0.91      0.89       990

    accuracy                           0.90      1980
   macro avg       0.90      0.90      0.90      1980
weighted avg       0.90      0.90      0.90      1980
```

*Note:* The exact numbers will vary based on the data.

---

#### 2. **Evaluating the Model**
Using `classification_report` provides detailed metrics including precision, recall, and F1-score for each class, which helps in understanding the model's performance.

**Example Code:**
```python
from sklearn.metrics import classification_report

# Print classification report
print(classification_report(Y_test, Y_pred))
```

---

#### 3. **Using K-Nearest Neighbors (KNN) Classifier**
KNN is another popular classifier. Here's how you can use it for your classification task.

**Example Code:**
```python
from sklearn.neighbors import KNeighborsClassifier

# Initialize the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)

# Train the classifier
knn.fit(X_train_scaled, Y_train)

# Make predictions
Y_pred_knn = knn.predict(X_test_scaled)

# Print classification report for KNN
print(classification_report(Y_test, Y_pred_knn))
```

**Expected Output:**
```
              precision    recall  f1-score   support

       fake       0.89      0.88      0.88       990
       real       0.88      0.89      0.88       990

    accuracy                           0.88      1980
   macro avg       0.88      0.88      0.88      1980
weighted avg       0.88      0.88      0.88      1980
```

*Note:* The performance of KNN can vary based on the choice of `n_neighbors` and other parameters.

---

#### Key Points:
- **Scaling:** Use `MinMaxScaler` to handle negative values for models like `MultinomialNB`.
- **Evaluation:** Use `classification_report` for comprehensive performance metrics.
- **KNN:** K-Nearest Neighbors is another classifier that can be used and evaluated similarly.

### Summary of Results and Key Takeaways

#### 1. **Performance of Classifiers**
- **K-Nearest Neighbors (KNN):** Achieved exceptional precision and recall, with scores close to 99%. This high performance is due to the lower dimensionality of the data (300-dimensional vectors), which avoids the issues KNN faces with high-dimensional spaces.
- **Multinomial Naive Bayes (NB):** Also performed well, with precision, recall, and F1 scores above 94%. This model is effective for text classification tasks with word embeddings.

#### 2. **Why KNN Performed Well**
- **Dense Representations:** KNN performs better with dense vector representations (e.g., 300-dimensional vectors from SpaCy) compared to high-dimensional sparse representations like TF-IDF or bag-of-words.
- **Dimensionality:** Lower-dimensional embeddings (like those from SpaCy) help KNN avoid issues with high-dimensional spaces, leading to better performance.

#### 3. **Conclusion**
- **Effective Models:** Both KNN and MultinomialNB are effective for text classification when using pre-trained word embeddings. KNN shows particularly impressive results with dense vectors.
- **Exercises and Contributions:** Future exercises related to these concepts will be posted. Contributions and feedback through pull requests are encouraged.

#### 4. **Next Steps**
- **Exercises:** Look out for exercises in the video descriptions to practice and apply the concepts.
- **Community Engagement:** Consider contributing back by sharing feedback or code improvements.
