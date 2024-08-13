### **Notes on Bag of Words Model and N-Grams**

#### **1. Bag of Words (BoW) Model**
- **Concept**: The Bag of Words model is a fundamental technique in NLP where text is represented as a collection (or "bag") of individual words or tokens without considering their order. The model creates a vocabulary of unique words from a corpus and counts the frequency of each word in the documents.
- **Limitations**: 
  - **Lack of Context**: BoW doesn't capture the order of words, which is crucial for understanding the meaning of sentences.
  - **Example**: Changing the word order in a sentence (e.g., "the cat sat on the mat" vs. "the mat sat on the cat") alters its meaning, but BoW treats both as similar.

#### **2. N-Grams**
- **Concept**: N-Grams are a sequence of 'n' words together. The model captures sequences of words to preserve some contextual information.
  - **Bigram**: N=2 (pairs of words)
    - Example: For the sentence "the cat sat on the mat," Bigrams would be: ("the cat", "cat sat", "sat on", "on the mat").
  - **Trigram**: N=3 (triplets of words)
    - Example: For the sentence "the cat sat on the mat," Trigrams would be: ("the cat sat", "cat sat on", "sat on the mat").
  - **N-Gram**: General term where N can be any positive integer (e.g., 4-gram, 5-gram).
  
- **Usage**:
  - **Capturing Context**: N-Grams help in capturing the order of words to some extent, providing context that is missing in BoW.
  - **Special Cases**: BoW can be seen as a special case of N-Grams where N=1.

#### **3. Text Preprocessing for NLP**
- **Steps**:
  - **Stop Words Removal**: Removing common words that do not add much meaning (e.g., "the", "and").
  - **Lemmatization**: Reducing words to their base or root form (e.g., "running" -> "run").
  - **Count Vectorization**: After preprocessing, we can build a BoW or N-Gram model by counting the occurrences of words or word pairs in documents.

#### **4. Document Representation**
- **Term Frequency**: Counting the number of times each word (or N-Gram) appears in a document.
- **Vocabulary**: A union of all unique words from the corpus.

### **Python Code Example: BoW and N-Grams**

```python
from sklearn.feature_extraction.text import CountVectorizer

# Sample documents
documents = [
    "The cat sat on the mat.",
    "The dog ate the cat.",
    "The cat ate the pizza."
]

# Bag of Words (BoW) Model
vectorizer_bow = CountVectorizer()
X_bow = vectorizer_bow.fit_transform(documents)

print("Vocabulary (BoW):", vectorizer_bow.get_feature_names_out())
print("BoW Matrix:\n", X_bow.toarray())

# Bigram Model
vectorizer_bigram = CountVectorizer(ngram_range=(2, 2))
X_bigram = vectorizer_bigram.fit_transform(documents)

print("\nVocabulary (Bigram):", vectorizer_bigram.get_feature_names_out())
print("Bigram Matrix:\n", X_bigram.toarray())

# Trigram Model
vectorizer_trigram = CountVectorizer(ngram_range=(3, 3))
X_trigram = vectorizer_trigram.fit_transform(documents)

print("\nVocabulary (Trigram):", vectorizer_trigram.get_feature_names_out())
print("Trigram Matrix:\n", X_trigram.toarray())
```

### **Expected Output**
```plaintext
Vocabulary (BoW): ['ate', 'cat', 'dog', 'mat', 'on', 'pizza', 'sat', 'the']
BoW Matrix:
 [[0 1 0 1 1 0 1 2]
  [1 1 1 0 0 0 0 2]
  [1 1 0 0 0 1 0 2]]

Vocabulary (Bigram): ['ate the', 'cat ate', 'cat sat', 'dog ate', 'on the', 'sat on', 'the cat', 'the dog', 'the pizza']
Bigram Matrix:
 [[0 0 1 0 1 1 1 0 0]
  [1 0 0 1 0 0 0 1 0]
  [1 1 0 0 0 0 1 0 1]]

Vocabulary (Trigram): ['cat ate the', 'cat sat on', 'dog ate the', 'sat on the', 'the cat ate', 'the cat sat', 'the dog ate']
Trigram Matrix:
 [[0 1 0 1 0 1 0]
  [1 0 1 0 0 0 1]
  [1 0 0 0 1 0 0]]
```


### **Notes on N-Grams and Document Representation**

#### **1. N-Grams in Document Representation**
- **Bigrams in Practice**:
  - **Concept**: When using Bigrams, each unit in the vector represents a pair of consecutive words.
  - **Example**: For a document with sentences like "Thor ate pizza" and "Loki ate pizza," the Bigrams would include pairs like ("Thor ate", "ate pizza", "Loki ate", etc.).
  - **Vocabulary Construction**: The Bigram vocabulary includes all unique pairs of words found in the corpus.

- **Document Vectorization**:
  - **Counting Bigrams**: After constructing the Bigram vocabulary, each document is represented by counting the occurrence of these Bigrams.
  - **Example**:
    - **Doc 1**: "Thor ate pizza"
      - Bigrams: ("Thor ate", "ate pizza")
      - Vector: [1, 1, 0, 0] (where each element corresponds to a Bigram in the vocabulary)
    - **Doc 2**: "Loki ate pizza"
      - Bigrams: ("Loki ate", "ate pizza")
      - Vector: [0, 1, 1, 1] (similarly mapped to the Bigram vocabulary)

- **Combining Unigrams and Bigrams**:
  - **Concept**: A combined approach where both individual words (Unigrams) and word pairs (Bigrams) are used in the vectorization process.
  - **Advantages**: 
    - **More Contextual Representation**: The combined vector includes information about individual word frequencies as well as word pair frequencies, offering a more comprehensive representation of the document.
    - **Example**: For "Thor ate pizza," the combined vector might include Unigrams like "Thor", "ate", "pizza" and Bigrams like "Thor ate", "ate pizza".
  
#### **2. Practical Application**
- **Similarity Detection**:
  - **Example**: When comparing documents, such as "Doc 1" ("Thor ate pizza") and "Doc 3" ("Loki ate pizza"), the combined approach will highlight similarities (e.g., "ate pizza") and differences, thus providing better insights for tasks like classification or clustering.
  
- **Training Machine Learning Models**:
  - **Process**:
    - **Vector Creation**: Each document is converted into a vector using the combined Unigram and Bigram representation.
    - **Model Training**: These vectors are then used as input features for machine learning algorithms, helping the model to learn patterns and make predictions based on the text data.

### **Python Code Example: Unigrams, Bigrams, and Combined Approach**

```python
from sklearn.feature_extraction.text import CountVectorizer

# Sample documents
documents = [
    "Thor ate pizza",
    "Loki ate pizza",
    "Hulk smashed Loki"
]

# Unigrams Model
vectorizer_unigram = CountVectorizer(ngram_range=(1, 1))
X_unigram = vectorizer_unigram.fit_transform(documents)

print("Vocabulary (Unigram):", vectorizer_unigram.get_feature_names_out())
print("Unigram Matrix:\n", X_unigram.toarray())

# Bigrams Model
vectorizer_bigram = CountVectorizer(ngram_range=(2, 2))
X_bigram = vectorizer_bigram.fit_transform(documents)

print("\nVocabulary (Bigram):", vectorizer_bigram.get_feature_names_out())
print("Bigram Matrix:\n", X_bigram.toarray())

# Combined Unigrams and Bigrams Model
vectorizer_combined = CountVectorizer(ngram_range=(1, 2))
X_combined = vectorizer_combined.fit_transform(documents)

print("\nVocabulary (Combined Unigram + Bigram):", vectorizer_combined.get_feature_names_out())
print("Combined Matrix:\n", X_combined.toarray())
```

### **Expected Output**
```plaintext
Vocabulary (Unigram): ['ate', 'hulk', 'loki', 'pizza', 'smashed', 'thor']
Unigram Matrix:
 [[1 0 0 1 0 1]
  [1 0 1 1 0 0]
  [0 1 1 0 1 0]]

Vocabulary (Bigram): ['ate pizza', 'hulk smashed', 'loki ate', 'smashed loki', 'thor ate']
Bigram Matrix:
 [[1 0 0 0 1]
  [1 0 1 0 0]
  [0 1 0 1 0]]

Vocabulary (Combined Unigram + Bigram): ['ate', 'ate pizza', 'hulk', 'hulk smashed', 'loki', 'loki ate', 'pizza', 'smashed', 'smashed loki', 'thor', 'thor ate']
Combined Matrix:
 [[1 1 0 0 0 0 1 0 0 1 1]
  [1 1 0 0 1 1 1 0 0 0 0]
  [0 0 1 1 1 0 0 1 1 0 0]]
```

### **Summary**
- **Unigrams**: Basic individual word frequency.
- **Bigrams**: Captures word pairs, providing more context.
- **Combined Approach**: Unigrams + Bigrams offer a richer representation, helping in tasks like document similarity and classification.
- **Application**: These representations can be used to train machine learning models that better understand the text data.

### **Notes on Document Similarity and N-Gram Limitations**

#### **1. Document Similarity Using N-Grams**
- **Concept**: When comparing documents, the similarity between them can be assessed by analyzing how many N-Grams (unigrams, bigrams, etc.) they share. This provides insight into the content overlap between the documents.
  
- **Example Analysis**:
  - **Doc 1 ("Thor ate pizza") vs. Doc 3 ("Loki ate pizza")**:
    - **Unigrams**: Both documents share the words "ate" and "pizza".
    - **Bigrams**: Both share the Bigram "ate pizza".
    - **Similarity**: There are multiple overlaps, making these documents similar.
  - **Doc 2 ("Loki ate pizza") vs. Doc 3 ("Loki ate pizza")**:
    - **Unigrams and Bigrams**: They have more overlaps due to identical content.

#### **2. Limitations of N-Gram Models**
- **Increased Dimensionality and Sparsity**:
  - **Concept**: As the value of 'n' in N-Grams increases, the number of possible N-Grams grows exponentially. This leads to high-dimensional and sparse feature vectors, especially when dealing with large corpora like hundreds of news articles or books.
  - **Challenges**:
    - **Computational Complexity**: Higher 'n' values increase the memory and processing requirements, making the model computationally expensive.
    - **Sparsity**: With many unique N-Grams, most will appear rarely, leading to sparse matrices that are harder to process effectively.

- **Out-of-Vocabulary (OOV) Problem**:
  - **Concept**: When a model is trained on a specific dataset, it learns a vocabulary based on that data. However, during prediction, new (previously unseen) words or N-Grams may appear, which the model cannot represent accurately.
  - **Example**: If the model trained on "Thor ate pizza" but encounters "Spider-Man ate sushi," the new terms "Spider-Man" and "sushi" may not be in the model's vocabulary, leading to difficulties in representing these terms in the feature space.

#### **3. Implementing N-Grams with `CountVectorizer` in Python**

- **`n_gram_range` Parameter**:
  - **Default Behavior**: When using `CountVectorizer`, the default `n_gram_range` is `(1, 1)`, meaning the model only considers unigrams.
  - **Adjusting for Bigrams or Trigrams**: By changing `n_gram_range` to `(2, 2)` or `(1, 3)`, you can generate bigrams, trigrams, or a combination of unigrams, bigrams, and trigrams.

### **Python Code Example: Document Similarity and N-Gram Vectorization**

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample documents
documents = [
    "Thor ate pizza",
    "Loki ate pizza",
    "Spider-Man ate sushi"
]

# Preprocess function (dummy, no real preprocessing for simplicity)
def preprocess(text):
    return text.lower()

# Apply preprocessing
documents = [preprocess(doc) for doc in documents]

# Combined Unigrams and Bigrams Model
vectorizer = CountVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(documents)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("Vectorized Documents:\n", X.toarray())

# Calculate similarity between documents
similarity_matrix = cosine_similarity(X)

print("\nDocument Similarity Matrix:\n", similarity_matrix)

# Preprocessing and N-Gram Vectorization for New Corpus
new_documents = [
    "The party was eating pizza and enjoying.",
    "They all loved the pizza and the party."
]

# Example of preprocessing (stop words removal, lemmatization can be added)
def preprocess(text):
    return text.lower()

new_documents = [preprocess(doc) for doc in new_documents]

# Create Bigram Vectorizer and fit_transform the new corpus
bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))
X_bigrams = bigram_vectorizer.fit_transform(new_documents)

print("\nVocabulary (Bigrams):", bigram_vectorizer.get_feature_names_out())
print("Bigram Vectorized Documents:\n", X_bigrams.toarray())
```

### **Expected Output**
```plaintext
Vocabulary: ['ate', 'ate pizza', 'loki', 'pizza', 'spider-man', 'spider-man ate', 'sushi', 'thor', 'thor ate']
Vectorized Documents:
 [[1 1 0 1 0 0 0 1 1]
  [1 1 1 1 0 0 0 0 0]
  [0 0 0 0 1 1 1 0 0]]

Document Similarity Matrix:
 [[1.         0.58       0.        ]
  [0.58       1.         0.        ]
  [0.         0.         1.        ]]

Vocabulary (Bigrams): ['and enjoying', 'eating pizza', 'loved the', 'pizza and', 'the party', 'was eating']
Bigram Vectorized Documents:
 [[1 1 0 1 1 1]
  [0 0 1 1 1 0]]
```

### **Summary**
- **Document Similarity**: By combining unigrams and bigrams, you can effectively assess the similarity between documents.
- **N-Gram Limitations**: While N-Grams provide context, they also increase computational costs and introduce sparsity, especially as 'n' increases.
- **Practical Use**: Adjusting `n_gram_range` allows for different granularity in text analysis, making it possible to balance between capturing context and managing model complexity.

  ### **Notes on Vector Space Models and Text Preprocessing with SpaCy**

#### **1. Vector Space Model (VSM) Overview**
- **Definition**: A Vector Space Model (VSM) is a mathematical model used to represent text documents as vectors of identifiers, typically words. This model enables the comparison of documents by calculating the similarity between their vectors.
- **Concept**: In VSM, each document is converted into a vector where each dimension corresponds to a term from the vocabulary. The value in each dimension is often the frequency of the corresponding term in the document.
- **Application**: This vector representation is foundational for tasks like document similarity, classification, and clustering.

#### **2. Text Preprocessing Using SpaCy**
- **Importing SpaCy**: To preprocess text, SpaCy is a powerful library that can handle tokenization, stop word removal, lemmatization, and more.
- **Key Steps**:
  - **Tokenization**: Breaking down the text into individual words or tokens.
  - **Stop Word Removal**: Filtering out common words like "is", "the", etc., which may not contribute meaningfully to the analysis.
  - **Lemmatization**: Reducing words to their base or root form, e.g., "eating" becomes "eat".

#### **3. Implementing the Preprocessing Function in Python**

- **Step-by-Step Process**:
  1. **Import SpaCy and Load Model**:
     - `import spacy`
     - `nlp = spacy.load('en_core_web_sm')`
     - This loads the English language model for SpaCy.
  
  2. **Define Preprocessing Function**:
     - The function processes the text by tokenizing, removing stop words and punctuation, and applying lemmatization.

- **Sample Python Code**:

```python
import spacy

# Load SpaCy English model
nlp = spacy.load('en_core_web_sm')

def preprocess(text):
    # Create a SpaCy document
    doc = nlp(text)
    
    # Filter tokens: remove stop words and punctuation, and apply lemmatization
    filtered_tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct:
            filtered_tokens.append(token.lemma_)
    
    # Convert the list of tokens into a string
    processed_text = " ".join(filtered_tokens)
    
    return processed_text

# Example usage
example_text = "Loki is eating pizza"
processed_text = preprocess(example_text)
print("Processed Text:", processed_text)
```

### **Expected Output**
```plaintext
Processed Text: loki eat pizza
```

#### **4. Applying Preprocessing to a Corpus**

- **Batch Processing with List Comprehension**:
  - The preprocessing function can be applied to an entire corpus (a collection of documents) using Python’s list comprehension.

```python
# Sample corpus
corpus = [
    "Thor is eating pizza.",
    "Loki is eating sushi.",
    "Spider-Man is swinging through the city."
]

# Apply preprocessing to the entire corpus
processed_corpus = [preprocess(text) for text in corpus]

print("Processed Corpus:", processed_corpus)
```

### **Expected Output**
```plaintext
Processed Corpus: ['thor eat pizza', 'loki eat sushi', 'spider-man swing city']
```

### **Summary**
- **Vector Space Model (VSM)**: Converts text into a vector format for similarity and classification tasks.
- **Text Preprocessing**:
  - **SpaCy**: Handles tokenization, stop word removal, and lemmatization.
  - **Function Implementation**: The function processes text to make it ready for vectorization.
  - **Batch Processing**: The preprocessing can be efficiently applied to an entire corpus using list comprehension.
  
### **Notes on Bag of N-Grams and Handling Out-of-Vocabulary Words**

#### **1. Overview of Bag of N-Grams**
- **Bag of N-Grams Model**: An extension of the Bag of Words model that considers sequences of words (n-grams) rather than just individual words. This model captures the order of words by creating pairs (bigrams), triples (trigrams), or higher-order n-grams.
- **N-Gram Range**: 
  - **1-gram (Unigram)**: Considers individual words.
  - **2-gram (Bigram)**: Considers pairs of consecutive words.
  - **3-gram (Trigram)**: Considers triples of consecutive words.
  - The choice of n-gram range allows the model to capture more contextual information in the text.

#### **2. Implementing Bag of N-Grams with `CountVectorizer`**

- **Using `CountVectorizer`**:
  - The `CountVectorizer` class from `sklearn` can be used to convert text data into a vector (a list of numbers) using n-grams.
  - The `ngram_range` parameter specifies the range of n-grams to consider. For example, `ngram_range=(1, 2)` will consider both unigrams and bigrams.

- **Sample Python Code**:

```python
from sklearn.feature_extraction.text import CountVectorizer

# Example corpus after preprocessing
processed_corpus = [
    'thor eat pizza',
    'loki eat sushi',
    'spider-man swing city'
]

# Initialize CountVectorizer with ngram_range for unigrams and bigrams
vectorizer = CountVectorizer(ngram_range=(1, 2))

# Fit the vectorizer on the processed corpus to create a vocabulary
vectorizer.fit(processed_corpus)

# Display the vocabulary
vocabulary = vectorizer.vocabulary_
print("Vocabulary:", vocabulary)
```

### **Expected Output**
```plaintext
Vocabulary: {'thor': 8, 'eat': 1, 'pizza': 5, 'thor eat': 9, 'eat pizza': 2, 'loki': 3, 'sushi': 7, 'loki eat': 4, 'eat sushi': 0, 'spider-man': 6, 'swing': 10, 'city': 11, 'spider-man swing': 12, 'swing city': 13}
```

#### **3. Converting Text to Vectors**

- **Text to Vector Conversion**:
  - Once the vocabulary is created, you can convert new sentences into vectors using the `transform` method.
  - Each word or n-gram in the sentence is mapped to its corresponding index in the vocabulary, and a vector (sparse matrix) is generated.

```python
# Convert a new sentence to a vector
sentence = "thor eat pizza"
vector = vectorizer.transform([sentence]).toarray()

print("Vector for sentence '{}':".format(sentence), vector)
```

### **Expected Output**
```plaintext
Vector for sentence 'thor eat pizza': [[0 1 1 0 0 1 0 0 0 1 0 0 0 0]]
```

#### **4. Handling Out-of-Vocabulary (OOV) Words**

- **Out-of-Vocabulary (OOV) Problem**:
  - **Definition**: When a word in a new sentence does not exist in the vocabulary built during training, it is considered an out-of-vocabulary word.
  - **Impact**: OOV words cannot be represented by the existing vectorizer, leading to potential loss of information.

- **Example of OOV**:
  - Given a new sentence "hulk eat pizza", if "hulk" was not part of the original training corpus, it will not be represented in the vector.

```python
# New sentence with OOV word
oov_sentence = "hulk eat pizza"
oov_vector = vectorizer.transform([oov_sentence]).toarray()

print("Vector for OOV sentence '{}':".format(oov_sentence), oov_vector)
```

### **Expected Output**
```plaintext
Vector for OOV sentence 'hulk eat pizza': [[0 1 1 0 0 1 0 0 0 0 0 0 0 0]]
```

- **Explanation**:
  - The word "hulk" is not in the vocabulary, so its representation in the vector is absent (represented by 0s in the vector).

#### **5. Limitations of N-Gram Models**
- **Dimensionality and Sparsity**:
  - As the n-gram size increases, the number of possible n-grams grows exponentially, leading to a high-dimensional and sparse vector space.
  - This results in increased computational and memory requirements.
  
- **Out-of-Vocabulary (OOV) Issues**:
  - OOV words are not captured, potentially reducing the model's ability to generalize to
 
    ### **Notes on News Category Classification with Bag of N-Grams and Handling Class Imbalance**

#### **1. Overview of the News Category Classification Problem**
- **Problem Statement**: 
  - The task is to classify news articles into one of several predefined categories (e.g., Business, Sports, Science).
  - The dataset is from a JSON file containing news articles and their corresponding categories.

#### **2. Dataset Exploration**
- **Loading the Dataset**:
  - The dataset is loaded using Pandas into a DataFrame for further analysis and model training.

- **Sample Python Code**:
  
  ```python
  import pandas as pd

  # Load the JSON dataset into a Pandas DataFrame
  df = pd.read_json('path_to_your_dataset.json')

  # Display the shape of the dataset and a few initial records
  print(df.shape)
  print(df.head())
  ```

- **Expected Output**:
  ```plaintext
  (number_of_records, number_of_columns)
     category       headline       short_description  ...
  0  Business      ...              ...
  1  Sports        ...              ...
  ```

- **Exploratory Data Analysis (EDA)**:
  - Analyze the distribution of categories to check for class imbalance.
  
  ```python
  # Get the count of each category
  category_counts = df['category'].value_counts()
  print(category_counts)
  ```

- **Expected Output**:
  ```plaintext
  Business       4200
  Sports         4000
  Science        1381
  Crime          ...
  ```

#### **3. Handling Class Imbalance**
- **Class Imbalance**:
  - The dataset shows an imbalance with categories like "Business" and "Sports" having a significantly higher number of samples compared to "Science".

- **Approach**:
  - **Under-Sampling**: Simplest technique where the larger classes are reduced to match the size of the smallest class.
  - **Why Under-Sampling**: 
    - Although not ideal in a real-world scenario due to data wastage, it simplifies the tutorial and is easy to implement.

- **Sample Python Code for Under-Sampling**:
  
  ```python
  # Find the minimum number of samples in the smallest category
  min_samples = df['category'].value_counts().min()

  # Under-sample each category to match the smallest category size
  df_balanced = df.groupby('category').apply(lambda x: x.sample(min_samples)).reset_index(drop=True)

  # Verify the balanced dataset
  print(df_balanced['category'].value_counts())
  ```

- **Expected Output**:
  ```plaintext
  Business       1381
  Sports         1381
  Science        1381
  Crime          1381
  ...
  ```

#### **4. Model Training with Bag of N-Grams**
- **Using Bag of N-Grams**:
  - After handling the class imbalance, the next step is to convert the text into numerical vectors using the Bag of N-Grams model.
  - This allows the model to consider word sequences and capture context better than simple Bag of Words.

- **Sample Python Code**:
  
  ```python
  from sklearn.feature_extraction.text import CountVectorizer
  from sklearn.model_selection import train_test_split
  from sklearn.naive_bayes import MultinomialNB
  from sklearn.metrics import accuracy_score

  # Example corpus after preprocessing
  X = df_balanced['text']  # Assuming 'text' column contains news articles
  y = df_balanced['category']

  # Convert text data into n-gram vectors
  vectorizer = CountVectorizer(ngram_range=(1, 2))  # Unigrams and Bigrams
  X_vectorized = vectorizer.fit_transform(X)

  # Split the data into training and test sets
  X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

  # Train a simple model, e.g., Naive Bayes
  model = MultinomialNB()
  model.fit(X_train, y_train)

  # Predict and evaluate the model
  y_pred = model.predict(X_test)
  print("Accuracy:", accuracy_score(y_test, y_pred))
  ```

- **Expected Output**:
  ```plaintext
  Accuracy: X.XX (e.g., 0.75)
  ```

#### **5. Conclusion**
- **Class Imbalance Handling**:
  - Simple under-sampling was used to balance the dataset. In real-world applications, more sophisticated techniques like SMOTE or ensemble methods may be preferred to handle class imbalance without discarding data.

- **Bag of N-Grams**:
  - The use of Bag of N-Grams (unigrams and bigrams) in the `CountVectorizer` allowed the model to capture more context and potentially improve classification performance. 

### **Continuing with Train-Test Split and Model Preparation**

#### **1. **Converting Categories to Numerical Labels**
- **Mapping Categories to Numbers**:
  - Before feeding the categorical labels (e.g., "Business", "Science") into a machine learning model, they need to be converted into numerical labels.

- **Creating a Mapping Dictionary**:
  - You can manually create a dictionary that maps each category to a unique integer value.
  
  ```python
  # Define a dictionary mapping categories to numbers
  category_mapping = {
      'Business': 0,
      'Science': 1,
      'Sports': 2,
      # Add other categories accordingly
  }

  # Map the categories in the DataFrame to their corresponding numbers
  df_balanced['category_num'] = df_balanced['category'].map(category_mapping)
  ```

- **Expected DataFrame Output**:
  ```plaintext
  category    category_num
  Business    0
  Science     1
  Sports      2
  ...
  ```

#### **2. **Splitting Data into Training and Testing Sets**
- **Using `train_test_split`**:
  - The data is split into training and testing sets using the `train_test_split` function from Scikit-learn.

- **Important Parameters**:
  - **`test_size`**: Determines the proportion of the dataset to include in the test split. E.g., `test_size=0.2` means 20% of the data will be used for testing.
  - **`random_state`**: Ensures reproducibility of the results. Every time you run the notebook, the data split remains the same.
  - **`stratify`**: Ensures that the training and testing datasets have a similar distribution of the different classes, which helps in maintaining balance in the splits.

- **Sample Python Code**:
  
  ```python
  from sklearn.model_selection import train_test_split

  # Splitting the data into train and test sets
  X_train, X_test, y_train, y_test = train_test_split(
      df_balanced['text'],       # Features (news articles)
      df_balanced['category_num'], # Target variable (numerical categories)
      test_size=0.2,             # 20% test size
      random_state=42,           # For reproducibility
      stratify=df_balanced['category_num']  # Maintain distribution of classes
  )
  ```

- **Verifying the Stratified Split**:
  - You can check the distribution of the target variable in both the training and test datasets to ensure they are balanced.

  ```python
  print("Training set distribution:")
  print(y_train.value_counts())

  print("Test set distribution:")
  print(y_test.value_counts())
  ```

- **Expected Output**:
  ```plaintext
  Training set distribution:
  0    1105
  1    1105
  2    1105
  ...
  Test set distribution:
  0    276
  1    276
  2    276
  ...
  ```

#### **3. **Conclusion: Ready for Model Training**
- With the balanced dataset and a properly stratified train-test split, the next steps would involve vectorizing the text data using methods like Bag of N-Grams and then training the classification model (e.g., Naive Bayes, Logistic Regression).
  
### **Building and Evaluating Machine Learning Models with Text Data**

#### **1. **Building the Naive Bayes Model with Bag of Words**
- **Choosing Naive Bayes**:
  - Naive Bayes is often recommended for text classification problems due to its simplicity and efficiency, especially with high-dimensional data like text.
  
- **Creating a Pipeline**:
  - A pipeline simplifies the process by chaining steps together, such as vectorization and model training.
  - **Vectorizer**: Converts text into a matrix of token counts.
  - **Classifier**: Multinomial Naive Bayes is used as the model.

  ```python
  from sklearn.pipeline import Pipeline
  from sklearn.feature_extraction.text import CountVectorizer
  from sklearn.naive_bayes import MultinomialNB
  from sklearn.metrics import classification_report
  
  # Creating a pipeline for Bag of Words model
  pipeline = Pipeline([
      ('vectorizer', CountVectorizer()),  # Bag of Words
      ('classifier', MultinomialNB())     # Naive Bayes classifier
  ])
  
  # Training the model
  pipeline.fit(X_train, y_train)
  
  # Making predictions on the test set
  y_pred = pipeline.predict(X_test)
  
  # Printing the classification report
  print(classification_report(y_test, y_pred))
  ```

#### **2. **Exploring Different N-Gram Models**
- **Modifying the Pipeline for N-Grams**:
  - By changing the `ngram_range` parameter, you can experiment with unigrams, bigrams, and trigrams.
  
  ```python
  # Bag of N-Grams model (Unigram + Bigram)
  pipeline_ngram = Pipeline([
      ('vectorizer', CountVectorizer(ngram_range=(1, 2))),  # Unigram + Bigram
      ('classifier', MultinomialNB())
  ])
  
  # Training the model
  pipeline_ngram.fit(X_train, y_train)
  
  # Making predictions
  y_pred_ngram = pipeline_ngram.predict(X_test)
  
  # Classification report
  print(classification_report(y_test, y_pred_ngram))
  ```

- **Performance Comparison**:
  - After running the above, you might find that unigrams perform better than bigrams or trigrams in certain cases. This is expected as more complex models (like trigrams) might overfit on small datasets.

  - **Performance Summary**:
    - **Bag of Words (Unigrams)**: High accuracy, especially in straightforward cases.
    - **N-Grams (Bigrams/Trigrams)**: May add context but could decrease performance due to sparsity or overfitting.

#### **3. **Predicting and Analyzing Results**
- **Example Predictions**:
  - You can use the trained model to make predictions on specific examples from the test set and compare them with the true labels.
  
  ```python
  # Example predictions
  print("First 5 test samples:")
  print(X_test[:5])

  print("True labels:")
  print(y_test[:5])

  print("Predicted labels:")
  print(y_pred[:5])
  ```

  - This allows you to visually inspect where the model performs well or where it makes mistakes, such as misclassifying business articles as crime-related.

#### **4. **Incorporating Preprocessing**
- **Adding a Preprocessing Step**:
  - Initially, the model is trained on raw text data. However, preprocessing can improve performance by removing noise such as stopwords, punctuation, and normalizing words.

  ```python
  # Adding preprocessed text to the DataFrame
  df_balanced['preprocessed_text'] = df_balanced['text'].apply(preprocess)

  # Updating train-test split with preprocessed text
  X_train, X_test, y_train, y_test = train_test_split(
      df_balanced['preprocessed_text'],
      df_balanced['category_num'],
      test_size=0.2,
      random_state=42,
      stratify=df_balanced['category_num']
  )
  ```

  - **Pipeline with Preprocessing**:
    - Integrate this preprocessed text column into your existing pipeline to see if the model's performance improves.

#### **5. **Summary and Recommendations**
- **Model Selection**:
  - While Naive Bayes works well for text classification, it's good practice to compare it with other models (like Random Forest, K-NN) to ensure the best performance for your specific dataset.
  
- **Experimentation**:
  - Experimenting with different n-gram ranges, preprocessing techniques, and models is crucial in NLP tasks as different datasets and tasks can yield varying results.
 
  ### **Model Comparison: Pre-Processed vs. Raw Text**

#### **1. **Training and Evaluating Models**

- **With Pre-Processed Text**:
  - After applying preprocessing techniques (e.g., removing stop words, lemmatization), retrain the model with the pre-processed text and compare its performance to the model trained with raw text.

  ```python
  # Assuming 'preprocessed_text' column is already created
  X_train, X_test, y_train, y_test = train_test_split(
      df_balanced['preprocessed_text'],
      df_balanced['category_num'],
      test_size=0.2,
      random_state=42,
      stratify=df_balanced['category_num']
  )
  
  # Training the Naive Bayes model with preprocessed text
  pipeline_preprocessed = Pipeline([
      ('vectorizer', CountVectorizer()),
      ('classifier', MultinomialNB())
  ])
  
  pipeline_preprocessed.fit(X_train, y_train)
  y_pred_preprocessed = pipeline_preprocessed.predict(X_test)
  
  print(classification_report(y_test, y_pred_preprocessed))
  ```

- **Comparison of Results**:
  - **Performance Metrics**: Look at precision, recall, F1 score, and accuracy.
  - **Visualization**: Using tools like confusion matrices can help visualize where the model is making mistakes.

  ```python
  from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
  
  cm = confusion_matrix(y_test, y_pred_preprocessed)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Business', 'Crime', 'Politics', 'Science', 'Sports'])
  disp.plot()
  ```

#### **2. **Impact of Preprocessing**

- **Improved Results**:
  - In many cases, preprocessing leads to better results by cleaning and normalizing the data. 
  - **Example**: In your case, F1 scores improved after preprocessing, indicating better performance in classifying texts.

- **General Advice**:
  - While preprocessing generally improves model performance, it's not always the case. It depends on the specific dataset and problem. Always validate with experiments.

#### **3. **Resources and Practice**

- **Exercises**:
  - Check the video description or notebook for additional exercises and resources. Practicing these exercises helps solidify your understanding and skills.

- **Active Learning**:
  - Watching videos alone isn't enough; hands-on practice is crucial. Try different models, preprocessing techniques, and parameters to understand their impacts.

### **Summary**

- **Pre-Processing**:
  - Often beneficial for improving model performance in NLP tasks. Always test and validate its impact on your specific problem.

- **Practice and Experimentation**:
  - Key to mastering NLP and machine learning. Apply learned concepts to different datasets and problems to gain practical experience.

