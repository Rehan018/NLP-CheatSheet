### Notes on Word Embeddings with SpaCy

**1. Introduction to Word Embeddings:**

- **Definition:** Word embeddings are dense vector representations of words that capture their meanings based on their usage in context. They are used to represent words in a continuous vector space where semantically similar words are mapped close to each other.

**2. SpaCy and Word Embeddings:**

- **SpaCy Models:** To utilize word embeddings in SpaCy, you need to load a model that includes them. SpaCy provides different models with varying levels of complexity:
  - **`en_core_web_sm`**: Small model, no vectors.
  - **`en_core_web_md`**: Medium model, contains 514,000 unique word vectors with 300 dimensions each.
  - **`en_core_web_lg`**: Large model, contains 20 million unique word vectors with 300 dimensions each.

**3. Installing SpaCy Models:**

To install the large model, run the following command in your command prompt:

```bash
python -m spacy download en_core_web_lg
```

**4. Comparing Word Vectors:**

Here's how to compare word vectors for different words using the SpaCy library in Python:

```python
import spacy

# Load the large SpaCy model
nlp = spacy.load('en_core_web_lg')

# Define words to compare
words = ["dog", "cat", "banana", "garbage"]

# Process words and get vectors
vectors = {word: nlp(word).vector for word in words}

# Compare vectors
for word, vector in vectors.items():
    print(f"Word: {word}")
    print(f"Vector: {vector[:10]}...")  # Print only the first 10 dimensions for brevity
    print(f"Vector Length: {len(vector)}")
    print()
```

**5. Explanation of the Code:**

- **Loading the Model:** The `spacy.load('en_core_web_lg')` function loads the large SpaCy model which includes word vectors.
- **Processing Words:** `nlp(word).vector` gets the vector representation of each word.
- **Comparing Vectors:** For demonstration purposes, only the first 10 dimensions of each vector are printed to keep the output manageable.

### Notes on Token Vectors and Vocabulary in SpaCy

**1. Checking Token Vectors:**

In SpaCy, each token (word) processed by the model has attributes related to word embeddings. You can inspect these attributes to understand which tokens have vectors and which do not.

**2. Key Attributes:**

- **`token.text`**: The actual text of the token.
- **`token.has_vector`**: A boolean flag indicating if the token has a vector representation.
- **`token.is_oov`**: A boolean flag indicating if the token is out of vocabulary (OOV). This means the token was not encountered during the model's training.

**3. Example Code:**

The following code demonstrates how to check these attributes for tokens:

```python
import spacy

# Load the large SpaCy model
nlp = spacy.load('en_core_web_lg')

# Define a sample text
text = "dog cat banana garbage"

# Process the text
doc = nlp(text)

# Print token attributes
for token in doc:
    print(f"Token: {token.text}")
    print(f"Has Vector: {token.has_vector}")
    print(f"Is OOV: {token.is_oov}")
    print()
```

**4. Explanation:**

- **Processing the Text:** `nlp(text)` processes the input text and returns a `Doc` object containing tokens.
- **Attributes Inspection:**
  - **`token.has_vector`**: This will be `True` for common words like "dog", "cat", and "banana" which have embeddings. For less common or random words like "garbage", it may be `False`.
  - **`token.is_oov`**: This will be `True` for words that are not in the model's vocabulary and thus do not have vectors.

**5. Model Training:**

SpaCy's large model uses GloVe (Global Vectors for Word Representation) embeddings, which are trained on popular English datasets and capture general English knowledge. However, words not encountered in the training data will not have vectors and will be flagged as out of vocabulary (OOV).

### Notes on Comparing Word Vectors and Similarity in SpaCy

**1. Printing Word Vectors:**

To print the vector of a specific token, you can access its `vector` attribute. For example, to print the vector for the word "dog":

```python
import spacy

# Load the large SpaCy model
nlp = spacy.load('en_core_web_lg')

# Define a sample text
text = "dog cat banana"

# Process the text
doc = nlp(text)

# Access and print the vector for the first token ("dog")
print(f"Vector for 'dog': {doc[0].vector}")
print(f"Vector Shape: {doc[0].vector.shape}")
```

**2. Understanding Vector Dimensions:**

The vector size for each token is 300 dimensions. This can be confirmed using the `shape` attribute:

```python
print(f"Vector Shape: {doc[0].vector.shape}")  # Should print (300,)
```

**3. Sentence Vectors:**

For a single-word sentence, the sentence vector is identical to the word vector. For multi-word sentences, SpaCy computes the sentence vector as the average of individual word vectors. 

**4. Comparing Word Vectors:**

To compare the similarity between word vectors, you can use SpaCy's similarity methods. Here’s how you can compare the word "bread" with other words:

```python
# Define words for comparison
words_to_compare = ["sandwich", "burger", "cat", "tiger", "human", "wheat"]

# Create a vector for the word "bread"
bread_vector = nlp("bread").vector

# Compare "bread" with other words
for word in words_to_compare:
    word_vector = nlp(word).vector
    similarity = bread_vector @ word_vector / (np.linalg.norm(bread_vector) * np.linalg.norm(word_vector))  # Cosine similarity
    print(f"Similarity between 'bread' and '{word}': {similarity}")
```

**5. Explanation of Similarity Calculation:**

- **Cosine Similarity:** The similarity between two vectors is computed as the dot product of the vectors divided by the product of their magnitudes (norms). Higher similarity values indicate more similar words.

**6. Summary of Expected Results:**

- Words like "sandwich" and "burger" should have high similarity with "bread" as they are contextually related.
- Words like "cat" and "bread" should have lower similarity as they are less contextually related.

**7. Practical Application:**

Comparing word vectors helps in understanding the semantic relationships between words. This technique can be used in various NLP applications like text classification, recommendation systems, and more.

### Notes on Similarity Calculation with SpaCy

**1. Calculating Similarity Between Tokens:**

To calculate and print the similarity between tokens, you can use SpaCy's built-in `similarity` method. This method computes the similarity between the vectors of two tokens.

**2. Example Code:**

```python
import spacy

# Load the large SpaCy model
nlp = spacy.load('en_core_web_lg')

# Define words for comparison
base_word = "bread"
words_to_compare = ["sandwich", "burger", "cat", "tiger", "human", "wheat"]

# Process the base word and other words
base_token = nlp(base_word)
tokens = [nlp(word) for word in words_to_compare]

# Print similarity
for token in tokens:
    similarity = base_token.similarity(token)
    print(f"Similarity between '{base_word}' and '{token.text}': {similarity:.2f}")
```

**3. Explanation:**

- **`base_token.similarity(token)`**: This calculates the cosine similarity between the vector of the `base_token` and the `token`. The result ranges from -1 (completely dissimilar) to 1 (completely similar).
- **Example Similarities:**
  - **Bread and Sandwich**: Higher similarity, close to 1 (e.g., 0.6) because they are contextually related.
  - **Bread and Cat**: Lower similarity, close to 0 (e.g., 0.06) because they are not contextually related.

**4. Understanding Model Limitations:**

- **Contextual Similarity:** The model measures similarity based on how often words appear together in similar contexts. Words that frequently appear in similar contexts will have higher similarity scores.
- **Model Training:** The SpaCy model is trained on large corpora like Google News and Wikipedia. It captures general language patterns, but may not perfectly understand specific contexts or domain-specific terminology.

**5. Practical Insights:**

- **Not Perfect:** Similarity scores provide a general sense of relatedness but may not always reflect nuanced or domain-specific meanings.
- **Contextual Clustering:** Words with high similarity scores are often used in similar contexts but may not be semantically identical. For instance, "profit" and "loss" are contextually similar because they appear in financial discussions.


### Notes on Similarity and Function for Comparing Word Vectors in SpaCy

**1. Understanding Similarity:**

- **Contextual Similarity:** Similarity in SpaCy is based on how often words appear in similar contexts rather than their semantic meanings. For example, "profit" and "loss" may have high similarity because they often appear together in financial contexts, despite being antonyms.

**2. Creating a Similarity Function:**

You can create a function to compare a base word with a list of other words, printing their similarities. This function will use SpaCy's similarity calculations.

**3. Example Code:**

Here’s how to implement a function to compare word similarities:

```python
import spacy

# Load the large SpaCy model
nlp = spacy.load('en_core_web_lg')

def print_similarity(base_word, words_to_compare):
    # Process the base word and create its token
    base_token = nlp(base_word)
    
    # Process the words to compare
    tokens = [nlp(word) for word in words_to_compare]
    
    # Print similarity
    for token in tokens:
        similarity = base_token.similarity(token)
        print(f"Similarity between '{base_word}' and '{token.text}': {similarity:.2f}")

# Example usage
base_word = "iPhone"
words_to_compare = ["apple", "Samsung", "dog", "kitten"]
print_similarity(base_word, words_to_compare)
```

**4. Explanation:**

- **Function `print_similarity`**: 
  - **Inputs:** `base_word` (the reference word) and `words_to_compare` (a list of words to compare with the base word).
  - **Process:** Converts the base word and comparison words into SpaCy tokens, calculates their similarities, and prints the results.
  
- **Sample Output:**
  - **iPhone and Apple:** Similarity might be lower if "Apple" is less contextually relevant in the model's training data.
  - **iPhone and Samsung:** Higher similarity due to frequent comparisons in news and articles.
  - **iPhone and Dog:** Low similarity as dogs and iPhones don't appear together frequently.

**5. Notes on Model Limitations:**

- **Training Data Influence:** The similarity results reflect the contexts found in the training data (e.g., news articles). The model may show higher similarity between "Samsung" and "iPhone" if they are often discussed together in the data.
- **Contextual Relevance:** Words appearing together in the same contexts will have higher similarity scores, which might not always align with logical or semantic similarities.

### Notes on Word Vectors and Similarity in SpaCy

**1. Handling Out of Vocabulary (OOV) Words:**

- **OOV Words:** If a word does not appear in the model's training corpus, it will not have a vector representation and will be flagged as out of vocabulary (OOV).
- **Regional Languages:** For example, a word from Gujarati or any other regional language will not be recognized by SpaCy's English model. To handle such words, a model trained on the respective language corpus would be required.

**2. Vector Operations and Cosine Similarity:**

- **Vector Arithmetic:** You can perform operations on word vectors to explore relationships between words. For example, subtracting the vector of "man" from "king" and adding the vector of "woman" should ideally result in a vector close to "queen" if the model captures the gender relationship accurately.
- **Cosine Similarity:** Measures the cosine of the angle between two vectors, reflecting their similarity. Values close to 1 indicate high similarity, while values close to 0 indicate low similarity.

**3. Example Code for Vector Arithmetic and Similarity:**

```python
import spacy
from sklearn.metrics.pairwise import cosine_similarity

# Load the large SpaCy model
nlp = spacy.load('en_core_web_lg')

# Define words
king = nlp("king")
man = nlp("man")
woman = nlp("woman")
queen = nlp("queen")

# Compute vector arithmetic
result_vector = king.vector - man.vector + woman.vector

# Compute cosine similarity with "queen"
similarity = cosine_similarity([result_vector], [queen.vector])[0][0]

print(f"Similarity between the result of 'king - man + woman' and 'queen': {similarity:.2f}")
```

**4. Explanation:**

- **Vector Arithmetic:** `king.vector - man.vector + woman.vector` performs a vector operation to find a result that ideally represents "queen".
- **Cosine Similarity Calculation:** `cosine_similarity` compares the result vector with the vector for "queen" to quantify their similarity.

**5. Understanding Similarity Scores:**

- **Expected Results:** A similarity score of around 0.61 indicates that the result is reasonably close to "queen" but not perfect. This reflects that while the model captures some semantic relationships, it may not always achieve perfect results due to limitations in the training data or model capacity.

**6. Future Topics:**

- **Gensim Word Vectors:** Explore word vectors using the Gensim library, which provides additional capabilities and options for working with word embeddings.
- **Text Classification:** Learn how to use SpaCy for text classification, applying word vectors and other NLP techniques for categorizing text data.

