## Word Embeddings Overview with Gensim

### Introduction
In the previous video, we covered text classification using SpaCy word embeddings. In this video, we'll explore word embeddings with another Python library called Gensim. While SpaCy is used for various NLP tasks, Gensim is primarily used for topic modeling. However, Gensim is also convenient for handling word vectors.

### Gensim Library
- **Gensim**: An NLP library mainly used for topic modeling but also offers tools for working with word embeddings.
- **Website**: You can find more information and install Gensim from its official website.

### Installation
To install Gensim, use the following command:
```bash
pip install gensim
```

### Loading Word Embeddings
Gensim provides various pre-trained word vectors. Here’s how to load them:

1. **Importing Gensim**:
   ```python
   from gensim.models import KeyedVectors
   ```

2. **Loading Word Embeddings**:
   Gensim offers different models, such as Word2Vec trained on Google News. This model is large (about 1.6 GB) and might take some time to download. To load the Google News Word2Vec model, you can use:
   ```python
   from gensim.downloader import load
   model = load("word2vec-google-news-300")
   ```

### Available Models
- **Google News Word2Vec**: Trained on Google News dataset with 300-dimensional vectors.
- **GloVe Models**: Gensim also supports models trained using the GloVe approach.

### Example Code and Expected Output

#### Example 1: Loading a Pre-trained Model
```python
from gensim.downloader import load

# Load the pre-trained Google News Word2Vec model
model = load("word2vec-google-news-300")
```
**Expected Output**: The model will be loaded, which may take a few minutes depending on your internet speed and system performance.

#### Example 2: Using the Loaded Model
```python
# Get the vector for a specific word
vector = model['computer']

# Find the most similar words to 'computer'
similar_words = model.most_similar('computer', topn=5)

print("Vector for 'computer':", vector)
print("Most similar words to 'computer':", similar_words)
```
**Expected Output**:
```plaintext
Vector for 'computer': [0.2, -0.1, ...]  # This will be a 300-dimensional vector
Most similar words to 'computer': [('laptop', 0.85), ('PC', 0.82), ...]
```


### Word Embeddings: Detailed Overview

#### Model Size and Details
- **Model Size**: The Google News Word2Vec model is approximately 1.6 GB and trained on 100 billion words from Google News articles.
- **Number of Vectors**: Around 3 million vectors, making it a substantial model.
- **Smaller Models**: For specific tasks like Twitter analysis, smaller models (~199 MB) with 1.1 million vectors are available. These often use GloVe (Global Vectors for Word Representation) technique, in contrast to Word2Vec.

#### Similarity Checking
Gensim allows checking the similarity between words using the pre-trained models. Here’s how to perform this:

1. **Checking Similarity Between Words**:
   ```python
   # Check the similarity between two words
   similarity_score = model.similarity('great', 'good')
   print("Similarity between 'great' and 'good':", similarity_score)
   ```

   **Expected Output**:
   ```plaintext
   Similarity between 'great' and 'good': 0.7  # Example value
   ```

   **Explanation**: The similarity score ranges from -1 to 1. A score closer to 1 indicates high similarity, while a score closer to -1 indicates dissimilarity. A score of 0 suggests no similarity. The similarity is context-dependent.

2. **Finding Similar Words**:
   ```python
   # Find similar words to 'profit'
   similar_words_profit = model.most_similar('profit', topn=5)
   print("Most similar words to 'profit':", similar_words_profit)
   ```

   **Expected Output**:
   ```plaintext
   Most similar words to 'profit': [('gain', 0.85), ('income', 0.82), ...]
   ```

   **Explanation**: The most similar words are determined based on their context and usage in the training data. Results can sometimes be counterintuitive if words appear in similar contexts but have different meanings.

#### Contextual Similarity
- **Context-Dependent**: Similarity scores reflect how words appear in similar contexts rather than being strict synonyms. For example, "profit" and "gain" might have high similarity if they frequently occur in similar contexts.

### Understanding Word Similarity in Word Embeddings

#### Similarity Functions in Gensim
- **Similarity Function**: The `similarity` function calculates how similar two words are based on their context in the training corpus. 
  ```python
  similarity_score = model.similarity('good', 'bad')
  print("Similarity between 'good' and 'bad':", similarity_score)
  ```

  **Expected Output**:
  ```plaintext
  Similarity between 'good' and 'bad': 0.7  # Example value
  ```

  **Explanation**: Despite "good" and "bad" being antonyms, their similarity score can be relatively high. This is because they appear in similar contexts in the training data. For example, both words might appear in sentences discussing feelings or evaluations, even though they express opposite sentiments.

- **Most Similar Function**: The `most_similar` function finds words most similar to a given word based on the model.
  ```python
  similar_words_good = model.most_similar('good', topn=5)
  print("Most similar words to 'good':", similar_words_good)
  ```

  **Expected Output**:
  ```plaintext
  Most similar words to 'good': [('great', 0.85), ('excellent', 0.82), ...]
  ```

  **Explanation**: This function lists words that appear in similar contexts as the input word, which may include synonyms or words that share contextual usage.

#### Contextual Similarity
- **Contextual Understanding**: Word embeddings capture how words are used in similar contexts rather than their strict linguistic meanings. For example, in the Google News corpus, "good" and "bad" might appear in similar types of sentences, leading to a higher similarity score despite their antonymous relationship.

#### Training Approach
- **Self-Supervised Learning**: Word embeddings models, like Word2Vec, are trained using a large corpus of text through self-supervised learning. This involves generating word samples and training the model to predict words based on their context within the text.

### Key Points
- **Similarity is Context-Dependent**: Similarity scores reflect contextual usage rather than direct semantic meaning.
- **Antonyms in Similar Contexts**: Words with opposite meanings can still have high similarity scores if they frequently occur in similar contexts.

### Advanced Word Embeddings with Gensim

#### Practical Uses of Word Embeddings

1. **Handling Synonyms and Related Words**:
   Word embeddings are useful for recognizing synonyms and related terms that traditional models like TF-IDF and Bag of Words cannot handle effectively.
   ```python
   # Find similar words to 'dog'
   similar_words_dog = model.most_similar('dog', topn=5)
   print("Most similar words to 'dog':", similar_words_dog)
   ```

   **Expected Output**:
   ```plaintext
   Most similar words to 'dog': [('puppy', 0.85), ('golden_retriever', 0.82), ...]
   ```

2. **Contextual Similarity**:
   Words like "cat" and "dog" may have high similarity scores due to their appearance in similar contexts, even though they are different species.
   ```python
   # Find similarity between 'cat' and 'dog'
   similarity_cat_dog = model.similarity('cat', 'dog')
   print("Similarity between 'cat' and 'dog':", similarity_cat_dog)
   ```

   **Expected Output**:
   ```plaintext
   Similarity between 'cat' and 'dog': 0.7  # Example value
   ```

#### Word Vector Arithmetic

Word embeddings allow for interesting operations such as vector arithmetic to find relationships between words:
- **Example 1**: King - Woman + Man = ?
  ```python
  # Perform vector arithmetic
  result = model.most_similar(positive=['king', 'man'], negative=['woman'], topn=1)
  print("King - Woman + Man:", result)
  ```

  **Expected Output**:
  ```plaintext
  King - Woman + Man: [('queen', 0.85)]
  ```

- **Example 2**: France - Paris + Berlin = ?
  ```python
  # Perform vector arithmetic
  result = model.most_similar(positive=['berlin'], negative=['paris'], topn=1)
  print("France - Paris + Berlin:", result)
  ```

  **Expected Output**:
  ```plaintext
  France - Paris + Berlin: [('germany', 0.8)]
  ```

  **Explanation**: These operations can reveal relationships such as countries and their capitals. In the second example, subtracting Paris (the capital of France) and adding Berlin (another capital) leads to Germany, illustrating how embeddings capture semantic relationships.

### Gensim vs. SpaCy
- **Gensim**: Offers convenient functions for word vector arithmetic and similarity checks.
- **SpaCy**: While powerful, may not be as straightforward for certain types of word vector manipulations.

### Key Points
- **Semantic Understanding**: Word embeddings capture semantic relationships, allowing for tasks like finding similar words and performing vector arithmetic.
- **Contextual Learning**: Words appearing in similar contexts have similar embeddings, enhancing the model’s ability to understand nuanced meanings.

### Advanced Features and Model Loading in Gensim

#### Word Vector Arithmetic

1. **Vector Arithmetic Examples**:
   - **France - Paris + Berlin**:
     ```python
     result = model.most_similar(positive=['france', 'berlin'], negative=['paris'], topn=1)
     print("France - Paris + Berlin:", result)
     ```

     **Expected Output**:
     ```plaintext
     France - Paris + Berlin: [('germany', 0.8)]
     ```

   - **King - Woman + Man**:
     ```python
     result = model.most_similar(positive=['king', 'man'], negative=['woman'], topn=1)
     print("King - Woman + Man:", result)
     ```

     **Expected Output**:
     ```plaintext
     King - Woman + Man: [('queen', 0.85)]
     ```

   **Explanation**: These operations illustrate the model's ability to understand and manipulate semantic relationships. For instance, "King - Woman + Man" should yield "Queen", and "France - Paris + Berlin" results in "Germany", reflecting how well embeddings capture such relationships.

2. **Handling Contextual Similarity**:
   - **Correcting Example**:
     ```python
     # Correct example
     result = model.most_similar(positive=['king', 'man'], negative=['woman'], topn=1)
     print("King - Woman + Man:", result)
     ```

     **Expected Output**:
     ```plaintext
     King - Woman + Man: [('queen', 0.85)]
     ```

   **Explanation**: The model captures gender relationships in royalty and accurately reflects common word associations.

#### Outlier Detection

- **Detecting Odd Words**:
  Gensim can identify words that do not fit within a given set.
  ```python
  # Identify the outlier
  odd_one_out = model.doesnt_match(['cat', 'dog', 'lion', 'microsoft'])
  print("Odd one out:", odd_one_out)
  ```

  **Expected Output**:
  ```plaintext
  Odd one out: 'microsoft'
  ```

  **Explanation**: The model recognizes "Microsoft" as an outlier because it is not an animal, while the others are.

#### Loading and Using Models

1. **Loading a Smaller Model**:
   - **Twitter Word2Vec Model**:
     ```python
     from gensim.downloader import load

     # Load the Twitter Word2Vec model
     twitter_model = load("word2vec-twitter-25")
     ```

     **Expected Output**: The model will be loaded, providing a smaller, more focused set of embeddings trained on 2 billion tweets.

2. **GloVe Models**:
   - **GloVe Technique**: Developed by Stanford, GloVe (Global Vectors for Word Representation) is another approach for learning word vectors.

     **Reference**: [Stanford GloVe Page](https://nlp.stanford.edu/projects/glove/)
### Summary of Gensim Word Embeddings and Advanced Features

#### Model Performance and Variability

1. **Model Loading and Variability**:
   - **Twitter Word2Vec Model**: When using different word embedding models, such as the Twitter Word2Vec model, you might see different results compared to models trained on other datasets like Google News. For example, the similarity results for the word "good" might differ between models due to the varying contexts in which they were trained.
   ```python
   # Find most similar words to 'good' using Twitter model
   similar_words_good_twitter = twitter_model.most_similar('good', topn=5)
   print("Most similar words to 'good' (Twitter):", similar_words_good_twitter)
   ```

   **Expected Output**:
   ```plaintext
   Most similar words to 'good' (Twitter): [('today', 0.65), ('excellent', 0.60), ...]
   ```

2. **Handling Outliers**:
   - **Odd One Out**: Identifying words that do not fit well with a given set.
   ```python
   # Identify the outlier in a set of words
   odd_one_out = twitter_model.doesnt_match(['breakfast', 'cereal', 'dinner', 'lunch'])
   print("Odd one out:", odd_one_out)
   ```

   **Expected Output**:
   ```plaintext
   Odd one out: 'cereal'
   ```

   **Explanation**: The model may determine "cereal" as an outlier if it doesn't fit the context of meal types compared to the others.

3. **Vector Arithmetic with Different Models**:
   - **Example**: Performing vector arithmetic with different embeddings can yield different results based on the model used.
   ```python
   # Example of vector arithmetic
   result = twitter_model.most_similar(positive=['king', 'man'], negative=['woman'], topn=1)
   print("King - Woman + Man (Twitter):", result)
   ```

   **Expected Output**:
   ```plaintext
   King - Woman + Man (Twitter): [('queen', 0.85)]
   ```

#### Techniques and Models

1. **Techniques**:
   - **Word2Vec**: A technique for learning word embeddings by predicting words in context.
   - **GloVe (Global Vectors for Word Representation)**: Another technique for learning word embeddings by aggregating global word-word co-occurrence statistics from a corpus.

2. **Datasets**:
   - **Google News 300**: A model trained on the Google News dataset with 300-dimensional vectors.
   - **Twitter 25**: A smaller model trained on 2 billion tweets with 25-dimensional vectors.

3. **Libraries**:
   - **Gensim**: Provides tools for working with pre-trained word vectors and performing vector arithmetic.
   - **SpaCy**: Another library for NLP that supports various word vector techniques but may be less straightforward for certain operations compared to Gensim.

#### Best Practices

- **Understanding Variability**: Be aware that different models and datasets can yield different results based on their training data and techniques.
- **Organizing Knowledge**: Given the complexity and variety of NLP techniques, it's important to organize and understand the different methods, datasets, and libraries to effectively apply them.

#### Next Steps

- **Text Classification**: In the next video, text classification using Gensim word vectors will be explored. This will build on the understanding of word embeddings and their applications.
