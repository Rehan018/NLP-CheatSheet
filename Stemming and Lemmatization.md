### Stemming and Lemmatization in NLP

#### Introduction
Stemming and lemmatization are crucial pre-processing steps in building NLP applications. These processes help in reducing words to their base or root forms, aiding in better text classification, search results, and overall language understanding. Let's explore both concepts in detail with code examples.

---

### 1. **Stemming**

**Definition:**  
Stemming is the process of reducing a word to its root form by applying simple heuristic rules, such as removing common suffixes like `-ing`, `-ed`, or `-able`. Stemming doesn't require linguistic knowledge of the language and is often considered a "crude" or "simplistic" approach.

**Example Use Case:**  
Consider a sentiment analysis model where you encounter words like "talking," "talked," and "talks." All these words share the same root, "talk." Stemming will reduce them to this base form.

**Code Implementation:**

We'll use the **Natural Language Toolkit (NLTK)** in Python for stemming, as spaCy does not support stemming.

```python
import nltk
from nltk.stem import PorterStemmer

# Initialize the PorterStemmer
stemmer = PorterStemmer()

# Example words
words = ["talking", "talked", "talks", "adjustable", "running", "ability"]

# Apply stemming
stemmed_words = [stemmer.stem(word) for word in words]

# Output the results
for word, stemmed in zip(words, stemmed_words):
    print(f"Original: {word} -> Stemmed: {stemmed}")
```

**Expected Output:**
```
Original: talking -> Stemmed: talk
Original: talked -> Stemmed: talk
Original: talks -> Stemmed: talk
Original: adjustable -> Stemmed: adjust
Original: running -> Stemmed: run
Original: ability -> Stemmed: abil
```

**Explanation:**  
Notice that while stemming effectively reduces words like "talking" and "adjustable" to their roots, it sometimes produces non-meaningful words, such as "abil" from "ability." This is one of the limitations of stemming.

---

### 2. **Lemmatization**

**Definition:**  
Lemmatization is a more sophisticated process that reduces a word to its base form, known as a lemma, by considering the word's meaning and linguistic rules. Unlike stemming, lemmatization requires knowledge of the language and is context-aware.

**Example Use Case:**  
In cases where you encounter words like "ate," "eaten," and "eating," the lemma would be "eat." Lemmatization ensures that the derived base form is a valid word in the language.

**Code Implementation:**

We'll use **spaCy** for lemmatization, as it provides a powerful and linguistically accurate lemmatizer.

```python
import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Example words
words = ["talking", "talked", "talks", "ate", "eating", "adjustable"]

# Apply lemmatization
lemmas = [token.lemma_ for word in words for token in nlp(word)]

# Output the results
for word, lemma in zip(words, lemmas):
    print(f"Original: {word} -> Lemma: {lemma}")
```

**Expected Output:**
```
Original: talking -> Lemma: talk
Original: talked -> Lemma: talk
Original: talks -> Lemma: talk
Original: ate -> Lemma: eat
Original: eating -> Lemma: eat
Original: adjustable -> Lemma: adjust
```

**Explanation:**  
Lemmatization accurately reduces words to their base forms, ensuring that the output is meaningful within the language. For example, "ate" is correctly lemmatized to "eat," and "adjustable" to "adjust," preserving the word's validity.

---

### 3. **Comparison of Stemming and Lemmatization**

- **Stemming:** 
  - Uses simple rules and heuristics.
  - Faster but can produce non-meaningful stems (e.g., "abil" from "ability").
  - Example: "running" → "run," "talked" → "talk."

- **Lemmatization:** 
  - Requires linguistic knowledge and context.
  - Slower but ensures valid lemmas (e.g., "ate" → "eat").
  - Example: "running" → "run," "ate" → "eat."

**When to Use:**
- **Stemming:** Useful in scenarios where speed is crucial, and minor inaccuracies are acceptable, such as in simple search engines or basic text classification.
- **Lemmatization:** Preferred when accuracy is vital, and the text's linguistic integrity must be preserved, such as in sentiment analysis, machine translation, or any sophisticated NLP tasks.

---

Both stemming and lemmatization have their place in NLP. While stemming is faster and less resource-intensive, it may lead to errors in word interpretation. Lemmatization, on the other hand, is more accurate but requires more computational resources and language knowledge. Depending on the application, you might choose one over the other or even use them in conjunction.



### Additional Notes on Stemming with NLTK

---

#### NLTK Overview
The **Natural Language Toolkit (NLTK)** is a powerful library in Python for handling various NLP tasks, including both stemming and lemmatization. In this section, we will focus on demonstrating stemming using the NLTK library.

---

### 1. **Installing NLTK**
If you haven't installed NLTK yet, you can easily do so using the following command:

```bash
pip install nltk
```

---

### 2. **Importing Libraries**
To use NLTK for stemming, you'll need to import the necessary classes from the library. Here’s how you can do that:

```python
import nltk
from nltk.stem import PorterStemmer

# Optional: Download the NLTK data (if needed)
nltk.download('punkt')  # Uncomment if needed
```

---

### 3. **Using Porter Stemmer**
The `PorterStemmer` is a popular stemming algorithm provided by NLTK. You can create an object of this class and use it to stem words.

**Code Implementation:**

```python
# Initialize the PorterStemmer
stemmer = PorterStemmer()

# List of words to be stemmed
words = ["talking", "talked", "talks", "eating", "running", "adjustable", "ability"]

# Apply stemming and print results
for word in words:
    stemmed_word = stemmer.stem(word)
    print(f"Original: {word} -> Stemmed: {stemmed_word}")
```

**Expected Output:**
```
Original: talking -> Stemmed: talk
Original: talked -> Stemmed: talk
Original: talks -> Stemmed: talk
Original: eating -> Stemmed: eat
Original: running -> Stemmed: run
Original: adjustable -> Stemmed: adjust
Original: ability -> Stemmed: abil
```

**Explanation:**  
In this example, the `PorterStemmer` processes each word by applying its fixed set of stemming rules. It effectively reduces words like "talking" and "eating" to their roots, while it also demonstrates limitations, as seen with "ability" becoming "abil."

---

### 4. **Using FirstLanguage.in for NLP Tasks**
- **Overview:** The platform simplifies building NLP applications by providing cloud-based solutions for various NLP tasks, eliminating the need for extensive local compute resources.
- **Functionality:** You can perform tasks like text classification without in-depth NLP knowledge by making simple API calls.
- **API Access:** You can sign up for a free tier, obtain your API key, and start making calls to perform tasks like sentiment analysis.

**Example Usage:**
To use the platform, you can submit text for classification:

1. **Negative Review Example:**
   - Input: "This product is terrible."
   - Output: Classified as negative.

2. **Positive Review Example:**
   - Input: "I love this product!"
   - Output: Classified as positive.

---

NLTK offers robust support for stemming with its `PorterStemmer`, making it easy to reduce words to their base forms. While using cloud-based platforms like **firstlanguage.in** can greatly simplify NLP tasks, local libraries like NLTK and spaCy remain essential for custom NLP solutions.


### Customizing Lemmatization in spaCy

---

#### Introduction to Customization

In this section, we'll explore how to customize the lemmatization behavior in spaCy to handle slang or non-standard words, such as "bro," "brah," and "bruh." By default, spaCy may not recognize these terms as synonymous with "brother," but we can modify its behavior using the attribute ruler in the spaCy pipeline.

---

### 1. **Understanding the Pipeline**

The spaCy pipeline consists of several components, including the `tok2vec`, `tagger`, `parser`, and `lemmatizer`. The `attribute ruler` component allows us to customize attributes (like lemmas) for specific tokens based on our needs.

---

### 2. **Customizing Lemmas**

To customize the lemmas for specific words, follow these steps:

**Code Implementation:**

```python
import spacy

# Load the small English language model
nlp = spacy.load("en_core_web_sm")

# Get the attribute ruler component from the pipeline
attribute_ruler = nlp.get_pipe("attribute_ruler")

# Define custom rules for lemmatization
patterns = [
    {"label": "lemma", "pattern": [{"LOWER": "bro"}], "lemma": "brother"},
    {"label": "lemma", "pattern": [{"LOWER": "brah"}], "lemma": "brother"},
    {"label": "lemma", "pattern": [{"LOWER": "bruh"}], "lemma": "brother"}
]

# Add the custom rules to the attribute ruler
attribute_ruler.add_rules(patterns)

# Test the customized lemmatization
test_words = ["bro", "brah", "bruh", "talked", "talking"]

for word in test_words:
    doc = nlp(word)
    for token in doc:
        print(f"Original: {word} -> Custom Lemma: {token.lemma_}")
```

**Expected Output:**
```
Original: bro -> Custom Lemma: brother
Original: brah -> Custom Lemma: brother
Original: bruh -> Custom Lemma: brother
Original: talked -> Custom Lemma: talk
Original: talking -> Custom Lemma: talk
```

**Explanation:**
- The words "bro," "brah," and "bruh" are now correctly lemmatized to "brother" using the custom rules we defined.
- The other words, such as "talked" and "talking," are lemmatized as expected.

---

By utilizing the attribute ruler in spaCy, you can easily customize the lemmatization behavior to handle slang or non-standard language more effectively. This ability to modify the model allows for improved accuracy in NLP applications, especially when dealing with informal language or domain-specific jargon.

---

### Additional Notes

- **Future Exercises:** While there are no exercises included in this video, check the video description for any updates or additional exercises that may be added later.
- **Feedback:** If you enjoy this series or find it helpful, consider giving it a thumbs up and sharing it with friends!
