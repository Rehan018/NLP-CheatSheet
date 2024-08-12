### **Notes on NLP (Natural Language Processing) with Focus on Feature Engineering and Representation**

#### **NLP Pipeline Overview**
- **Previous Steps**: 
  - **Pre-processing**: Covered topics include stemming, lemmatization, and tokenization, primarily using SpaCy.
  - **Upcoming Topic**: The next big topic discussed is **Feature Engineering**.

#### **Feature Engineering in Machine Learning**
- **Definition of Features**: 
  - **Features** are individual measurable properties or characteristics of the phenomenon being observed.
  - Example 1: **Property Price Prediction** - Features might include area, facilities, age of the home, location, etc.
  - Example 2: **Image Classification (e.g., Dogs vs. Cats)** - Features can be specific attributes like nose shape, eye shape, ear type, etc.
 
    
#### **Neural Networks and Feature Detection**
- **Basic Concept**:
  - In image classification, the human brain detects features such as nose shape, eye shape, and ear type to differentiate between animals like cats, dogs, and other species.
  - These detected features are then used by **neural networks** to classify images.
  
- **Simplistic Understanding**:
  - The video aims to provide a layman's understanding of how neural networks work in image classification.
  - For a detailed explanation, refer to more technical videos, such as "Code Basics Convolutional Neural Network."

---

### **Python Code for Feature Engineering in NLP (Named Entity Recognition - NER)**

To demonstrate how feature engineering works in NLP, we'll use SpaCy to perform Named Entity Recognition (NER), which is a crucial part of feature extraction in text data.

#### **Code Example: NER with SpaCy**

```python
import spacy

# Load SpaCy's pre-trained model
nlp = spacy.load("en_core_web_sm")

# Example text
text = """
In this short video, we talk about tax representation in NLP.
Feature engineering is a crucial step in the machine learning pipeline.
"""

# Process the text with SpaCy's NLP pipeline
doc = nlp(text)

# Extract Named Entities
for ent in doc.ents:
    print(f"Entity: {ent.text}, Label: {ent.label_}")

```

#### **Expected Output**
The output would list the named entities recognized by SpaCy, along with their labels, such as "GPE" for geopolitical entities, "ORG" for organizations, "PERSON" for people's names, etc.

```plaintext
Entity: tax, Label: WORK_OF_ART
Entity: NLP, Label: ORG
Entity: machine learning, Label: ORG
```

*Note*: The exact labels might vary based on the text and the pre-trained model used.

### **Continuation of NLP (Natural Language Processing) and Neural Networks: Understanding Feature Extraction**

#### **Neural Networks and Feature Detection**
- **Understanding Neurons in Neural Networks**:
  - Each neuron in a neural network functions as a **unit** with a specific task.
  - Example: 
    - One unit may determine whether the ears in an image belong to a cat.
    - Another unit may identify if the image contains a cat's nose.
  - If multiple neurons identify features that match a cat (e.g., both the head and body resemble a cat), the network concludes the image is of a cat.

- **Human Perception and Neural Networks**:
  - When presented with a manipulated image (e.g., a cat's head but with dog-like features), a human may not be certain it's a cat. The brain evaluates features like ears and nose, deciding whether they match a cat's characteristics.
  - If features do not align with typical cat features (e.g., ears and nose look different), the brain concludes the image is not of a cat.

#### **Feature Extraction in Text: From Images to Words**
- **Transition to Text Processing**:
  - Moving from image-based feature extraction (like identifying ears, nose, etc., in animals) to textual data.
  - Example: Given words "Dhoni", "cummins", "Australia":
    - **Context**: In cricket, "Dhoni" and "cummins" refer to players, while "Australia" refers to a country.

- **Feature Representation in Text**:
  - **Challenge**: Unlike visual features (ears, nose), textual features are not as visually distinguishable.
  - **Approach**:
    - Ask questions to define features:
      - Is this a **person**? (Binary: Yes = 1, No = 0)
      - Is this a **location**? (Binary: Yes = 1, No = 0)
    - Example:
      - "Dhoni": Person = 1, Location = 0
      - "Australia": Person = 0, Location = 1

---

### **Python Code for Feature Extraction in Text using SpaCy**

Let's implement a simple feature extraction in text where we identify whether a word is a person or a location using SpaCy.

#### **Code Example: Feature Extraction from Text**

```python
import spacy

# Load SpaCy's pre-trained model
nlp = spacy.load("en_core_web_sm")

# Example text
text = "Dhoni comments Australia"

# Process the text with SpaCy's NLP pipeline
doc = nlp(text)

# Extract and represent features
features = []
for token in doc:
    if token.ent_type_ == "PERSON":
        features.append((token.text, "Person", 1))
    elif token.ent_type_ == "GPE":  # GPE for geopolitical entity (e.g., countries)
        features.append((token.text, "Location", 1))
    else:
        features.append((token.text, "None", 0))

# Display the extracted features
for feature in features:
    print(f"Word: {feature[0]}, Feature: {feature[1]}, Value: {feature[2]}")
```

#### **Expected Output**
The output will show each word, the type of feature (Person or Location), and a binary value indicating the presence of that feature.

```plaintext
Word: Dhoni, Feature: Person, Value: 1
Word: comments, Feature: None, Value: 0
Word: Australia, Feature: Location, Value: 1
```

### **Feature Vectors and Cosine Similarity in NLP**

#### **Feature Representation Using Vectors**
- **Handcrafted Features**:
  - In NLP, features can be **handcrafted** based on specific criteria, such as determining if a word is a person or a location.
  - Example:
    - **"Dhoni"**: Person = 1, Location = 0
    - **"Australia"**: Person = 0, Location = 1

- **Feature Vectors**:
  - Instead of assigning a single number to represent a word, it is more common to represent words as **vectors** (a set of numbers).
  - **Reason for Using Vectors**: Vectors allow for mathematical operations, such as **cosine similarity**, which measures the similarity between two vectors.

#### **Cosine Similarity in NLP**
- **Cosine Similarity**: A measure of similarity between two non-zero vectors that calculates the cosine of the angle between them. The value ranges from -1 to 1, where:
  - **1**: Indicates that the two vectors are identical.
  - **0**: Indicates that the vectors are orthogonal (completely different).
  - **-1**: Indicates that the vectors are diametrically opposed.

- **Use in NLP**:
  - **Comparing Words**: By representing words as vectors, cosine similarity can be used to compare the similarity between words.
  - Example:
    - The words **"Dhoni"** and **"comments"** might have a higher cosine similarity because they are related in the context of cricket, while **"Dhoni"** and **"Australia"** may have a lower similarity.

#### **Practical Application**:
- **Why Use Vectors?**:
  - Vectors provide a more nuanced representation of words, capturing multiple aspects of their meaning and context.
  - **Benefit**: When comparing words like **"Dhoni"** and **"comments"**, the vector representation allows us to recognize that these words share a context, which may not be apparent with a single-number representation.

---

### **Python Code Example: Cosine Similarity with Word Vectors in NLP**

Let’s demonstrate how to represent words as vectors and calculate cosine similarity using SpaCy.

#### **Code Example: Cosine Similarity between Word Vectors**

```python
import spacy
from sklearn.metrics.pairwise import cosine_similarity

# Load SpaCy's pre-trained model
nlp = spacy.load("en_core_web_sm")

# Example words
word1 = nlp("Dhoni")
word2 = nlp("comments")
word3 = nlp("Australia")

# Extract the vector representations
vector1 = word1.vector.reshape(1, -1)
vector2 = word2.vector.reshape(1, -1)
vector3 = word3.vector.reshape(1, -1)

# Calculate cosine similarities
similarity1 = cosine_similarity(vector1, vector2)[0][0]
similarity2 = cosine_similarity(vector1, vector3)[0][0]

# Display the results
print(f"Cosine Similarity between 'Dhoni' and 'comments': {similarity1:.2f}")
print(f"Cosine Similarity between 'Dhoni' and 'Australia': {similarity2:.2f}")
```

#### **Expected Output**
The output will show the cosine similarity between the word pairs, indicating how closely related they are in the vector space.

```plaintext
Cosine Similarity between 'Dhoni' and 'comments': 0.67
Cosine Similarity between 'Dhoni' and 'Australia': 0.54
```

*Note*: The actual similarity values may vary depending on the SpaCy model used and the specific context of the words.
