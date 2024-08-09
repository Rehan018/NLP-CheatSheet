### Notes on Part of Speech (POS) Tagging with spaCy

#### Introduction to Part of Speech (POS)
Part of Speech (POS) tagging is a fundamental concept in Natural Language Processing (NLP) where each word in a sentence is assigned a part of speech, such as noun, verb, adjective, etc. Understanding POS tagging helps in analyzing the structure of sentences and is crucial for many NLP applications.

##### Brief Overview of Basic POS
- **Noun:** A person, place, thing, or idea (e.g., "Elon", "Mars", "fruits").
- **Verb:** An action (e.g., "eating", "play").
- **Pronoun:** A word that substitutes a noun (e.g., "he", "she", "they").
- **Adjective:** A word that describes a noun (e.g., "sweet", "many", "red").

##### Example Sentences:
1. "Elon would have been to Mars."
   - **Nouns:** Elon, Mars
   - **Verb:** would have been

2. "I ate many fruits."
   - **Pronoun:** I
   - **Verb:** ate
   - **Adjective:** many
   - **Noun:** fruits

#### Implementing POS Tagging with spaCy

To implement POS tagging, we can use the spaCy library, which provides a pre-trained pipeline for tagging parts of speech in sentences.

##### 1. **Installing spaCy:**
   First, you need to install spaCy and download the English language model:
   ```bash
   pip install spacy
   python -m spacy download en_core_web_sm
   ```

##### 2. **Loading the spaCy Model:**
   Load the pre-trained model in spaCy:
   ```python
   import spacy

   # Load the pre-trained English model
   nlp = spacy.load("en_core_web_sm")
   ```

##### 3. **Processing a Sentence:**
   Now, you can process a sentence and extract the POS tags:
   ```python
   # Example sentence
   sentence = "Elon would have been to Mars."

   # Process the sentence
   doc = nlp(sentence)

   # Print each token and its part of speech
   for token in doc:
       print(f"{token.text}: {token.pos_}")
   ```

##### 4. **Output Explanation:**
   When you run the code above, you will get an output similar to this:
   ```
   Elon: PROPN
   would: AUX
   have: AUX
   been: VERB
   to: ADP
   Mars: PROPN
   .: PUNCT
   ```

   - **PROPN:** Proper Noun (e.g., Elon, Mars)
   - **AUX:** Auxiliary Verb (e.g., would, have)
   - **VERB:** Verb (e.g., been)
   - **ADP:** Adposition (e.g., to)
   - **PUNCT:** Punctuation (e.g., .)

##### 5. **Tagging More Sentences:**
   You can process additional sentences to see how POS tagging works with different sentence structures:
   ```python
   sentences = [
       "I ate many fruits.",
       "He is playing cricket.",
       "The quick brown fox jumps over the lazy dog."
   ]

   for sentence in sentences:
       doc = nlp(sentence)
       print(f"\nSentence: {sentence}")
       for token in doc:
           print(f"{token.text}: {token.pos_}")
   ```

##### 6. **Using POS Tags in NLP Applications:**
   POS tagging is essential for many NLP tasks such as parsing, named entity recognition, and sentiment analysis. By understanding the role of each word in a sentence, you can better analyze text and make informed decisions in your NLP applications.

##### 7. **Firstlanguage.in Platform:**
   Firstlanguage.in simplifies the process of building NLP applications by allowing you to perform various tasks via an HTTP call. Unlike local setups where you may need higher compute resources, this platform provides cloud-based solutions, making it easier to implement and scale NLP models.


#### Additional Part of Speech (POS) Concepts

##### 1. **Adverbs:**
Adverbs add more meaning to a verb, adjective, or even another adverb. They describe how an action is performed or modify the meaning of adjectives or other adverbs.

- **Example Sentences:**
  - "I slowly ate many fruits."
    - **slowly:** Adverb (modifies the verb "ate")
  - "The dog quickly ran."
    - **quickly:** Adverb (modifies the verb "ran")
  - "Maria always scores 90 percent on her exams."
    - **always:** Adverb (modifies the verb "scores")

##### 2. **Using spaCy to Identify Adverbs:**
Let’s enhance our POS tagging example by identifying adverbs in sentences.

```python
import spacy

# Load the pre-trained English model
nlp = spacy.load("en_core_web_sm")

# Example sentences containing adverbs
sentences = [
    "I slowly ate many fruits.",
    "The dog quickly ran.",
    "Maria always scores 90 percent on her exams."
]

for sentence in sentences:
    doc = nlp(sentence)
    print(f"\nSentence: {sentence}")
    for token in doc:
        print(f"{token.text}: {token.pos_}")
```

##### 3. **Output Explanation:**
Running the above code will output something like:
```
Sentence: I slowly ate many fruits.
I: PRON
slowly: ADV
ate: VERB
many: ADJ
fruits: NOUN

Sentence: The dog quickly ran.
The: DET
dog: NOUN
quickly: ADV
ran: VERB

Sentence: Maria always scores 90 percent on her exams.
Maria: PROPN
always: ADV
scores: VERB
90: NUM
percent: NOUN
on: ADP
her: DET
exams: NOUN
```

- **ADV:** Adverb (e.g., slowly, quickly, always)
- **DET:** Determiner (e.g., The, her)
- **NUM:** Numeral (e.g., 90)

#### Cloud-Based NLP Solutions with Firstlanguage.in

##### 1. **Cloud-Based NLP:**
Firstlanguage.in allows you to perform NLP tasks like text classification directly on the cloud, eliminating the need for local computational resources.

##### 2. **Text Classification Example:**
Firstlanguage.in provides a straightforward API for text classification, where you can classify text as positive, negative, or other categories based on the content.

- **Negative Review Example:**
  - You paste a negative review from Amazon, and the API immediately classifies it as negative.

- **Positive Review Example:**
  - You paste a positive review, and it’s classified as positive.

##### 3. **Getting Started with Firstlanguage.in:**
- **API Key Access:**
  - Sign up on the platform to access the free tier.
  - Once signed in, navigate to the dashboard to obtain your API key.
  - Use this API key to make API calls.

- **SDKs:**
  - SDKs are available in Python and TypeScript, making it easier to integrate with your applications.

##### 4. **Example API Call (Python):**

Here’s a basic example of making an API call using the Python SDK:

```python
import requests

# Replace with your API key
api_key = "your_api_key"

# Sample text for classification
text = "This product is amazing and works perfectly!"

# Endpoint for text classification
url = "https://api.firstlanguage.in/api/v1/classify"

# Headers and data payload
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
data = {
    "text": text
}

# Make the POST request
response = requests.post(url, json=data, headers=headers)

# Print the classification result
print(response.json())
```

##### 5. **Understanding Adverbs in spaCy:**
Just as adjectives add meaning to a noun, adverbs provide context to verbs or entire actions. With spaCy, identifying adverbs can enhance your understanding of how actions are described in text, making it a powerful tool for deeper linguistic analysis.


### Final Notes on Part of Speech (POS) Tagging with spaCy

#### Additional Parts of Speech (POS)

##### 1. **Interjections:**
Interjections are words or phrases that express strong emotions or sudden feelings. They often stand alone and are followed by an exclamation mark.

- **Example Sentences:**
  - "Wow! Doctor Strange made 265 million dollars."
  - "Alas, he got a no from the Google interview."
  - "Hey, give me back my Lamborghini!"

These words ("wow," "alas," "hey") convey strong emotions and are known as interjections.

##### 2. **Conjunctions:**
Conjunctions are words that connect clauses, sentences, or words. They help in forming more complex sentences by linking different ideas.

- **Example Sentences:**
  - "I want to eat pizza, but I want to be healthy."
  - "Hulk took a pen and started writing a story."
  - "You either give me a job, or I will leave."

In these examples, the words "but," "and," and "or" are conjunctions that connect different groups of words or clauses.

##### 3. **Prepositions:**
Prepositions are words that link nouns, pronouns, or phrases to other words within a sentence. They usually indicate location, direction, time, or the relationship between two entities.

- **Example Sentences:**
  - "Thor is on the bus."
  - "Thor is in the bus."
  - "Thor is at the bus."

The words "on," "in," and "at" are prepositions, and they can change the meaning of the sentence by indicating different relationships between "Thor" and "the bus."

#### Implementing Additional POS Tagging with spaCy

##### 1. **Using spaCy to Identify Interjections, Conjunctions, and Prepositions:**
Let’s extend our POS tagging example to include interjections, conjunctions, and prepositions.

```python
import spacy

# Load the pre-trained English model
nlp = spacy.load("en_core_web_sm")

# Example sentences containing various parts of speech
sentences = [
    "Wow! Doctor Strange made 265 million dollars.",
    "I want to eat pizza, but I want to be healthy.",
    "Thor is on the bus."
]

for sentence in sentences:
    doc = nlp(sentence)
    print(f"\nSentence: {sentence}")
    for token in doc:
        print(f"{token.text}: {token.pos_}")
```

##### 2. **Output Explanation:**
Running the above code will output something like:
```
Sentence: Wow! Doctor Strange made 265 million dollars.
Wow: INTJ
!: PUNCT
Doctor: PROPN
Strange: PROPN
made: VERB
265: NUM
million: NUM
dollars: NOUN
.: PUNCT

Sentence: I want to eat pizza, but I want to be healthy.
I: PRON
want: VERB
to: PART
eat: VERB
pizza: NOUN
,: PUNCT
but: CCONJ
I: PRON
want: VERB
to: PART
be: VERB
healthy: ADJ
.: PUNCT

Sentence: Thor is on the bus.
Thor: PROPN
is: AUX
on: ADP
the: DET
bus: NOUN
.: PUNCT
```

- **INTJ:** Interjection (e.g., Wow)
- **CCONJ:** Coordinating Conjunction (e.g., but)
- **ADP:** Adposition/Preposition (e.g., on)

#### Summary of Key POS Concepts

- **Nouns** represent people, places, things, or ideas.
- **Verbs** indicate actions.
- **Adjectives** describe nouns.
- **Adverbs** describe verbs, adjectives, or other adverbs.
- **Pronouns** replace nouns.
- **Interjections** express strong emotions.
- **Conjunctions** connect words, phrases, or clauses.
- **Prepositions** link nouns, pronouns, or phrases to other words.

#### Practical Application with spaCy
Using spaCy, you can easily identify and analyze the different parts of speech in a sentence. This allows for a deeper understanding of sentence structure, which is critical for tasks like text classification, sentiment analysis, and other NLP applications.

#### Exploring Cloud-Based NLP Solutions
As previously mentioned, platforms like Firstlanguage.in provide cloud-based NLP services, allowing you to perform complex NLP tasks without requiring extensive local resources. Whether you're classifying text, detecting sentiments, or identifying parts of speech, these platforms make NLP accessible and scalable.

### Extended Notes on Part of Speech (POS) Tagging with spaCy in a Jupyter Notebook

In this section, we’ll go over the practical implementation of Part of Speech (POS) tagging using spaCy in a Jupyter Notebook. We’ll explore how to load the spaCy model, process a text, and extract POS information for each word (token) in the text.

#### 1. **Introduction to Prepositions:**
Prepositions are connectors that link a noun or pronoun with another word in a sentence, indicating relationships of time, place, direction, etc.

- **Example Sentence:** 
  - "Thor is on the bus."
  - "The cat is under the table."

In the examples, "on" and "under" are prepositions, linking the noun "Thor" and "the bus," or "the cat" and "the table," respectively.

#### 2. **Exploring Prepositions Further:**
If you search online for preposition examples, you’ll find extensive lists, including common ones like "in," "on," "at," "by," "under," "over," etc. Understanding these is essential in grasping sentence structure.

#### 3. **Launching a Jupyter Notebook for POS Tagging:**
To get started with POS tagging using spaCy in a Jupyter Notebook, follow these steps:

1. **Launch Jupyter Notebook:**
   - Open your terminal or command prompt and run:
     ```bash
     jupyter notebook
     ```
   - This will launch a Jupyter Notebook in your web browser.

2. **Create a New Python Notebook:**
   - In the Jupyter interface, create a new Python notebook by selecting "New" -> "Python 3."

3. **Importing spaCy and Loading the Model:**
   - In your new notebook, import spaCy and load a pre-trained English model:
     ```python
     import spacy

     # Load the pre-trained English model
     nlp = spacy.load("en_core_web_sm")
     ```

4. **Creating a Document with spaCy:**
   - Write a sentence for processing:
     ```python
     # Example sentence
     sentence = "Elon Musk is planning to launch a rocket to Mars."

     # Process the sentence using spaCy
     doc = nlp(sentence)
     ```

5. **Printing Tokens and Their POS Tags:**
   - Iterate through the tokens in the processed document and print their text along with their POS tags:
     ```python
     # Iterate through tokens and print text and POS tags
     for token in doc:
         print(f"{token.text}: {token.pos_}")
     ```

   - **Expected Output:**
     ```plaintext
     Elon: PROPN
     Musk: PROPN
     is: AUX
     planning: VERB
     to: PART
     launch: VERB
     a: DET
     rocket: NOUN
     to: ADP
     Mars: PROPN
     .: PUNCT
     ```

   - In the output:
     - **PROPN** stands for Proper Noun.
     - **AUX** is Auxiliary (helping) Verb.
     - **VERB** represents the action.
     - **DET** is a Determiner (e.g., "a").
     - **ADP** is Adposition/Preposition (e.g., "to").

6. **Understanding POS Tags:**
   - **`token.pos_`** provides the POS tag as a string (e.g., "NOUN," "VERB").
   - **`token.pos`** returns the POS tag as an integer value.

#### 4. **Exploring More Examples:**
You can experiment with more sentences to see how different parts of speech are tagged:

```python
# Another example sentence
sentence_2 = "She quickly learned NLP and became an expert."

# Process the sentence
doc_2 = nlp(sentence_2)

# Print tokens with POS tags
for token in doc_2:
    print(f"{token.text}: {token.pos_}")
```

- **Expected Output:**
  ```plaintext
  She: PRON
  quickly: ADV
  learned: VERB
  NLP: PROPN
  and: CCONJ
  became: VERB
  an: DET
  expert: NOUN
  .: PUNCT
  ```

#### 5. **Summary and Further Learning:**
In this notebook, we have explored how to perform POS tagging using spaCy, including identifying common parts of speech such as nouns, verbs, adjectives, adverbs, prepositions, conjunctions, and interjections. This understanding is crucial for NLP tasks, and you can deepen your knowledge by exploring further resources on English grammar and POS tagging.


### Additional Notes on POS Tagging with spaCy

In this section, we’ll dive deeper into the POS tagging process using spaCy, exploring how to get more detailed explanations for the POS tags and understanding the differences between fundamental grammar categories and what spaCy identifies.

#### 1. **Understanding Proper Nouns and Common Nouns:**
- **Noun:** A word that represents a person, place, thing, or idea.
  - **Example:** "person," "city," "dog."
- **Proper Noun:** A specific name of a person, place, or thing.
  - **Example:** "Elon," "New York," "Tesla."

In spaCy, a proper noun is tagged as `PROPN`, while a common noun is tagged as `NOUN`.

#### 2. **Fundamental Parts of Speech vs. spaCy POS Tags:**
- The traditional 8 parts of speech in English grammar include nouns, pronouns, verbs, adjectives, adverbs, prepositions, conjunctions, and interjections.
- However, spaCy's POS tagging system is more granular, adding categories like determiners, numerals, and articles.

#### 3. **Exploring POS Tags with spaCy:**
To understand the detailed POS tags provided by spaCy, we can use `spacy.explain` to get explanations for each tag.

#### 4. **Code Implementation:**
Here’s how you can use spaCy to print the POS tags along with their explanations:

1. **Import spaCy and Load the Model:**
   ```python
   import spacy

   # Load the pre-trained English model
   nlp = spacy.load("en_core_web_sm")
   ```

2. **Process a Sentence and Extract POS Tags:**
   ```python
   # Example sentence
   sentence = "Elon Musk is planning to launch a rocket to Mars."

   # Process the sentence using spaCy
   doc = nlp(sentence)

   # Print tokens with their POS tags and explanations
   for token in doc:
       print(f"Token: {token.text}, POS: {token.pos_}, Explanation: {spacy.explain(token.pos_)}")
   ```

   - **Expected Output:**
     ```plaintext
     Token: Elon, POS: PROPN, Explanation: proper noun
     Token: Musk, POS: PROPN, Explanation: proper noun
     Token: is, POS: AUX, Explanation: auxiliary
     Token: planning, POS: VERB, Explanation: verb
     Token: to, POS: PART, Explanation: particle
     Token: launch, POS: VERB, Explanation: verb
     Token: a, POS: DET, Explanation: determiner
     Token: rocket, POS: NOUN, Explanation: noun
     Token: to, POS: ADP, Explanation: adposition
     Token: Mars, POS: PROPN, Explanation: proper noun
     Token: ., POS: PUNCT, Explanation: punctuation
     ```

3. **Using spaCy’s `explain` Function for Detailed Tag Explanations:**
   ```python
   # Example of using spacy.explain
   pos_tag = "PROPN"
   explanation = spacy.explain(pos_tag)
   print(f"POS Tag: {pos_tag}, Explanation: {explanation}")
   ```

   - **Expected Output:**
     ```plaintext
     POS Tag: PROPN, Explanation: proper noun
     ```

#### 5. **Difference Between spaCy POS Tags and Traditional Grammar:**
spaCy provides a more detailed POS tagging system that includes categories beyond the traditional 8 parts of speech. For example:
- **Numeral (`NUM`)**: Represents numbers or numerical expressions.
- **Determiner (`DET`)**: Words that introduce nouns, such as "a," "an," "the."
- **Particle (`PART`)**: Words that do not fit neatly into the standard categories but modify the meaning of other words, such as "to" in "to run."

Using spaCy, you can gain a deeper understanding of POS tagging by not only identifying the grammatical role of each word in a sentence but also by exploring the nuances that modern NLP systems like spaCy introduce. The `spacy.explain` function is a valuable tool for decoding these POS tags into more understandable terms. This approach helps bridge the gap between traditional grammar education and advanced NLP techniques.

### Advanced POS Tagging and Tense Identification with spaCy

In this section, we’ll explore how spaCy can be used to perform more advanced POS tagging, including identifying verb tenses and subcategories beyond the basic parts of speech.

#### 1. **Pipeline Components in spaCy:**
- When you load a trained model in spaCy, it automatically provides a pipeline that includes components like the tagger, parser, and NER (Named Entity Recognition).
- These components work together to process text, automatically tagging words with their part of speech, parsing sentence structure, and identifying entities.

#### 2. **Prepositions vs. Adpositions:**
- In traditional grammar, we talk about prepositions (e.g., "on," "in," "at") which link nouns to other words.
- spaCy uses a broader category called "adpositions," which includes prepositions, postpositions, and circumpositions.
  - **Preposition:** Appears before the noun (e.g., "on the table").
  - **Postposition:** Appears after the noun (less common in English).
  - **Circumposition:** Surrounds the noun (rare in English).

#### 3. **Detailed POS Tagging with spaCy:**
- spaCy’s POS tags provide basic grammatical information, but its `tag_` attribute gives even more detail, such as verb tense.
- **Example:**
  ```python
  for token in doc:
      print(f"Token: {token.text}, POS: {token.pos_}, Tag: {token.tag_}, Explanation: {spacy.explain(token.tag_)}")
  ```

- **Output Example:**
  ```plaintext
  Token: made, POS: VERB, Tag: VBD, Explanation: verb, past tense
  ```

#### 4. **Verb Tenses with spaCy:**
- **Present Tense:** spaCy identifies present tense verbs (e.g., "quits") using tags like `VBZ`, which stands for "Verb, 3rd person singular present."
- **Past Tense:** For past tense verbs (e.g., "quit"), spaCy uses tags like `VBD`, which stands for "Verb, past tense."
  - **Example:**
    ```python
    sentence = "He quits the job."
    doc = nlp(sentence)
    for token in doc:
        print(f"Token: {token.text}, POS: {token.pos_}, Tag: {token.tag_}, Explanation: {spacy.explain(token.tag_)}")
    ```

    - **Expected Output:**
      ```plaintext
      Token: quits, POS: VERB, Tag: VBZ, Explanation: verb, 3rd person singular present
      ```

- **Changing Tenses:**
  - If you modify the sentence to "He quit the job," spaCy will correctly identify "quit" as a past tense verb.
    ```python
    sentence = "He quit the job."
    doc = nlp(sentence)
    for token in doc:
        print(f"Token: {token.text}, POS: {token.pos_}, Tag: {token.tag_}, Explanation: {spacy.explain(token.tag_)}")
    ```

    - **Expected Output:**
      ```plaintext
      Token: quit, POS: VERB, Tag: VBD, Explanation: verb, past tense
      ```

#### 5. **Subcategories and Nuances in spaCy:**
- **Tags vs. POS:** The `pos_` attribute provides the main part of speech, while `tag_` gives more granular details.
- **Examples of Tags:**
  - **VBZ:** Verb, 3rd person singular present.
  - **VBD:** Verb, past tense.
  - **CD:** Cardinal number (e.g., "two").
  - **.**: Sentence-ending punctuation.
  
- **Utility in NLP Applications:**
  - Understanding these details is crucial for building sophisticated NLP applications. For example, knowing whether a verb is in past or present tense can affect how an NLP model interprets or generates text.

#### 6. **Practical Example - Analyzing Text:**
- Let’s analyze a corporate earnings report or any other formal document. The goal is to understand the distribution of tenses, nouns, verbs, and other parts of speech, which can be useful for tasks like sentiment analysis or document summarization.

- **Code Example:**
  ```python
  # Example document (replace with actual text from a report)
  doc = nlp("Microsoft reported a revenue increase of 15% last quarter.")
  
  # Analyze POS tags and other attributes
  for token in doc:
      print(f"Token: {token.text}, POS: {token.pos_}, Tag: {token.tag_}, Explanation: {spacy.explain(token.tag_)}")
  ```

- **Expected Output:**
  ```plaintext
  Token: Microsoft, POS: PROPN, Tag: NNP, Explanation: proper noun, singular
  Token: reported, POS: VERB, Tag: VBD, Explanation: verb, past tense
  Token: a, POS: DET, Tag: DT, Explanation: determiner
  Token: revenue, POS: NOUN, Tag: NN, Explanation: noun, singular or mass
  Token: increase, POS: NOUN, Tag: NN, Explanation: noun, singular or mass
  Token: of, POS: ADP, Tag: IN, Explanation: conjunction, subordinating or preposition
  Token: 15%, POS: NUM, Tag: CD, Explanation: cardinal number
  Token: last, POS: ADJ, Tag: JJ, Explanation: adjective
  Token: quarter, POS: NOUN, Tag: NN, Explanation: noun, singular or mass
  Token: ., POS: PUNCT, Tag: ., Explanation: punctuation mark, sentence closer
  ```

Understanding and using advanced POS tagging with spaCy allows for more precise text analysis and can significantly enhance the capabilities of NLP applications. Whether you’re categorizing text, detecting sentiment, or analyzing document structure, the detailed tags provided by spaCy are invaluable tools.

### Using POS Tagging in Real-World Applications

POS (Part of Speech) tagging is more than just a linguistic exercise; it has practical applications in real-world scenarios. Let’s explore how POS tagging can be utilized in tasks such as filtering text, counting parts of speech, and extracting valuable insights from a document.

#### 1. **Filtering Text:**
   - When working with text data, you often need to clean up unnecessary tokens like punctuation and other non-essential elements.
   - You can filter tokens using spaCy, only keeping the relevant ones, such as nouns and verbs.

   **Example:**
   ```python
   filter_tokens = []
   for token in doc:
       if not token.is_punct:  # Filtering out punctuation
           filter_tokens.append(token.text)
   print(filter_tokens)
   ```

   - **Output:**
     This will provide a list of tokens with all the unnecessary punctuation removed.

#### 2. **Counting Parts of Speech:**
   - Knowing how many nouns, verbs, or other POS categories are present in a text can be valuable for text analysis.
   - spaCy provides a convenient method called `count_by()` to count the occurrence of different POS tags.

   **Example:**
   ```python
   from spacy.attrs import POS

   # Counting the parts of speech
   pos_counts = doc.count_by(POS)
   for pos, count in pos_counts.items():
       print(f"{doc.vocab[pos].text}: {count}")
   ```

   - **Output:**
     This will display the count of each part of speech in the text, such as how many nouns, verbs, etc., are present.

#### 3. **Understanding POS Tag Codes:**
   - The `count_by()` method returns POS tags as numbers. To understand what these numbers represent, you can look them up in spaCy’s vocabulary.

   **Example:**
   ```python
   # Example to decode the POS tag
   for pos, count in pos_counts.items():
       print(f"{doc.vocab[pos].text}: {count}")
   ```

   - **Output:**
     This will print something like:
     ```plaintext
     PROPN: 13
     NOUN: 48
     PUNCT: 10
     ```
     This indicates the number of proper nouns, common nouns, and punctuation marks in the text.

#### 4. **Applying POS Tagging to a Business Use Case:**
   - Consider a real-world scenario like analyzing a corporate earnings report or a news article about inflation. Understanding the distribution of different parts of speech can provide insights into the focus and tone of the document.

   **Example:**
   ```python
   # Loading the text of a report or article
   doc = nlp("Microsoft reported a revenue increase of 15% last quarter. Inflation is rising in several sectors.")

   # Counting parts of speech
   pos_counts = doc.count_by(POS)
   for pos, count in pos_counts.items():
       print(f"{doc.vocab[pos].text}: {count}")
   ```

   - **Expected Output:**
     ```plaintext
     PROPN: 2
     NOUN: 5
     VERB: 3
     NUM: 1
     ```
     This gives a quick overview of the types of words used in the text.

#### 5. **Exercise: Analyze a News Article**
   - As an exercise, grab a news article from a reputable source, such as CNBC.com, and apply the POS tagging techniques you’ve learned. For instance, analyze a story about inflation and see how the language is structured. Count the nouns, verbs, and other parts of speech to understand the article’s focus.

   **Example Workflow:**
   1. Load the article text.
   2. Filter out irrelevant tokens.
   3. Count the parts of speech.
   4. Interpret the results to gain insights.

   **Tip:** Always check the video description or accompanying notes in a tutorial for links to exercises or additional resources.
