# Tokenization in spaCy

here we discuss tokenization using the spaCy library. While NLTK also offers tokenization, we've chosen spaCy for its advantages discussed in a previous video.

## Introduction to Tokenization

Tokenization is a key step in the NLP pipeline, particularly in pre-processing. This involves breaking down text into sentences and words. Here's a brief overview:

- **Sentence Tokenization**: Splitting a paragraph into individual sentences.
- **Word Tokenization**: Splitting each sentence into individual words.

## Why Not Use Simple Rules?

Using simple rules like splitting by spaces or dots can lead to errors. For example:

- "Dr." in "Dr. Smith" or "n.y." for New York are not sentence end markers.
  
To handle such cases, we need a library like spaCy, which incorporates language-specific rules.

## Getting Started with spaCy

### Installation

To install spaCy, run:
```bash
pip install spacy
```

### Setting Up

1. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
   Navigate to the directory where you want to work.

2. **Create a spaCy Tokenization Directory**:
   Create your desired directory, navigate to it, and run:
   ```bash
   python3
   ```

### Basic Usage

1. **Import spaCy**:
   ```python
   import spacy
   ```

2. **Create a Language Object**:
   ```python
   nlp = spacy.blank('en')  # 'en' for English, 'de' for German, etc.
   ```

3. **Create a Document**:
   ```python
   doc = nlp("Doctor Strange's visit in Mumbai. He loved pav bhaji so much.")
   ```

The language object (`nlp`) can be created in multiple ways, such as using `spacy.blank('en')` for English or other codes like 'de' for German. You can find all available language models via `spaCy language models`.

### Additional Information

- **Pre-trained Pipelines**: In future videos, we'll discuss creating pre-trained language pipelines.
- **Language Codes**:
  - English: `en`
  - French: `fr`
  - German: `de`
  - Hindi: `hi`


here we discuss tokenization using the spaCy library. While NLTK also offers tokenization, we've chosen spaCy for its advantages discussed in a previous video.

## Introduction to Tokenization

Tokenization is a key step in the NLP pipeline, particularly in pre-processing. This involves breaking down text into sentences and words. Here's a brief overview:

- **Sentence Tokenization**: Splitting a paragraph into individual sentences.
- **Word Tokenization**: Splitting each sentence into individual words.

## Why Not Use Simple Rules?

Using simple rules like splitting by spaces or dots can lead to errors. For example:

- "Dr." in "Dr. Smith" or "n.y." for New York are not sentence end markers.
  
To handle such cases, we need a library like spaCy, which incorporates language-specific rules.

## Getting Started with spaCy

### Installation

To install spaCy, run:
```bash
pip install spacy
```

### Setting Up

1. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
   Navigate to the directory where you want to work.

2. **Create a spaCy Tokenization Directory**:
   Create your desired directory, navigate to it, and run:
   ```bash
   python3
   ```

### Basic Usage

1. **Import spaCy**:
   ```python
   import spacy
   ```

2. **Create a Language Object**:
   ```python
   nlp = spacy.blank('en')  # 'en' for English, 'de' for German, etc.
   ```

3. **Create a Document**:
   ```python
   doc = nlp("Doctor Strange's visit in Mumbai. He loved pav bhaji so much.")
   ```

The language object (`nlp`) can be created in multiple ways, such as using `spacy.blank('en')` for English or other codes like 'de' for German. You can find all available language models via `spaCy language models`.

### Tokenization Example

1. **Tokenizing Text**:
   ```python
   for token in doc:
       print(token)
   ```
   This will tokenize the text into words, and you can access each token individually.

2. **Using Python List-Style Indexing**:
   ```python
   print(doc[0])  # Output: Dr.
   print(doc[1])  # Output: Strange
   ```

### Advanced Tokenization

spaCy's tokenizer handles various complexities, such as:

- **Splitting Currency**:
  ```python
  doc = nlp("I have $200.")
  for token in doc:
      print(token)
  ```
  This will split "$200" into two tokens: `$` and `200`.

- **Handling Punctuation**:
  ```python
  doc = nlp("Let's go to N.Y.!")
  for token in doc:
      print(token)
  ```
  This will properly tokenize "N.Y." and handle punctuation correctly.

## Additional Information

- **Pre-trained Pipelines**: In future videos, we'll discuss creating pre-trained language pipelines.
- **Language Codes**:
  - English: `en`
  - French: `fr`
  - German: `de`
  - Hindi: `hi`

## Cloud-Based NLP

For heavy NLP tasks requiring substantial compute resources, consider using cloud-based solutions. These services allow you to make HTTP calls and perform tasks on the cloud without needing local compute resources.

### Example Service

A cloud-based NLP service allows for text classification and other tasks via API calls. You can:

1. **Sign Up**: Access the free tier by signing up and obtaining an API key from the dashboard.
2. **SDKs**: Use available SDKs in Python or TypeScript.
3. **Text Classification Example**:
   - **Negative Review**:
     ```python
     # API call to classify text
     ```
   - **Positive Review**:
     ```python
     # API call to classify text
     ```

This simplifies the process and makes NLP tasks accessible without deep technical knowledge.


Tokenization is a fundamental NLP task, and spaCy provides powerful tools to handle it efficiently. Whether working locally or using cloud-based services, spaCy and similar tools offer flexibility and ease of use for a variety of NLP applications.


In this video, we discuss tokenization using the spaCy library. While NLTK also offers tokenization, we've chosen spaCy for its advantages discussed in a previous video.

## Introduction to Tokenization

Tokenization is a key step in the NLP pipeline, particularly in pre-processing. This involves breaking down text into sentences and words. Here's a brief overview:

- **Sentence Tokenization**: Splitting a paragraph into individual sentences.
- **Word Tokenization**: Splitting each sentence into individual words.

## Why Not Use Simple Rules?

Using simple rules like splitting by spaces or dots can lead to errors. For example:

- "Dr." in "Dr. Smith" or "n.y." for New York are not sentence end markers.
  
To handle such cases, we need a library like spaCy, which incorporates language-specific rules.

## Getting Started with spaCy

### Installation

To install spaCy, run:
```bash
pip install spacy
```

### Setting Up

1. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
   Navigate to the directory where you want to work.

2. **Create a spaCy Tokenization Directory**:
   Create your desired directory, navigate to it, and run:
   ```bash
   python3
   ```

### Basic Usage

1. **Import spaCy**:
   ```python
   import spacy
   ```

2. **Create a Language Object**:
   ```python
   nlp = spacy.blank('en')  # 'en' for English, 'de' for German, etc.
   ```

3. **Create a Document**:
   ```python
   doc = nlp("Doctor Strange's visit in Mumbai. He loved pav bhaji so much.")
   ```

The language object (`nlp`) can be created in multiple ways, such as using `spacy.blank('en')` for English or other codes like 'de' for German. You can find all available language models via `spaCy language models`.

### Tokenization Example

1. **Tokenizing Text**:
   ```python
   for token in doc:
       print(token)
   ```
   This will tokenize the text into words, and you can access each token individually.

2. **Using Python List-Style Indexing**:
   ```python
   print(doc[0])  # Output: Dr.
   print(doc[1])  # Output: Strange
   ```

### Advanced Tokenization

spaCy's tokenizer handles various complexities, such as:

- **Splitting Currency**:
  ```python
  doc = nlp("I have $200.")
  for token in doc:
      print(token)
  ```
  This will split "$200" into two tokens: `$` and `200`.

- **Handling Punctuation**:
  ```python
  doc = nlp("Let's go to N.Y.!")
  for token in doc:
      print(token)
  ```
  This will properly tokenize "N.Y." and handle punctuation correctly.

### Benefits of Tokenization

Tokenizing text into individual components is beneficial for language analysis and building NLP applications. spaCy's tokenizer handles prefixes, suffixes, and exceptions efficiently.

- **Prefix Handling**: 
  ```python
  doc = nlp('"Let\'s go to N.Y.!"')
  for token in doc:
      print(token)
  ```
  This will split prefixes like quotes correctly.

- **Suffix Handling**: 
  ```python
  doc = nlp('The distance is 5km.')
  for token in doc:
      print(token)
  ```
  This will split suffixes like "km" from the preceding number.

- **Exception Handling**: 
  ```python
  doc = nlp("Let's go to N.Y.!")
  for token in doc:
      print(token)
  ```
  This handles exceptions such as "N.Y." correctly.

### Object Types and Attributes

- **Document Object**:
  ```python
  print(type(doc))  # Output: <class 'spacy.tokens.doc.Doc'>
  ```

- **Token Object**:
  ```python
  print(type(doc[0]))  # Output: <class 'spacy.tokens.token.Token'>
  ```

- **Span Object**:
  ```python
  span = doc[1:5]
  print(type(span))  # Output: <class 'spacy.tokens.span.Span'>
  ```

### Exploring Token Attributes

Tokens have various attributes that can be useful for text analysis:
```python
token = doc[0]
print(token.is_alpha)  # Check if the token is alphabetic
print(token.is_stop)   # Check if the token is a stop word
```

Use `dir()` to explore all available methods and attributes:
```python
print(dir(token))
```

## Cloud-Based NLP

For heavy NLP tasks requiring substantial compute resources, consider using cloud-based solutions. These services allow you to make HTTP calls and perform tasks on the cloud without needing local compute resources.

### Example Service

A cloud-based NLP service allows for text classification and other tasks via API calls. You can:

1. **Sign Up**: Access the free tier by signing up and obtaining an API key from the dashboard.
2. **SDKs**: Use available SDKs in Python or TypeScript.
3. **Text Classification Example**:
   - **Negative Review**:
     ```python
     # API call to classify text
     ```
   - **Positive Review**:
     ```python
     # API call to classify text
     ```

This simplifies the process and makes NLP tasks accessible without deep technical knowledge.

Tokenization is a fundamental NLP task, and spaCy provides powerful tools to handle it efficiently. Whether working locally or using cloud-based services, spaCy and similar tools offer flexibility and ease of use for a variety of NLP applications.




here we discuss tokenization using the spaCy library. While NLTK also offers tokenization, we've chosen spaCy for its advantages discussed in a previous video.

## Introduction to Tokenization

Tokenization is a key step in the NLP pipeline, particularly in pre-processing. This involves breaking down text into sentences and words. Here's a brief overview:

- **Sentence Tokenization**: Splitting a paragraph into individual sentences.
- **Word Tokenization**: Splitting each sentence into individual words.

## Why Not Use Simple Rules?

Using simple rules like splitting by spaces or dots can lead to errors. For example:

- "Dr." in "Dr. Smith" or "n.y." for New York are not sentence end markers.

To handle such cases, we need a library like spaCy, which incorporates language-specific rules.

## Getting Started with spaCy

### Installation

To install spaCy, run:
```bash
pip install spacy
```

### Setting Up

1. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
   Navigate to the directory where you want to work.

2. **Create a spaCy Tokenization Directory**:
   Create your desired directory, navigate to it, and run:
   ```bash
   python3
   ```

### Basic Usage

1. **Import spaCy**:
   ```python
   import spacy
   ```

2. **Create a Language Object**:
   ```python
   nlp = spacy.blank('en')  # 'en' for English, 'de' for German, etc.
   ```

3. **Create a Document**:
   ```python
   doc = nlp("Doctor Strange's visit in Mumbai. He loved pav bhaji so much.")
   ```

The language object (`nlp`) can be created in multiple ways, such as using `spacy.blank('en')` for English or other codes like 'de' for German. You can find all available language models via `spaCy language models`.

### Tokenization Example

1. **Tokenizing Text**:
   ```python
   for token in doc:
       print(token)
   ```
   This will tokenize the text into words, and you can access each token individually.

2. **Using Python List-Style Indexing**:
   ```python
   print(doc[0])  # Output: Dr.
   print(doc[1])  # Output: Strange
   ```

### Advanced Tokenization

spaCy's tokenizer handles various complexities, such as:

- **Splitting Currency**:
  ```python
  doc = nlp("I have $200.")
  for token in doc:
      print(token)
  ```
  This will split "$200" into two tokens: `$` and `200`.

- **Handling Punctuation**:
  ```python
  doc = nlp("Let's go to N.Y.!")
  for token in doc:
      print(token)
  ```
  This will properly tokenize "N.Y." and handle punctuation correctly.

### Benefits of Tokenization

Tokenizing text into individual components is beneficial for language analysis and building NLP applications. spaCy's tokenizer handles prefixes, suffixes, and exceptions efficiently.

- **Prefix Handling**: 
  ```python
  doc = nlp('"Let\'s go to N.Y.!"')
  for token in doc:
      print(token)
  ```
  This will split prefixes like quotes correctly.

- **Suffix Handling**: 
  ```python
  doc = nlp('The distance is 5km.')
  for token in doc:
      print(token)
  ```
  This will split suffixes like "km" from the preceding number.

- **Exception Handling**: 
  ```python
  doc = nlp("Let's go to N.Y.!")
  for token in doc:
      print(token)
  ```
  This handles exceptions such as "N.Y." correctly.

### Object Types and Attributes

- **Document Object**:
  ```python
  print(type(doc))  # Output: <class 'spacy.tokens.doc.Doc'>
  ```

- **Token Object**:
  ```python
  print(type(doc[0]))  # Output: <class 'spacy.tokens.token.Token'>
  ```

- **Span Object**:
  ```python
  span = doc[1:5]
  print(type(span))  # Output: <class 'spacy.tokens.span.Span'>
  ```

### Exploring Token Attributes

Tokens have various attributes that can be useful for text analysis:
```python
token = doc[0]
print(token.is_alpha)  # Check if the token is alphabetic
print(token.is_stop)   # Check if the token is a stop word
```

Use `dir()` to explore all available methods and attributes:
```python
print(dir(token))
```

#### Example: Token Attributes
```python
token = doc[2]  # Token: "two"
print(token.text)       # Output: two
print(token.is_alpha)   # Output: False
print(token.is_num)     # Output: True
print(token.is_currency) # Output: False
```

## Cloud-Based NLP

For heavy NLP tasks requiring substantial compute resources, consider using cloud-based solutions. These services allow you to make HTTP calls and perform tasks on the cloud without needing local compute resources.

### Example Service

A cloud-based NLP service allows for text classification and other tasks via API calls. You can:

1. **Sign Up**: Access the free tier by signing up and obtaining an API key from the dashboard.
2. **SDKs**: Use available SDKs in Python or TypeScript.
3. **Text Classification Example**:
   - **Negative Review**:
     ```python
     # API call to classify text
     ```
   - **Positive Review**:
     ```python
     # API call to classify text
     ```

This simplifies the process and makes NLP tasks accessible without deep technical knowledge.

## Use Case: Extracting Email Addresses

Imagine you have a file with student information and you need to extract email addresses to send notifications. Here's how spaCy can help:

1. **Read the File**:
   ```python
   with open('student.txt', 'r') as f:
       lines = f.readlines()
   text = ' '.join(lines)
   ```

2. **Tokenize and Extract Emails**:
   ```python
   doc = nlp(text)
   emails = [token.text for token in doc if token.like_email]
   print(emails)
   ```

This approach can be more convenient and powerful than using regular expressions.


Tokenization is a fundamental NLP task, and spaCy provides powerful tools to handle it efficiently. Whether working locally or using cloud-based services, spaCy and similar tools offer flexibility and ease of use for a variety of NLP applications.



here we discuss tokenization using the spaCy library. While NLTK also offers tokenization, we've chosen spaCy for its advantages discussed in a previous video.

## Introduction to Tokenization

Tokenization is a key step in the NLP pipeline, particularly in pre-processing. This involves breaking down text into sentences and words. Here's a brief overview:

- **Sentence Tokenization**: Splitting a paragraph into individual sentences.
- **Word Tokenization**: Splitting each sentence into individual words.

## Why Not Use Simple Rules?

Using simple rules like splitting by spaces or dots can lead to errors. For example:

- "Dr." in "Dr. Smith" or "N.Y." for New York are not sentence end markers.

To handle such cases, we need a library like spaCy, which incorporates language-specific rules.

## Getting Started with spaCy

### Installation

To install spaCy, run:
```bash
pip install spacy
```

### Setting Up

1. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
   Navigate to the directory where you want to work.

2. **Create a spaCy Tokenization Directory**:
   Create your desired directory, navigate to it, and run:
   ```bash
   python3
   ```

### Basic Usage

1. **Import spaCy**:
   ```python
   import spacy
   ```

2. **Create a Language Object**:
   ```python
   nlp = spacy.blank('en')  # 'en' for English, 'de' for German, etc.
   ```

3. **Create a Document**:
   ```python
   doc = nlp("Doctor Strange's visit in Mumbai. He loved pav bhaji so much.")
   ```

The language object (`nlp`) can be created in multiple ways, such as using `spacy.blank('en')` for English or other codes like 'de' for German. You can find all available language models via `spaCy language models`.

### Tokenization Example

1. **Tokenizing Text**:
   ```python
   for token in doc:
       print(token)
   ```
   This will tokenize the text into words, and you can access each token individually.

2. **Using Python List-Style Indexing**:
   ```python
   print(doc[0])  # Output: Dr.
   print(doc[1])  # Output: Strange
   ```

### Advanced Tokenization

spaCy's tokenizer handles various complexities, such as:

- **Splitting Currency**:
  ```python
  doc = nlp("I have $200.")
  for token in doc:
      print(token)
  ```
  This will split "$200" into two tokens: `$` and `200`.

- **Handling Punctuation**:
  ```python
  doc = nlp("Let's go to N.Y.!")
  for token in doc:
      print(token)
  ```
  This will properly tokenize "N.Y." and handle punctuation correctly.

### Benefits of Tokenization

Tokenizing text into individual components is beneficial for language analysis and building NLP applications. spaCy's tokenizer handles prefixes, suffixes, and exceptions efficiently.

- **Prefix Handling**: 
  ```python
  doc = nlp('"Let\'s go to N.Y.!"')
  for token in doc:
      print(token)
  ```
  This will split prefixes like quotes correctly.

- **Suffix Handling**: 
  ```python
  doc = nlp('The distance is 5km.')
  for token in doc:
      print(token)
  ```
  This will split suffixes like "km" from the preceding number.

- **Exception Handling**: 
  ```python
  doc = nlp("Let's go to N.Y.!")
  for token in doc:
      print(token)
  ```
  This handles exceptions such as "N.Y." correctly.

### Object Types and Attributes

- **Document Object**:
  ```python
  print(type(doc))  # Output: <class 'spacy.tokens.doc.Doc'>
  ```

- **Token Object**:
  ```python
  print(type(doc[0]))  # Output: <class 'spacy.tokens.token.Token'>
  ```

- **Span Object**:
  ```python
  span = doc[1:5]
  print(type(span))  # Output: <class 'spacy.tokens.span.Span'>
  ```

### Exploring Token Attributes

Tokens have various attributes that can be useful for text analysis:
```python
token = doc[0]
print(token.is_alpha)  # Check if the token is alphabetic
print(token.is_stop)   # Check if the token is a stop word
```

Use `dir()` to explore all available methods and attributes:
```python
print(dir(token))
```

#### Example: Token Attributes
```python
token = doc[2]  # Token: "two"
print(token.text)       # Output: two
print(token.is_alpha)   # Output: False
print(token.is_num)     # Output: True
print(token.is_currency) # Output: False
```

## Cloud-Based NLP

For heavy NLP tasks requiring substantial compute resources, consider using cloud-based solutions. These services allow you to make HTTP calls and perform tasks on the cloud without needing local compute resources.

### Example Service

A cloud-based NLP service allows for text classification and other tasks via API calls. You can:

1. **Sign Up**: Access the free tier by signing up and obtaining an API key from the dashboard.
2. **SDKs**: Use available SDKs in Python or TypeScript.
3. **Text Classification Example**:
   - **Negative Review**:
     ```python
     # API call to classify text
     ```
   - **Positive Review**:
     ```python
     # API call to classify text
     ```

This simplifies the process and makes NLP tasks accessible without deep technical knowledge.

## Use Case: Extracting Email Addresses

Imagine you have a file with student information and you need to extract email addresses to send notifications. Here's how spaCy can help:

1. **Read the File**:
   ```python
   with open('student.txt', 'r') as f:
       lines = f.readlines()
   text = ' '.join(lines)
   ```

2. **Tokenize and Extract Emails**:
   ```python
   doc = nlp(text)
   emails = [token.text for token in doc if token.like_email]
   print(emails)
   ```

This approach can be more convenient and powerful than using regular expressions.

## Accessing spaCy Documentation

To explore spaCy's documentation, Google "spaCy token attributes" or visit the official spaCy website. The documentation is well-written and includes articles and examples that are highly recommended. 

## Working with Different Languages

spaCy supports multiple languages. Let's look at an example using Hindi:

### Tokenizing Hindi Text

1. **Create a Language Object for Hindi**:
   ```python
   nlp = spacy.blank('hi')  # 'hi' is the code for Hindi
   ```

2. **Create a Document with Hindi Text**:
   ```python
   doc = nlp("उसने किसी से पैसे उधार लिए हैं और वह उन्हें लौटाने के लिए कह रहा है।")
   ```

3. **Tokenize and Print Attributes**:
   ```python
   for token in doc:
       print(token.text, token.is_alpha, token.like_num, token.is_currency)
   ```

This shows how spaCy can handle different languages, including their specific attributes.

### Customizing the Tokenizer

Sometimes you need to customize the tokenizer to handle specific cases, like slang or abbreviations. Here's how you can do that:

1. **Import Necessary Symbols**:
   ```python
   from spacy.symbols import ORTH
   ```

2. **Add Special Case**:
   ```python
   nlp.tokenizer.add_special_case("gimme", [{ORTH: "gim"}, {ORTH: "me"}])
   ```

3. **Create Document and Tokenize**:
   ```python
   doc = nlp("Gimme the book.")
   for token in doc:
       print(token)
   ```

This will split "gimme" into "gim" and "me".

### Sentence Tokenization

Sentence tokenization is another crucial task. Here's an example:

1. **Create a Document with Multiple Sentences**:
   ```python
   doc = nlp("Hulk and Strange both are loving India trip. They are enjoying the food.")
   ```

2. **Tokenize into Sentences**:
   ```python
   for sent in doc.sents:
       print(sent)
   ```


## Adding Components to the NLP Pipeline

When you create a blank NLP object with spaCy, it doesn't include any components other than the tokenizer. To perform more complex tasks like sentence tokenization, you'll need to add components to your pipeline.

### Checking the Pipeline

You can check the current components in your pipeline using:
```python
print(nlp.pipe_names)  # Output: []
```
Initially, this will be empty because a blank pipeline only includes the tokenizer.

### Adding a Sentencizer

To add a component for sentence segmentation:
```python
nlp.add_pipe('sentencizer')
print(nlp.pipe_names)  # Output: ['sentencizer']
```

### Tokenizing Sentences

After adding the sentencizer, you can tokenize text into sentences:
```python
doc = nlp("Dr. Strange loves New York. He also likes Mumbai.")
for sent in doc.sents:
    print(sent)
```
Note: The sentencizer might split sentences incorrectly if it doesn't fully understand the language. Using a full language model can help address these issues.

### Full NLP Pipeline

A full pipeline includes several components like the tagger, parser, and named entity recognizer (NER). You can load a full pipeline with:
```python
nlp = spacy.load('en_core_web_sm')
print(nlp.pipe_names)  # Output: ['tagger', 'parser', 'ner']
```

### Customizing the Tokenizer

Customizing the tokenizer allows you to handle specific cases, such as abbreviations or slang. Here’s an example of customizing the tokenizer to handle "gimme" as two tokens:
```python
from spacy.symbols import ORTH

nlp.tokenizer.add_special_case("gimme", [{ORTH: "gim"}, {ORTH: "me"}])
doc = nlp("Gimme the book.")
for token in doc:
    print(token)
```

## Exercise

Practical exercises help reinforce your learning. Here are two tasks to work on:

### Task 1: Extract URLs

Given a paragraph, write code to extract all URLs. For example:
```python
text = "Check out these datasets: https://example.com/dataset1, http://example.org/dataset2."
doc = nlp(text)
urls = [token.text for token in doc if token.like_url]
print(urls)  # Output: ['https://example.com/dataset1', 'http://example.org/dataset2']
```

### Task 2: Extract Monetary Transactions

Extract all monetary transactions from a text. For example:
```python
text = "The total cost was $200 and €500."
doc = nlp(text)
transactions = [token.text for token in doc if token.is_currency or token.like_num]
print(transactions)  # Output: ['$200', '€500']
```

### Practice and Apply

Working through these exercises will help you build practical skills in NLP using spaCy. The solutions and more exercises are available in the video description linked below. Avoid the temptation to check the solution before attempting the exercises on your own.

## Conclusion

Mastering NLP with spaCy requires practice and a solid understanding of the basics. By following this guide and completing the exercises, you'll be well on your way to becoming proficient in NLP.
