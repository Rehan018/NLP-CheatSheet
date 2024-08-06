### NLP Task: Text Classification : [Source Video](https://www.youtube.com/watch?v=In7jB8TUGPA&list=PLeo1K3hjS3uuvuAXhYjV2lMEShq2UYSwX&index=5)

**Definition:**
Text classification is a natural language processing (NLP) task that involves categorizing text into predefined classes or categories based on its content. This process is commonly used for sentiment analysis, spam detection, and topic labeling.

---

### Real-Life Use Case: Camtasia Support Forum

**Context:**
Imagine a scenario where users report bugs or issues with the Camtasia Studio software on their support forum. The company (TechSmith) receives numerous complaints daily, and they need to prioritize these complaints based on their severity (high, medium, low).

**Example Complaints:**
1. "Camtasia 10 won't import MP4 files."
2. "I can't find a video tutorial."

**Classification Objective:**
- Complaint 1: High Severity
- Complaint 2: Low Severity

---

### Process of Text Classification

1. **Text Vectorization:**
   To process the text, we need to convert it into a numerical format (vector) that machine learning algorithms can understand. One common method for this is **Term Frequency-Inverse Document Frequency (TF-IDF)**.

   - **TF-IDF Explanation:**
     - **Term Frequency (TF):** Measures how frequently a term appears in a document. It’s calculated as the number of times a term appears in a document divided by the total number of terms in that document.
     - **Inverse Document Frequency (IDF):** Measures how important a term is in the entire corpus of documents. It’s calculated as the logarithm of the total number of documents divided by the number of documents that contain the term.
     
     The TF-IDF score helps to identify the importance of a word in a document relative to a collection of documents.

2. **Vector Representation:**
   Once you compute the TF-IDF scores, you can represent each complaint as a vector of numbers. 

   Example Vector (for simplicity):
   - Complaint 1: `[0.7, 0.2, 0.1]` (High severity)
   - Complaint 2: `[0.1, 0.05, 0.85]` (Low severity)

3. **Classification Algorithm:**
   After vectorizing the text, you can use a classification algorithm like **Naive Bayes** to categorize the complaints. 

   - **Naive Bayes Classifier Explanation:**
     This probabilistic classifier applies Bayes' theorem, assuming independence between the features (words) in the vector. It is often used for text classification tasks due to its simplicity and effectiveness.

4. **Implementation in Python:**
   Below is a simple code example demonstrating the text classification process using Python and the `scikit-learn` library.

```python

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Sample complaints
data = [
    "Camtasia 10 won't import MP4 files.",
    "I can't find a video tutorial.",
    "The software crashes when I try to export.",
    "Where can I download additional effects?",
]

labels = ['High', 'Low', 'High', 'Low']

model = make_pipeline(TfidfVectorizer(), MultinomialNB())

model.fit(data, labels)

test_complaint = ["Cannot open the project file."]
predicted_severity = model.predict(test_complaint)

print(f"Predicted severity: {predicted_severity[0]}")
```

### Cosine Similarity

**Definition:**  
Cosine similarity is a metric used to measure how similar two vectors are, regardless of their magnitude. It calculates the cosine of the angle between two non-zero vectors in an inner product space. The value ranges from -1 to 1, where:
- 1 indicates that the vectors are identical,
- 0 indicates orthogonality (no similarity),
- -1 indicates that the vectors are diametrically opposed.

**Formula:**  
The cosine similarity between two vectors **A** and **B** can be computed using the formula:

\[
\text{cosine similarity} = \frac{A \cdot B}{\|A\| \|B\|}
\]

Where:
- \(A \cdot B\) is the dot product of vectors **A** and **B**.
- \(\|A\|\) and \(\|B\|\) are the magnitudes (norms) of the vectors.

**Use Case:**  
Cosine similarity is often used in various natural language processing tasks, such as:
- **Text Similarity:** Measuring how similar two text documents are based on their vector representations (e.g., TF-IDF or word embeddings).
- **Recommendation Systems:** Finding items (like movies or products) that are similar based on user preferences or content descriptions.

**Implementation in Python:**

```python
import numpy as np

A = np.array([1, 2, 3])
B = np.array([4, 5, 6])

dot_product = np.dot(A, B)

magnitude_A = np.linalg.norm(A)
magnitude_B = np.linalg.norm(B)

cosine_similarity = dot_product / (magnitude_A * magnitude_B)

print("Cosine Similarity:", cosine_similarity)
```

**In Practice:**  
In NLP tasks, you can use cosine similarity to compare text vectors. For instance, after converting documents or sentences into vectors using techniques like TF-IDF or word embeddings, you can compute the cosine similarity to determine how closely related they are. This is particularly useful in tasks like document clustering, plagiarism detection, and finding similar job resumes.

### Workflow
1. **Incoming Complaints:** Collect complaints from users.
2. **Vectorization:** Convert the text to numerical vectors using TF-IDF.
3. **Classification:** Use a classifier (e.g., Naive Bayes) to categorize complaints into severity levels.
4. **Action:** Depending on the predicted severity, route the complaint to the appropriate response workflow (e.g., high severity complaints may go directly to customer support).


### Additional Use Cases for Text Classification

1. **Healthcare Document Classification**

   **Context:**
   In the healthcare industry, there are numerous documents, including prescriptions and patient records, that need to be classified efficiently. Health professionals often scan these documents and upload them to a cloud storage system without the time to manually categorize each document.

   **Process:**
   - **OCR (Optical Character Recognition):** Use a tool like Tesseract to convert scanned images of documents into text data.
   - **Text Vectorization:** Apply a method like **Doc2Vec** to convert the extracted text into numerical vectors.
   - **Classification:** Use a classification algorithm such as **Logistic Regression** to categorize the documents as either prescriptions or patient medical records.

   **Example Implementation:**
   ```python
   import pytesseract
   from gensim.models.doc2vec import Doc2Vec
   from sklearn.linear_model import LogisticRegression
   from sklearn.pipeline import make_pipeline

   scanned_documents = ["prescription_image.jpg", "patient_record_image.jpg"]

   extracted_texts = [pytesseract.image_to_string(img) for img in scanned_documents]

   labeled_data = [gensim.models.doc2vec.TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(extracted_texts)]

   doc2vec_model = Doc2Vec(labeled_data, vector_size=100, window=2, min_count=1, workers=4)

   vectors = [doc2vec_model.infer_vector(text.split()) for text in extracted_texts]

   model = LogisticRegression()
   model.fit(vectors, ['Prescription', 'Patient Record'])

   new_document = "New patient record text here."
   new_vector = doc2vec_model.infer_vector(new_document.split())
   prediction = model.predict([new_vector])

   print(f"Document classification: {prediction[0]}")
   ```

2. **Hate Speech Detection**

   **Context:**
   Social media platforms like Facebook and LinkedIn face issues with hate speech, racist comments, and the creation of fake profiles. To combat this, companies develop classification algorithms that can automatically detect and categorize content.

   **Process:**
   - **Text Analysis:** The algorithm analyzes text posts to identify hate speech or abusive content.
   - **Training Data:** Use labeled datasets containing examples of hate speech and non-hate speech to train the model.
   - **Deployment:** Implement the trained model to monitor user-generated content and automatically remove posts that contain hate speech.

   **Example Implementation:**
   ```python
   from sklearn.feature_extraction.text import TfidfVectorizer
   from sklearn.naive_bayes import MultinomialNB
   from sklearn.pipeline import make_pipeline

   posts = [
       "I hate this group of people!",
       "What a beautiful day!",
       "All lives matter!",
       "Get out of here, you racist!"
   ]
   labels = ['Hate Speech', 'Non-Hate Speech', 'Non-Hate Speech', 'Hate Speech']

   hate_speech_model = make_pipeline(TfidfVectorizer(), MultinomialNB())

   hate_speech_model.fit(posts, labels)

   new_post = ["This is an offensive comment!"]
   predicted_category = hate_speech_model.predict(new_post)

   print(f"Predicted category: {predicted_category[0]}")
   ```

3. **Fake Profile Detection on LinkedIn**

   **Context:**
   LinkedIn has to monitor profiles to detect and eliminate fake accounts that may use deceptive practices to exploit users or manipulate data.

   **Process:**
   - **Feature Extraction:** Identify key features or keywords associated with genuine profiles versus fake profiles.
   - **Model Training:** Train a classification model using labeled data (genuine vs. fake profiles).
   - **Real-time Monitoring:** Implement the model to analyze new profiles and flag or block those that appear suspicious.

   **Example Implementation:**
   ```python
   from sklearn.feature_extraction.text import TfidfVectorizer
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.pipeline import make_pipeline

   profiles = [
       "Experienced software developer with skills in Python and machine learning.",
       "Click here to win a free iPhone!",
       "Marketing professional with 10+ years of experience.",
       "I make money fast with this online system!"
   ]
   labels = ['Genuine', 'Fake', 'Genuine', 'Fake']

   profile_model = make_pipeline(TfidfVectorizer(), RandomForestClassifier())

   profile_model.fit(profiles, labels)

   new_profile = ["Earn thousands overnight!"]
   predicted_type = profile_model.predict(new_profile)

   print(f"Predicted profile type: {predicted_type[0]}")
   ```

---

### Use Cases
- **Text Classification** helps organizations automate the categorization of various types of text, from user complaints to medical documents and social media posts.
- Machine learning models, such as Naive Bayes, Logistic Regression, and Random Forest, are commonly used for classification tasks.
- Technologies like OCR (Tesseract) and text vectorization methods (TF-IDF, Doc2Vec) play a crucial role in converting raw text into structured data for analysis. 


### Text Similarity Use Case

**Definition:**
Text similarity is an NLP task that involves measuring how alike two pieces of text are. This can be useful in various applications, such as resume screening, plagiarism detection, and information retrieval.

---

### Example Use Case: Resume Matching

**Context:**
In recruitment, hiring managers often receive numerous resumes for a single job position. For example, if there's an opening for a Data Scientist, they might receive hundreds or thousands of resumes, making it challenging to manually review each one. Using text similarity techniques can help automate this process.

**Process:**
1. **Job Description:** The job description outlines the necessary skills, qualifications, and experience for the position.
2. **Resume Submission:** Candidates submit their resumes detailing their skills and experiences.
3. **Text Similarity Calculation:** By comparing the text in resumes to the job description, recruiters can quickly identify candidates whose skills match the requirements.

### Implementation Steps

1. **Text Representation:**
   - Use a model like **Sentence Transformers** to convert text (job description and resumes) into vectors.
   - These vectors represent the semantic meaning of the text.

2. **Calculate Similarity:**
   - Compute the similarity score between the resume vectors and the job description vector using cosine similarity or other distance metrics.

3. **Threshold for Selection:**
   - Define a similarity threshold (e.g., 50%). Only resumes with a similarity score above this threshold are considered for further review.

### Example Code Implementation

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

job_description = "We are looking for a Data Scientist with experience in machine learning, Python, and data analysis."
resume1 = "Data Scientist with 5 years of experience in machine learning and data analysis using Python."
resume2 = "Graphic designer skilled in Adobe Photoshop and Illustrator."

job_desc_vector = model.encode(job_description, convert_to_tensor=True)
resume1_vector = model.encode(resume1, convert_to_tensor=True)
resume2_vector = model.encode(resume2, convert_to_tensor=True)

similarity1 = util.pytorch_cos_sim(job_desc_vector, resume1_vector)
similarity2 = util.pytorch_cos_sim(job_desc_vector, resume2_vector)

print(f"Similarity between job description and Resume 1: {similarity1.item() * 100:.2f}%")
print(f"Similarity between job description and Resume 2: {similarity2.item() * 100:.2f}%")

threshold = 0.5  # 50%
if similarity1.item() >= threshold:
    print("Resume 1 is a good match for the job.")
else:
    print("Resume 1 is not a good match for the job.")

if similarity2.item() >= threshold:
    print("Resume 2 is a good match for the job.")
else:
    print("Resume 2 is not a good match for the job.")
```

### Explanation of the Code:
- **SentenceTransformer:** This is a pre-trained model that can convert sentences into vectors while preserving their semantic meaning.
- **Cosine Similarity:** This metric is used to determine how similar two vectors are. A higher cosine similarity score indicates that the texts are more alike.
- **Thresholding:** By comparing the similarity scores to a predefined threshold, the recruiter can quickly decide which resumes are worth reviewing.

---

### Similarity in Recruitment:
- **Automates Resume Screening:** Text similarity can drastically reduce the time needed to filter through resumes.
- **Increases Efficiency:** Recruiters can focus on candidates who closely match job requirements, improving the overall hiring process.
- **Utilizes Advanced NLP Techniques:** Sentence encoding and similarity calculations enable a deeper understanding of the text, moving beyond simple keyword matching.



### Information Extraction and Information Retrieval

**Definition:**
- **Information Extraction (IE):** The process of automatically extracting structured information from unstructured text. This often involves identifying specific entities, relationships, or events within the text.
- **Information Retrieval (IR):** The process of finding relevant documents or pieces of information from a large dataset or corpus, based on user queries. This typically involves searching through indexed data to retrieve documents that match search criteria.

---

### Use Case: Information Extraction

**Example: Flight Itinerary Extraction**

1. **Context:**
   When you receive an email with your flight itinerary, services like Gmail automatically extract useful information such as:
   - Flight number
   - Departure and arrival times
   - Passenger name
   - Confirmation number

2. **Process:**
   - **Regular Expressions:** Used to identify and extract specific patterns in the text (e.g., flight numbers, dates).
   - **Natural Language Processing (NLP):** Techniques such as tokenization and named entity recognition (NER) can be employed to identify relevant entities within the email.

**Example Implementation:**
Here’s a simple example using Python and regular expressions to extract flight information from a sample email text.

```python
import re

email_text = """
Dear John Doe,

Your flight itinerary is as follows:
Flight ID: AA1234
Departure: 10:00 AM from JFK
Arrival: 1:00 PM at LAX
Confirmation Number: ABC123

Thank you for choosing our airline!
"""

flight_id_pattern = r'Flight ID:\s*(\w+[\d]+)'
departure_pattern = r'Departure:\s*(.*)from\s*(\w+)'
arrival_pattern = r'Arrival:\s*(.*)at\s*(\w+)'
confirmation_number_pattern = r'Confirmation Number:\s*(\w+)'

flight_id = re.search(flight_id_pattern, email_text).group(1)
departure = re.search(departure_pattern, email_text).groups()
arrival = re.search(arrival_pattern, email_text).groups()
confirmation_number = re.search(confirmation_number_pattern, email_text).group(1)

print(f"Flight ID: {flight_id}")
print(f"Departure: {departure[0]} from {departure[1]}")
print(f"Arrival: {arrival[0]} at {arrival[1]}")
print(f"Confirmation Number: {confirmation_number}")
```

### Use Case: Information Retrieval

**Example: Google Search**

1. **Context:**
   When you type a query such as "vada pav places near me" into Google, the search engine retrieves and displays a list of relevant web pages based on indexed content that matches your query.

2. **Process:**
   - **Indexing:** Google indexes billions of web pages to facilitate quick retrieval of relevant documents.
   - **Query Processing:** The search engine analyzes the user's query to understand intent and context.
   - **Ranking:** Retrieved documents are ranked based on relevance, utilizing algorithms like PageRank and machine learning techniques.

### Differences Between Information Extraction and Information Retrieval:

| Feature                     | Information Extraction               | Information Retrieval               |
|-----------------------------|--------------------------------------|-------------------------------------|
| Objective                   | Extract structured information       | Retrieve relevant documents         |
| Input                       | Unstructured text                    | Structured queries                  |
| Output                      | Structured data (entities, facts)    | List of documents                   |
| Techniques                  | Regular expressions, NER             | Indexing, ranking algorithms        |
| Example                     | Extracting flight details from email | Searching for nearby restaurants    |

---

