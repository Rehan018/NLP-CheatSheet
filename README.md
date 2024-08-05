# NLP-CheatSheet



### **Use Cases in NLP: Customer Service Chatbot and Information Extraction**

**1. Customer Service Chatbot:**
   - **Description:** Chatbots assist customers by retrieving order details or resolving issues based on user input.
   - **Example Scenario:** 
     - **User Input:** "I'm having an issue with my order number #123456."
     - **Chatbot Response:** Extracts order number using pattern matching.
   - **Pattern Recognition:** 
     - Variations in user input:
       - "My order has an issue."
       - "I have a problem with my order number 123456."
     - **Regular Expression Pattern:** 
       - Common pattern: `order\s*#?\s*\d+`
       - Matches "order", optional "#", and a sequence of digits.

**2. Information Extraction:**
   - **Description:** Extracting key information such as phone numbers and email addresses from user responses.
   - **Example Scenario:** 
     - **User Input:** "Here's my phone number (123)-456-7890 and email john.doe@example.com."
   - **Pattern Recognition:** 
     - **Email Pattern:** 
       - Common pattern: `\w+@\w+\.\w+`
       - Matches username, "@" symbol, domain, and top-level domain.
     - **Phone Number Patterns:** 
       - Format 1: `\(\d{3}\)-\d{3}-\d{4}`
       - Format 2: `\d{10}` (continuous 10 digits)

### **Regular Expressions: Key Concepts**
- **Definition:** A sequence of characters that forms a search pattern, mainly used for string matching.
- **Importance in NLP:** 
  - Regular expressions are essential for extracting structured data from unstructured text.
  - Useful in various scenarios where machine learning is not necessary.

### **Getting Started with Python:**
- **Prerequisites:**
  - Basic knowledge of Python.
  - Familiarity with regular expressions.
  
- **Installation Steps:**
  1. **Install Python:** Search for an installation video on YouTube and follow the instructions.
  2. **Install Git Bash:** Useful for running Unix commands on Windows.

### **Sample Python Code to Extract Information:**
```python
import re

# Sample user input
user_input = "I'm having an issue with my order number #123456. Here's my phone number (123)-456-7890 and email john.doe@example.com."

# Patterns
order_pattern = r'order\s*#?\s*(\d+)'
phone_pattern = r'\(\d{3}\)-\d{3}-\d{4}'
email_pattern = r'(\w+@\w+\.\w+)'

# Extracting information
order_number = re.findall(order_pattern, user_input)
phone_number = re.findall(phone_pattern, user_input)
email_address = re.findall(email_pattern, user_input)

# Display results
print("Extracted Order Number:", order_number)
print("Extracted Phone Number:", phone_number)
print("Extracted Email Address:", email_address)
```

### **Exercise for Viewers:**
- **Task:** Create your  regular expression patterns to extract additional information, such as dates or addresses from a sample text. Test your patterns with various user inputs.
  Here are some regular expression patterns for extracting dates and addresses from sample text, along with examples and explanations.

### 1. Extracting Dates

#### Regular Expression for Dates
```regex
\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\w{3,9} \d{1,2}, \d{4})\b
```

- **Explanation**:
  - `\b`: Word boundary to ensure we match whole words.
  - `(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}`: Matches dates in the format `dd-mm-yyyy` or `dd/mm/yyyy`.
  - `|\w{3,9} \d{1,2}, \d{4}`: Matches dates in the format `Month dd, yyyy`.
  - `\b`: Another word boundary.

#### Test Inputs
- "Today is 05-08-2024."
- "The event is scheduled for August 5, 2024."
- "I was born on 12/01/99."

### 2. Extracting Addresses

#### Regular Expression for Addresses
```regex
\d+\s[A-Za-z0-9\s,.-]+(?:St|Ave|Blvd|Rd|Ln|Dr|Ct|Terr|Way|Pl|Pkwy)\.?\s*\d{5}
```

- **Explanation**:
  - `\d+`: Matches the house number.
  - `\s`: Matches a whitespace.
  - `[A-Za-z0-9\s,.-]+`: Matches the street name, which can contain letters, numbers, spaces, commas, periods, and hyphens.
  - `(?:St|Ave|Blvd|Rd|Ln|Dr|Ct|Terr|Way|Pl|Pkwy)`: Non-capturing group for common street types.
  - `\.?`: Matches an optional period.
  - `\s*`: Matches optional whitespace.
  - `\d{5}`: Matches the zip code (5 digits).

#### Test Inputs
- "I live at 123 Main St. 90210."
- "Send the package to 456 Elm Avenue 12345."
- "Her office is located at 789 Broadway Blvd, New York, NY 10001."

### Testing the Patterns

```python
import re

# Sample text for dates
date_texts = [
    "Today is 05-08-2024.",
    "The event is scheduled for August 5, 2024.",
    "I was born on 12/01/99."
]

# Date regex pattern
date_pattern = r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\w{3,9} \d{1,2}, \d{4})\b'

# Extracting dates
for text in date_texts:
    dates = re.findall(date_pattern, text)
    print(f"Extracted dates from '{text}': {dates}")


# Sample text for addresses
address_texts = [
    "I live at 123 Main St. 90210.",
    "Send the package to 456 Elm Avenue 12345.",
    "Her office is located at 789 Broadway Blvd, New York, NY 10001."
]

# Address regex pattern
address_pattern = r'\d+\s[A-Za-z0-9\s,.-]+(?:St|Ave|Blvd|Rd|Ln|Dr|Ct|Terr|Way|Pl|Pkwy)\.?\s*\d{5}'

# Extracting addresses
for text in address_texts:
    addresses = re.findall(address_pattern, text)
    print(f"Extracted addresses from '{text}': {addresses}")
```

### Output
This will print the extracted dates and addresses based on the given patterns and sample texts.



```
Second Phase
```

### **Setting Up the Environment for NLP Coding**

**1. Command Prompt Setup:**
   - **Windows:** You can use Windows Command Prompt or Git Bash for running commands.
   - **Linux:** Ubuntu users can use the default command prompt without additional installations.

**2. Prerequisites:**
   - **Jupyter Notebook:** Learn how to use Jupyter Notebook for writing Python code.
     - **Recommendation:** Search for "Code Basics Jupyter Notebook Tutorial" on YouTube for guidance.
   - **Python Knowledge:** Follow the first 14 videos from the Code Basics Python tutorial playlist to understand Python fundamentals.
   - **Regular Expressions Knowledge:** Review the Code Basics Regular Expression video to get acquainted with regex.

**3. Creating a Directory:**
   - **Navigate to the Directory:** Create a directory called `nlp_tutorials` and navigate to it using Git Bash or your command prompt.
   - **Launch Jupyter Notebook:** 
     - Open Git Bash and run the command `jupyter notebook` to start the Jupyter environment.
   - **Create a New Python File:** Create a new file (e.g., `regex_tutorial_nlp.py`) for your regex tutorial.

### **Using Regular Expressions in Python:**
**1. Importing the Regex Module:**
   - In your Python file, import the `re` module, which is built into Python for handling regular expressions.
   ```python
   import re
   ```

**2. Testing Regular Expressions:**
   - Use the website [regex101.com](https://regex101.com) to test and refine your regular expressions before using them in your Python code.

### **Extracting Phone Numbers:**
**1. Phone Number Patterns:**
   - **Pattern 1:** Continuous sequence of 10 digits (e.g., `1234567890`).
   - **Pattern 2:** Formatted as `(123)-456-7890`.

**2. Regular Expression for Phone Numbers:**
   - You can use the following regex patterns for the two formats:
     - Continuous 10 digits: `\d{10}`
     - Bracketed format: `\(\d{3}\)-\d{3}-\d{4}`

### **Example Python Code to Extract Phone Numbers:**
```python
import re

# Sample text containing phone numbers
text = """
You can reach me at (123)-456-7890 or call my alternate number 9876543210.
"""

# Regular expression patterns
pattern1 = r'\d{10}'  # Continuous 10 digits
pattern2 = r'\(\d{3}\)-\d{3}-\d{4}'  # Bracketed format

# Extracting phone numbers
continuous_numbers = re.findall(pattern1, text)
bracketed_numbers = re.findall(pattern2, text)

# Display results
print("Extracted Continuous Phone Numbers:", continuous_numbers)
print("Extracted Bracketed Phone Numbers:", bracketed_numbers)
```

### **Extracting Email Addresses:**
**1. Email Pattern:**
   - **Regular Expression for Email:** A basic pattern could be `\w+@\w+\.\w+`.

### **Example Python Code to Extract Email Addresses:**
```python
# Sample text containing email addresses
text_with_emails = "Contact me at john.doe@example.com or jane_doe123@gmail.com."

# Regular expression pattern for emails
email_pattern = r'\w+@\w+\.\w+'

# Extracting email addresses
emails = re.findall(email_pattern, text_with_emails)

# Display results
print("Extracted Email Addresses:", emails)
```

### **Exercise for Viewers:**
- **Task:** Write regex patterns to extract additional information, such as dates in the format `MM/DD/YYYY`, from a given text sample. Test your patterns using regex101.com and implement them in your Python code.

### Regex Pattern for Dates in MM/DD/YYYY Format

#### Regular Expression
```regex
\b(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/\d{4}\b
```

- **Explanation**:
  - `\b`: Asserts a word boundary to ensure we match whole words.
  - `(0[1-9]|1[0-2])`: Matches the month part, allowing for `01` to `12`.
  - `/`: Matches the literal forward slash.
  - `(0[1-9]|[12][0-9]|3[01])`: Matches the day part, allowing for `01` to `31`.
  - `/`: Matches another literal forward slash.
  - `\d{4}`: Matches the year part, which consists of four digits.
  - `\b`: Another word boundary.

### Testing the Pattern on regex101.com

1. Go to [regex101.com](https://regex101.com/).
2. Paste the regex pattern above in the "Regular Expression" field.
3. In the "Test String" field, enter sample text, such as:
   ```
   Today's date is 08/05/2024. Tomorrow will be 08/06/2024. 
   The deadline is 12/31/2024, but I can't remember 13/01/2024.
   ```
4. Verify that the matches highlight the correct dates.

### Implementing in Python Code

Here’s how to implement the regex in Python to extract dates in the `MM/DD/YYYY` format:

```python
import re

# Sample text
text = """
Today's date is 08/05/2024. Tomorrow will be 08/06/2024. 
The deadline is 12/31/2024, but I can't remember 13/01/2024.
"""

# Regex pattern for MM/DD/YYYY format
date_pattern = r'\b(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/\d{4}\b'

# Extracting dates
extracted_dates = re.findall(date_pattern, text)

# Formatting the output
formatted_dates = ['/'.join(date) for date in extracted_dates]

# Print the extracted dates
print("Extracted dates in MM/DD/YYYY format:", formatted_dates)
```

### Output
Running the above code will print:
```
Extracted dates in MM/DD/YYYY format: ['08/05/2024', '08/06/2024', '12/31/2024']
```



