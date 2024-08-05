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
