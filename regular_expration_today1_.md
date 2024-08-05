# NLP-CheatSheet



### **Use Cases in NLP: Customer Service Chatbot and Information Extraction** - [Source Video](https://youtu.be/lK9gx4q_vfI?si=o0-qEuokP-3ltGtf)

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
user_input = "I'm having an issue with my order number #123456. Here's my phone number (123)-456-7890 and email john.doe@example.com."
order_pattern = r'order\s*#?\s*(\d+)'
phone_pattern = r'\(\d{3}\)-\d{3}-\d{4}'
email_pattern = r'(\w+@\w+\.\w+)'


order_number = re.findall(order_pattern, user_input)
phone_number = re.findall(phone_pattern, user_input)
email_address = re.findall(email_pattern, user_input)

print("Extracted Order Number:", order_number)
print("Extracted Phone Number:", phone_number)
print("Extracted Email Address:", email_address)
```

### **Problem:**
- **Task:** Create your  regular expression patterns to extract additional information, such as dates or addresses from a sample text. Test your patterns with various user inputs.

### Solution

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
date_texts = [
    "Today is 05-08-2024.",
    "The event is scheduled for August 5, 2024.",
    "I was born on 12/01/99."
]
date_pattern = r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\w{3,9} \d{1,2}, \d{4})\b'
for text in date_texts:
    dates = re.findall(date_pattern, text)
    print(f"Extracted dates from '{text}': {dates}")

address_texts = [
    "I live at 123 Main St. 90210.",
    "Send the package to 456 Elm Avenue 12345.",
    "Her office is located at 789 Broadway Blvd, New York, NY 10001."
]
address_pattern = r'\d+\s[A-Za-z0-9\s,.-]+(?:St|Ave|Blvd|Rd|Ln|Dr|Ct|Terr|Way|Pl|Pkwy)\.?\s*\d{5}'


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


text = """
You can reach me at (123)-456-7890 or call my alternate number 9876543210.
"""
pattern1 = r'\d{10}'  # Continuous 10 digits
pattern2 = r'\(\d{3}\)-\d{3}-\d{4}'  # Bracketed format

continuous_numbers = re.findall(pattern1, text)
bracketed_numbers = re.findall(pattern2, text)
s
print("Extracted Continuous Phone Numbers:", continuous_numbers)
print("Extracted Bracketed Phone Numbers:", bracketed_numbers)
```

### **Extracting Email Addresses:**
**1. Email Pattern:**
   - **Regular Expression for Email:** A basic pattern could be `\w+@\w+\.\w+`.

### **Example Python Code to Extract Email Addresses:**
```python

text_with_emails = "Contact me at john.doe@example.com or jane_doe123@gmail.com."

email_pattern = r'\w+@\w+\.\w+'

emails = re.findall(email_pattern, text_with_emails)

print("Extracted Email Addresses:", emails)
```

### **Problem:**
- **Task:** Write regex patterns to extract additional information, such as dates in the format `MM/DD/YYYY`, from a given text sample. Test your patterns using [regex101.com](https://regex101.com) website and implement them in your Python code.

### Solution:
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

text = """
Today's date is 08/05/2024. Tomorrow will be 08/06/2024. 
The deadline is 12/31/2024, but I can't remember 13/01/2024.
"""

date_pattern = r'\b(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/\d{4}\b'

extracted_dates = re.findall(date_pattern, text)

formatted_dates = ['/'.join(date) for date in extracted_dates]

print("Extracted dates in MM/DD/YYYY format:", formatted_dates)
```

### Output
Running the above code will print:
```
Extracted dates in MM/DD/YYYY format: ['08/05/2024', '08/06/2024', '12/31/2024']
```


```
Third Phase
```

### **Using Regular Expressions for Phone Numbers and Email Addresses**

**1. Understanding Regular Expressions:**
   - **Matching Digits:**
     - Use `\d` to match a single digit (0-9).
     - To match multiple digits, use `{n}` where `n` is the number of digits you want to match.
       - Example: `\d{10}` matches exactly 10 digits.
   - **Matching Patterns:**
     - `\d{3}` matches exactly three continuous digits.
     - To extract sequences, use `re.findall()` to find all occurrences of the pattern in the text.

### **Extracting Phone Numbers:**
**1. Patterns for Phone Numbers:**
   - **Continuous Digits:** 
     - `\d{10}` for matching 10 consecutive digits.
   - **Formatted Numbers:** 
     - For the format `(123)-456-7890`, you need to escape the parentheses with a backslash: `\(\d{3}\)-\d{3}-\d{4}`.
   - **Combining Patterns:**
     - Use the pipe `|` symbol for an OR condition.
     - Example: `r'\(\d{3}\)-\d{3}-\d{4}|\d{10}'` matches both formats.

### **Example Python Code to Extract Phone Numbers:**
```python
import re

text = """
Contact me at (123)-456-7890 or my other number 9876543210. 
I can also be reached at 999-888-1234.
"""
pattern = r'\(\d{3}\)-\d{3}-\d{4}|\d{10}'

phone_numbers = re.findall(pattern, text)

print("Extracted Phone Numbers:", phone_numbers)
```

### **Extracting Email Addresses:**
**1. Basic Pattern for Emails:**
   - Email addresses generally follow the format `username@domain.extension`.
   - A simple regex pattern to match basic email formats can be: `\w+@\w+\.\w+`.
   - To make it more robust, include possible characters (letters, digits, underscores) before and after the `@` symbol.

### **Example Python Code to Extract Email Addresses:**
```python

text_with_emails = "You can contact me at john.doe@example.com or jane_doe123@gmail.com."


email_pattern = r'\w+@\w+\.\w+'

emails = re.findall(email_pattern, text_with_emails)

print("Extracted Email Addresses:", emails)
```

### **Creating More Complex Patterns:**
- **Emails with Numbers:**
  - Example of a more complex email pattern: `r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}'`.
- **Extraction Code:**
```python

complex_email_text = "Reach out to support123@domain.co or info@example.org."

complex_email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}'

complex_emails = re.findall(complex_email_pattern, complex_email_text)

print("Extracted Complex Email Addresses:", complex_emails)
```

### **Problem:**
- **Task:** Create regex patterns to extract additional information such as URLs or dates from a provided text. Test and refine your patterns using[regex101.com](https://regex101.com) website and implement them in your Python code.

### **Solution**
### Regex Patterns

1. **URL Pattern**:
   ```regex
   https?://[^\s/$.?#].[^\s]*
   ```
   - This pattern matches URLs starting with `http://` or `https://`, followed by any characters that do not include whitespace, making it a broad match for most URLs.

2. **Date Pattern** (for formats like `YYYY-MM-DD`, `DD/MM/YYYY`, or `MM-DD-YYYY`):
   ```regex
   \b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[/]\d{1,2}[/]\d{4}|\d{1,2}[-]\d{1,2}[-]\d{4})\b
   ```
   - This pattern matches dates in the formats `YYYY-MM-DD`, `DD/MM/YYYY`, and `MM-DD-YYYY`.

### Testing Patterns
You can test these regex patterns on [regex101.com](https://regex101.com/) to ensure they capture the intended matches. Paste your test text and the regex patterns to see what matches are found.

### Python Implementation
```python
import re

def extract_information(text):
    url_pattern = r'https?://[^\s/$.?#].[^\s]*'
    date_pattern = r'\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[/]\d{1,2}[/]\d{4}|\d{1,2}[-]\d{1,2}[-]\d{4})\b'
    urls = re.findall(url_pattern, text)
    dates = re.findall(date_pattern, text)

    return urls, dates

sample_text = """
Visit our website at https://www.example.com for more details.
Key dates to remember: 2024-08-15, 12/09/2024, and 03-03-2024.
"""
urls, dates = extract_information(sample_text)

print("Extracted URLs:", urls)
print("Extracted Dates:", dates)
```

### Explanation of the Python Code
- **Import `re`**: The regex library in Python.
- **Define `extract_information` function**: This function takes a text string as input and uses `re.findall` to extract all matching URLs and dates based on the defined patterns.
- **Example Usage**: A sample text is provided to demonstrate how the function works, and it prints the extracted URLs and dates.


```
Forth Phase
```

### **Extracting Information with Regular Expressions**

**1. Extracting Email Addresses:**
   - **Valid Characters in Emails:**
     - Small letters (a-z), capital letters (A-Z), numbers (0-9), and underscores (_).
   - **Email Pattern Construction:**
     - Use ranges to specify valid characters.
     - The regex pattern for emails can look like this:
       ```regex
       [a-zA-Z0-9._]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}
       ```
   - **Special Characters:**
     - `.` has a special meaning in regex (matches any character). To match a literal dot, use `\.`.

### **Example Python Code to Extract Email Addresses:**
```python
import re

text = "You can contact me at john.doe@example.com or jane_doe123@gmail.com."

email_pattern = r'[a-zA-Z0-9._]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}'

emails = re.findall(email_pattern, text)

print("Extracted Email Addresses:", emails)
```

**2. Extracting Order Numbers:**
   - **Order Number Patterns:**
     - Extract phrases like "my order number is 12345" or "order number 98765".
   - **Regex Construction:**
     - Match the word "order", followed by any non-digit characters, and then capture the sequence of digits.
     - Example pattern:
       ```regex
       order[^0-9]*([\d]+)
       ```
     - `[^0-9]*` matches any character that is not a digit, and `([\d]+)` captures the order number.

### **Example Python Code to Extract Order Numbers:**
```python
text_with_orders = "My order number is 12345. Also, order number: 98765."

order_pattern = r'order[^0-9]*([\d]+)'

order_numbers = re.findall(order_pattern, text_with_orders)

print("Extracted Order Numbers:", order_numbers)
```

**3. Information Extraction from Websites:**
   - **Using Wikipedia:**
     - You can extract key information by scraping or parsing Wikipedia pages.
   - **Key Steps:**
     1. Use libraries like `requests` to fetch the page content.
     2. Use `BeautifulSoup` for parsing the HTML.
     3. Apply regex patterns to extract specific information, like names, dates, or facts.

### **Example Code to Scrape Wikipedia:**
```python
import requests
from bs4 import BeautifulSoup
response = requests.get('https://en.wikipedia.org/wiki/Elon_Musk')
soup = BeautifulSoup(response.text, 'html.parser')

info_box = soup.find('table', class_='infobox')
rows = info_box.find_all('tr')

for row in rows:
    print(row.text.strip())
```


```
Fifth Phase
```

### **Information Extraction Using Regular Expressions**

**1. Context:**
   - This segment discusses how to retrieve key personal information from text, similar to what Google does using multiple sources (like Wikipedia).

**2. Key Information Extraction Tasks:**
   - Extract age, name, birthplace, and birthdate of a person from a structured text format.

---

### **Key Concepts in Regular Expressions:**

**1. Extracting Age:**
   - **Pattern for Age:**
     - The pattern follows the format: `age <space> <number>`.
     - Use the regex pattern:
       ```regex
       age\s+(\d+)
       ```
     - Explanation:
       - `age`: matches the literal word "age".
       - `\s+`: matches one or more whitespace characters.
       - `(\d+)`: captures one or more digits, which represents the age.

### **Example Python Code to Extract Age:**
```python
import re

def extract_age(text):
    age_pattern = r'age\s+(\d+)'
    match = re.search(age_pattern, text)
    return int(match.group(1)) if match else None

text = "Elon Musk, age 53, is a billionaire."
age = extract_age(text)
print("Extracted Age:", age)
```

---

**2. Extracting Name:**
   - **Pattern for Name:**
     - Names often follow the word "born".
     - Use the regex pattern:
       ```regex
       born\s+(.+)
       ```
     - Explanation:
       - `born`: matches the literal word "born".
       - `\s+`: matches one or more whitespace characters.
       - `(.+)`: captures any characters following "born" until the end of the line.

### **Example Python Code to Extract Name:**
```python
def extract_name(text):
    name_pattern = r'born\s+(.+)'
    match = re.search(name_pattern, text)
    return match.group(1).strip() if match else None

text = "Elon Musk was born in 1971."
name = extract_name(text)
print("Extracted Name:", name)
```

**3. Extracting Birthdate:**
   - **Pattern for Birthdate:**
     - Birthdate follows after the age line.
     - Use the regex pattern:
       ```regex
       born\s+\d+\s+\(.*?\)\s+(\w+\s+\d+,\s+\d+)
       ```
     - Explanation:
       - This pattern matches the birthdate format like "June 28, 1971".

### **Example Python Code to Extract Birthdate:**
```python
def extract_birthdate(text):
    birthdate_pattern = r'born\s+\d+\s+\(.*?\)\s+(\w+\s+\d+,\s+\d+)'
    match = re.search(birthdate_pattern, text)
    return match.group(1) if match else None

text = "Elon Musk was born on June 28, 1971."
birthdate = extract_birthdate(text)
print("Extracted Birthdate:", birthdate)
```

---

**4. Extracting Birthplace:**
   - **Pattern for Birthplace:**
     - Birthplace often follows the birthdate.
     - Use the regex pattern:
       ```regex
       age\s+\d+\s+\n(.+)
       ```
     - Explanation:
       - This pattern captures the line immediately following the age line.

### **Example Python Code to Extract Birthplace:**
```python
def extract_birthplace(text):
    birthplace_pattern = r'age\s+\d+\s+\n(.+)'
    match = re.search(birthplace_pattern, text)
    return match.group(1).strip() if match else None

text = "Elon Musk, age 53, was born in Pretoria, South Africa."
birthplace = extract_birthplace(text)
print("Extracted Birthplace:", birthplace)
```

---

### **Combining Extraction Functions:**
- Create a function that combines all the extraction functionalities into a single dictionary for easy access.

### **Example Combined Function:**
```python
def extract_personal_info(text):
    return {
        'age': extract_age(text),
        'name': extract_name(text),
        'birthdate': extract_birthdate(text),
        'birthplace': extract_birthplace(text)
    }

text = """
Elon Musk, age 53, was born on June 28, 1971, in Pretoria, South Africa.
"""
personal_info = extract_personal_info(text)
print("Extracted Personal Information:", personal_info)
```
