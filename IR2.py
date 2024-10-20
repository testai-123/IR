import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources (only need to run once)
nltk.download('punkt')
nltk.download('stopwords')

# Stop words
stop_words = set(stopwords.words('english'))

# Document dictionary
doc_dict = [
    ["d1", "Welcome to hotel heaven such a lovely place"],
    ["d2", "She is buying a stairway to heaven"],
    ["d3", "Don't make it bad"],
    ["d4", "Take me to the heaven"]
]

# Get query from user
query = input("Enter query: ")

# Lists for storing matches
doc_with_exact_match = []
doc_with_best_match = []

# Tokenize the query
query_tokens = [word.lower() for word in word_tokenize(query)]

# Process each document
for doc_id, content in doc_dict:
    # Tokenize and filter stop words from the document
    content_tokens = [word.lower() for word in word_tokenize(content) if word.lower() not in stop_words]

    # Check for best match (at least one matching word)
    if any(word in content_tokens for word in query_tokens):
        doc_with_best_match.append(doc_id)

    # Check for exact match (all query words must match)
    if all(word in content_tokens for word in query_tokens):
        # Move from best match to exact match
        doc_with_best_match.remove(doc_id)
        doc_with_exact_match.append(doc_id)

# Display the results
print("Documents with best match:", doc_with_best_match)
print("Documents with exact match:", doc_with_exact_match)

