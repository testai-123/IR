import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download NLTK resources (only need to run once)
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Tokenize the text
    words = word_tokenize(text)
    print("After Tokenization:", words)

    # Convert to lowercase
    words = [word.lower() for word in words]
    print("After Lowercasing:", words)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    print("After Stop Word Removal:", words)

    # Perform stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    print("After Stemming:", words)

    # Join the processed words back into a single string
    processed_text = ' '.join(words)
    return processed_text

# Sample text
text = "This is a simple example to demonstrate text preprocessing, including stop word removal and stemming."

# Preprocess the text
processed_text = preprocess_text(text)

# Print the final result
print("\nOriginal Text:", text)
print("Processed Text:", processed_text)

