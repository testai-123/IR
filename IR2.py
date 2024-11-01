import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 

nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))


doc_dict =[
    ["d1","NLP has various text preprocessing techniques"],
    ["d2","Text Preprocessing techniques includes Tokenization, Stemming, Stopwords Removal, etc"],
    ["d3","Tokenization meaning breaking down of sentences into small individual tokens"],
    ["d4","NLP techniques forms a base for GenAI"]
]
    
doc_with_best_match = []
doc_with_exact_match = []

query = input("Enter the query :")

query_tokens = [word.lower() for word in word_tokenize(query) if word.lower() not in stop_words]

for doc_id, content in doc_dict:
    
    content_tokens = [word.lower() for word in word_tokenize(content) if word.lower() not in stop_words]
    
    matched_words = [word for word in query_tokens if word in content_tokens]
    
    if matched_words:
        doc_with_best_match.append((doc_id,matched_words))
        
    
    if all(word in content_tokens for word in query_tokens):
        doc_with_best_match.remove((doc_id,matched_words))
        doc_with_exact_match.append(doc_id)
        
print("Exact Match :",doc_with_exact_match)
print("Best Match :",doc_with_best_match)    
