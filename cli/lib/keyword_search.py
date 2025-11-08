import string
from nltk.stem import PorterStemmer
from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_Stop_words

stemmer = PorterStemmer()
def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    
    results = []
    for movie in movies:
        query_tokens = tokenization(query)
        title_tokens = tokenization(movie["title"])
        query_token = []
        for token in query_tokens:
            query_token.append(stemmer.stem(token))
        if has_matching_token(query_token,title_tokens):
            results.append(movie)
        if len(results) >= limit:
                break
        # if preprocessed_query in preprocessed_title:
            
    return results
def has_matching_token(query_tokens:list[str],title_tokens:list[str]):
    
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text
def tokenization(text:str):
    text = preprocess_text(text)
    tokens = text.split(" ")
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    stop_words = load_Stop_words()
    filtered_words = []
    for word in valid_tokens:
        if word not in stop_words:
            filtered_words.append(word)
    return filtered_words
