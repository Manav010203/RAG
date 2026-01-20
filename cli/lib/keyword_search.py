import os
import pickle
import string
import math
import operator
from collections import defaultdict,Counter

from nltk.stem import PorterStemmer

from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    load_stopwords,
    BM25_B,
    BM25_k1
)


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies = defaultdict(Counter)
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR,"term_frequencies.pkl")
        self.doc_lengths = {}
        self.doc_lengths_path = os.path.join(CACHE_DIR,"doc_lengths.pkl")
    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            doc_description = f"{m['title']} {m['description']}"
            self.docmap[doc_id] = m
            self.__add_document(doc_id, doc_description)

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(dict(self.index), f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(dict(self.term_frequencies),f)
        with open(self.doc_lengths_path,"wb") as f:
            pickle.dump(dict(self.doc_lengths),f)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        self.doc_lengths[doc_id] = len(tokens)
        for token in tokens:
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token]+=1
    def load(self)->None:
        # if not os.path.exists(self.index_path) or not os.path.exists(self.docmap_path) or not os.path.exists(self.term_frequencies_path):
        #     raise FileNotFoundError("Index or docmap or  not existed")
        with open(self.index_path, "rb") as f:
            # self.index = pickle.load(f)
            self.index = defaultdict(set, pickle.load(f))

        with open(self.docmap_path,"rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path,"rb") as f:
            self.term_frequencies = defaultdict(Counter,pickle.load(f))
        with open(self.doc_lengths_path,"rb") as f:
            self.doc_lengths = pickle.load(f)
    def get_tf(self,doc_id,term):
        tokens = tokenize_text(term)
        if len(tokens)!=1:
            raise ValueError("get_tf except only one token at a time")
        token = tokens[0]
        # return self.term_frequencies[doc_id][token]
        return self.term_frequencies.get(doc_id, Counter()).get(token, 0)
    def get_idf(self,term):
        tokens = tokenize_text(term)
        if(len(tokens)!=1):
            raise ValueError("term is big in length")
        token = tokens[0]
        occurence = 0
        for id in self.docmap:
            if self.term_frequencies[id][token]:
                occurence+=1
        return occurence
    def get_bm25_idf(self,term:str):
        tokens = tokenize_text(term)
        if len(tokens)!=1:
            raise ValueError("only single term required")
        token = tokens[0]
        N = len(self.docmap)
        df = 0
        for id in self.docmap:
            if self.term_frequencies[id][token]:
                df+=1
        bm25_idf = math.log((N-df+0.5)/(df+0.5)+1)
        return bm25_idf
    def get_bm25_tf(self,doc_id,term,K1: float = BM25_k1, b: float = BM25_B):
        freq = self.get_tf(doc_id,term)
        len_norm = 1-b +b*(self.doc_lengths[doc_id]/self.__get_avg_doc_length())
        res = (freq * (K1 +1)) / (freq + K1*len_norm)
        return res
    # def doc_lengths(seldd)
    def __get_avg_doc_length(self)->float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)+1
    def bm25(self,doc_id,term):
        bm25_tf = self.get_bm25_tf(doc_id,term,BM25_k1,BM25_B)
        bm25_idf = self.get_bm25_idf(term)
        bm25_score = bm25_idf * bm25_tf
        return bm25_score
    def bm25_search(self,query,limit):
        tokens = tokenize_text(query)
        scores = defaultdict(float)
        for doc_id in self.docmap:
            total_score = 0.0
            for token in tokens:
                tf_score = self.get_bm25_tf(doc_id,token)
                idf_score = self.get_bm25_idf(token)
                total_score += tf_score * idf_score
            if total_score > 0:
                scores[doc_id] = total_score
        ranked = sorted(scores.items(),key=lambda x : x[1],reverse=True)
        return ranked[:limit]
def bm25_tf_command(doc_id,term,k1=BM25_k1,b=BM25_B):
    index = InvertedIndex()
    index.load()
    res = index.get_bm25_tf(doc_id,term,k1,b)
    return res

def bm25_idf_command(term:str)-> None:
    index = InvertedIndex()
    index.load()
    res = index.get_bm25_idf(term)
    return res 
def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()
    # docs = idx.get_documents("")
    # print(f"First document for token 'merida' = {docs[0]}")

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    query_tokens = tokenize_text(query)
    seen, results = set(), []
    for query_token in query_tokens:
        matching_doc_ids = idx.get_documents(query_token)
        for doc_id in matching_doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = idx.docmap[doc_id]
            results.append(doc)
            if len(results) >= limit:
                return results

    return results


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    stop_words = load_stopwords()
    filtered_words = []
    for word in valid_tokens:
        if word not in stop_words:
            filtered_words.append(word)
    stemmer = PorterStemmer()
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words
