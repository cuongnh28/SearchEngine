from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
doc1 = "good boy"
doc2 = "good girl"
doc3 = "boy girl good"
class LemmaTokenizer:
    """
    Interface to the WordNet lemmatizer from nltk
    """
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`']
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in self.ignore_tokens]

# Demonstrate the job of the tokenizer

tokenizer=LemmaTokenizer()

stop_words = set(stopwords.words('english'))
token_stop = tokenizer(' '.join(stop_words))
vectorizer = TfidfVectorizer(stop_words=token_stop, tokenizer=tokenizer)
vectors = vectorizer.fit_transform([doc1] + [doc2] + [doc3])
print(vectorizer.idf_)
print(vectorizer.get_feature_names())
print(vectors)

