import json
# Load test data
# with open('test_data.json') as in_file:
#     test_data = json.load(in_file)
#
# titles = [item[0] for item in test_data['data']]
# documents = [item[1] for item in test_data['data']]
# f = open('text.txt','a')
# for i in range(0, len(titles))    :
#     f.write(titles[i] + ": " + documents[i] + "\n")
# from: https://scikit-learn.org/stable/modules/feature_extraction.html

# Đọc dữ liệu
from os import listdir
from os.path import isfile, join
from tika import parser
titles = [f for f in listdir("data") if isfile(join("data", f))]
documents = []
for file in titles:
    raw = parser.from_file('data/' + file)
    documents.append(raw['content'])

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords

nltk.download('punkt')
stop_words = set(stopwords.words('english'))
# love loving loves -> love
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer

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

# tokenizer('It was raining cats and dogs in FooBar')

# search_terms = 'Declan Rice (born 14 January 1999) is an English professional footballer who plays as a defensive midfielder or centre-back for Premier League club West Ham United and the England national team.'
search_terms = 'So, you want to finally discover how to be successful?. First, imagine where you’ll honestly be in the next five years.'
# search_terms =Born in England, Rice is of English and Irish descent. He had previously represented the Republic of Ireland internationally at both youth and senior levels, before switching allegiance to England in 2019.' 'sewing machine'

# Initialise TfidfVectorizer with the LemmaTokenizer. Also need to lemmatize the stop words as well
token_stop = tokenizer(' '.join(stop_words))
vectorizer = TfidfVectorizer(stop_words=token_stop, tokenizer=tokenizer)

# Calculate the word frequency, and calculate the cosine similarity of the search terms to the documents
vectors = vectorizer.fit_transform([search_terms] + documents) # Trich xuat dac trung.
cosine_similarities = linear_kernel(vectors, vectors[0:1]).flatten() #Tinh linear kerner.
document_scores = [item.item() for item in cosine_similarities[1:]]  # convert back to native Python dtypes

score_titles = [(score, title) for score, title in zip(document_scores, titles)]

for score, title in (sorted(score_titles, reverse=True, key=lambda x: x[0])[:10]):
    print(f'{score:0.3f} \t {title}')

    # get_feature.... -> dua tu index sang word. 258 co the duc ve Tomato.