import json
import logging
from re import sub
from multiprocessing import cpu_count

import numpy as np

import gensim.downloader as api
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities import SoftCosineSimilarity

import logging

# Initialize logging.
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)  # DEBUG # INFO

stopwords = {'a','about','above','after','again','against','ain','all','am','an','and','any','are','aren',"aren't",'as','at','be','because','been','before','being','below','between','both','but','by','can','couldn',"couldn't",'d','did','didn',"didn't",'do','does','doesn',"doesn't",'doing','don',"don't",'down','during','each','few','for','from','further','had','hadn',"hadn't",'has','hasn',"hasn't",'have','haven',"haven't",'having','he','her','here','hers','herself','him','himself','his','how','i','if','in','into','is','isn',"isn't",'it',"it's",'its','itself','just','ll','m','ma','me','mightn',"mightn't",'more','most','mustn',"mustn't",'my','myself','needn',"needn't",'no','nor','not','now','o','of','off','on','once','only','or','other','our','ours','ourselves','out','over','own','re','s','same','shan',"shan't",'she',"she's",'should',"should've",'shouldn',"shouldn't",'so','some','such','t','than','that',"that'll",'the','their','theirs','them','themselves','then','there','these','they','this','those','through','to','too','under','until','up','ve','very','was','wasn',"wasn't",'we','were','weren',"weren't",'what','when','where','which','while','who','whom','why','will','with','won',"won't",'wouldn',"wouldn't",'y','you',"you'd","you'll","you're","you've",'your','yours','yourself','yourselves'}

# Support functions for pre-processing and calculation
# From: https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/soft_cosine_tutorial.ipynb

# Tiền xử lý dữ liệu
def preprocess(doc):
    # Tokenize, clean up input document string
    doc = sub(r'<img[^<>]+(>|$)', " image_token ", doc)
    doc = sub(r'<[^<>]+(>|$)', " ", doc)
    doc = sub(r'\[img_assist[^]]*?\]', " ", doc)
    doc = sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', " url_token ", doc)
    return [token for token in simple_preprocess(doc, min_len=0, max_len=float("inf")) if token not in stopwords]

# Load data -> có thể sử dụng MongoDb khi tệp dữ liệu lớn.
with open('test_data.json') as in_file:
    test_data = json.load(in_file)

titles = [item[0] for item in test_data['data']]
documents = [item[1] for item in test_data['data']]

query_string = 'i am going to play football AND play game'

# Tách các câu thành các từ sau khi đã xử lý và đã xoá các stopwords.
corpus = [preprocess(document) for document in documents]
query = preprocess(query_string)

# Download and/or load the GloVe word vector embeddings

if 'glove' not in locals():  # only load if not already in memory
    glove = api.load("glove-wiki-gigaword-50")

similarity_index = WordEmbeddingSimilarityIndex(glove)

# Build the term dictionary, TF-idf model
# The search query must be in the dictionary as well, in case the terms do not overlap with the documents (we still want similarity)
dictionary = Dictionary(corpus+[query])
tfidf = TfidfModel(dictionary=dictionary)

# Create the term similarity matrix.
# The nonzero_limit enforces sparsity by limiting the number of non-zero terms in each column.
# For my application, I got best results by removing the default value of 100
similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary, tfidf)  # , nonzero_limit=None)

# Compute Soft Cosine Measure between the query and the documents.
query_tf = tfidf[dictionary.doc2bow(query)]
# Ví dụ có 2 chữ play và going.
index = SoftCosineSimilarity(
            tfidf[[dictionary.doc2bow(document) for document in corpus]],
            similarity_matrix)

doc_similarity_scores = index[query_tf]

# Output the similarity scores for top 15 documents
sorted_indexes = np.argsort(doc_similarity_scores)[::-1]
for idx in sorted_indexes[:10]:
    print(f'{idx} \t {doc_similarity_scores[idx]:0.3f} \t {titles[idx]}')

### Find the most relevant terms in the documents
# For each term in the search query, what were the most similar words in each document?
doc_similar_terms = []
max_results_per_doc = 5
for term in query:
    idx1 = dictionary.token2id[term]
    for document in corpus:
        results_this_doc = []
        for word in set(document):
            idx2 = dictionary.token2id[word]
            score = similarity_matrix.matrix[idx1, idx2]
            if score > 0.0:
                results_this_doc.append((word, score))
        results_this_doc = sorted(results_this_doc, reverse=True, key=lambda x: x[1])  # sort results by score
        results_this_doc = results_this_doc[:min(len(results_this_doc), max_results_per_doc)]  # take the top results
        doc_similar_terms.append(results_this_doc)

# # Output the results for the top 10 documents
for idx in sorted_indexes[:10]:
    similar_terms_string = ', '.join([result[0] for result in doc_similar_terms[idx]])
    print(f'{idx} \t {doc_similarity_scores[idx]:0.3f} \t {titles[idx]}  :  {similar_terms_string}')

# love smile -> 0.2

