# Not sure where to put these guys.  From py-sa but
# made so they can stand on their own with not many
# dependencies
from typing import List
from collections import defaultdict
from gensim import corpora, models, similarities
# import pprint

def create_ngrams(bow: List[str], ngram : int=3) -> List[str]:
    s = "$"+"$".join(bow)+"$"
    return [s[i:i+ngram] for i in range(len(s)-ngram+1)]

def create_corpus(docs, min_use : int =1):
    """
    Create a gensim corpus using specified ngram on given Bag-Of-Words
    return: list of used trigrams, corpus dictionary of used trigrams
    """
    frequency = defaultdict(int)
    for ngrams in docs:
        for token in ngrams:
            frequency[token] += 1

    # Create dictionary for tokens used over min_use times
    processed_corpus = \
        [[token for token in ngrams \
          if frequency[token] > min_use] \
         for ngrams in docs]
    # pprint.pprint(processed_corpus)
    dictionary = corpora.Dictionary(processed_corpus)
    bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
    return bow_corpus, dictionary

def create_ldasim(docs, ntopics : int = 64, min_use : int = 1):
    """
    Return lda model for given set docs for given number of topics
    """
    bow_corpus, dictionary = create_corpus(docs, min_use)
    lda = models.LdaModel(bow_corpus,
                          id2word=dictionary,
                          num_topics = ntopics)
    nf = len(dictionary.token2id)
    simindex = similarities.SparseMatrixSimilarity(lda[bow_corpus],
                                                   num_features=nf)
    return dictionary, lda, simindex
        
def lda_score(text, dictionary, lda, simindex, ngram : int=3):
    doc = create_ngrams(text.split(), ngram)
    vec_bow = dictionary.doc2bow(doc)
    vec_lda = lda[vec_bow]
    sims = simindex[vec_lda]
    return vec_lda, sims
