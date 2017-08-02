import numpy as np
import pandas as pd
import gensim

def gensim_tfidf_vectorizer(tokenized_text_data, dictionary, normalize=True, as_array=False, **kwargs):
    if isinstance(tokenized_text_data, pd.DataFrame):
        corpus = tokenized_text_data.loc[:, 'Text']
    else:
        corpus = tokenized_text_data
    model = gensim.models.TfidfModel(corpus, normalize=normalize, **kwargs)
    corpus = model[corpus]
    if as_array:
        return model, gensim.matutils.corpus2dense(corpus, num_docs=len(tokenized_text_data), num_terms=len(dictionary)).T
    else:
        return model, corpus

def gensim_topic_vectorizer(_model,
                            tokenized_text_data,
                            dictionary,
                            num_topics=100,
                            tfidf={'apply': True, 'normalize': True, 'kwargs': {}},
                            as_array=False,
                            **kwargs):

    if isinstance(tokenized_text_data, pd.DataFrame):
        corpus = tokenized_text_data.loc[:, 'Text']
    else:
        corpus = tokenized_text_data

    if tfidf['apply']:
        _, corpus = gensim_tfidf_vectorizer(tokenized_text_data, dictionary, tfidf['normalize'], False, **tfidf['kwargs'])
    model = _model(corpus, id2word=dictionary, num_topics=num_topics, **kwargs)
    corpus = model[corpus]
    if as_array:
        return model, gensim.matutils.corpus2dense(corpus, num_docs=len(tokenized_text_data), num_terms=num_topics).T
    else:
        return model, corpus

def gensim_lsi_vectorizer(tokenized_text_data, dictionary, num_topics=100, normalize_tfidf=True, as_array=False, **kwargs):
    tfidf = {'apply': True, 'normalize': normalize_tfidf, 'kwargs': {}}
    return gensim_topic_vectorizer(gensim.models.LsiModel, tokenized_text_data, dictionary, num_topics, tfidf, as_array, **kwargs)

def gensim_rp_vectorizer(tokenized_text_data, dictionary, num_topics=100, normalize_tfidf=True, as_array=False):
    tfidf = {'apply': True, 'normalize': normalize_tfidf, 'kwargs': {}}
    return gensim_topic_vectorizer(gensim.models.RpModel, tokenized_text_data, dictionary, num_topics, tfidf, as_array, **kwargs)

def gensim_lda_vectorizer(tokenized_text_data, dictionary, num_topics=100, as_array=False):
    tfidf = {'apply': False, 'normalize': None, 'kwargs': {}}
    return gensim_topic_vectorizer(gensim.models.LdaModel, tokenized_text_data, dictionary, num_topics, tfidf, as_array, **kwargs)

def gensim_hdp_vectorizer(tokenized_text_data, dictionary, num_topics=100, as_array=False):
    tfidf = {'apply': False, 'normalize': None, 'kwargs': {}}
    return gensim_topic_vectorizer(gensim.models.HdpModel, tokenized_text_data, dictionary, num_topics, tfidf, as_array, **kwargs)
