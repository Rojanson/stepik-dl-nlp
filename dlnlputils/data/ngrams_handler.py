import collections
import re
from nltk.util import ngrams, bigrams

import numpy as np

TOKEN_RE = re.compile(r'[\w\d]+')


def ngram_text_simple_regex(txt, ngram_size=2, t_re=TOKEN_RE):

    all_tokens = t_re.findall(txt)

    all_ngrams = []

    for word in all_tokens:
        if ngram_size == 2:
            ngrams_ = bigrams(word)
        else:
            ngrams_ = ngrams(word, ngram_size)

        for ngram in ngrams_:
            all_ngrams.append(''.join(ngram))

    return all_ngrams


def ngram_corpus(texts, ngrammer=ngram_text_simple_regex, **tokenizer_kwargs):
    return [ngrammer(text, **tokenizer_kwargs) for text in texts]
