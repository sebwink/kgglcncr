'''
This module contains preprocessing funtionalities meant
to be flexibly chained into larger preprocessing pipelines...

Preprocessing is understood here as taking text and producing
another text. Making word bags, etc. is considered to fall
into the responsibility of kgglcncr.features ...
'''

import re
import functools

import pandas as pd

import nltk

################################################################################
# Generic ######################################################################
################################################################################

class PreprocessingPipeline(object):
    def __init__(self, preprocessors, args=None):
        self.preprocessors = preprocessors
        if args is None:
            self.args = [[] for pp in preprocessors]
    def add(self, preprocessor, args=None):
        self.preprocessors.append(preprocessor)
        if args is None:
            args = []
        self.args.append(args)
    def __call__(self, data):
        for i, preprocessor in enumerate(self.preprocessors):
            args = [data]+self.args[i]
            data = preprocessor(*args)
        return data

# generic for text data ########################################################

class ModifiedTextDataGenerator(object):
    def __init__(self, generator, preprocessor):
        self.generator = generator
        self.preprocessor = preprocessor
    def __iter__(self):
        for ID, text in self.generator:
            yield ID, self.preprocessor(text)

def text_data_preprocessor(_preprocessor, text_data):
    if isinstance(text_data, pd.DataFrame):
        text_data['Text'] = text_data['Text'].apply(_preprocessor)
        return text_data
    else: # assuming text_data is generator yielding (ID, Text) tuples
        return ModifiedTextDataGenerator(text_data, _preprocessor)


################################################################################
# Citations ####################################################################
################################################################################

def remove_citations_from_text(text):
    '''
    This function removes likely citations from a Python string.

    May have unexpected (or expected) glitches...

    >>> text = 'asmn [1],[x et al.]. (1)(2), (7, 8)aaslkkj [1]aslj[2]sdö [999[sds]. Propanyl-(3,4)-melanase'
    >>> remove_citations_from_text(text)
    'asmn  , .   ,  aaslkkj  aslj sdö  . Propanyl- -melanase'
    '''
    text = re.sub('\([\s, \,,0-9]+\)', ' ',text)
    return re.sub('\[.*?\]', ' ', text)

def remove_citations(text_data):
    '''
    Remove putative citations of the form [1], (1,2),
    ..., [blabla et al. 0] etc. from text data.

    (Other citation formats not covered so far!!!
     Clash with fancy gene ids, metabolite names, ... ???)

    Args:
        text_data: challenge text data as imported by the functions
                   of the kgglcncr.data_import module

    Returns:
        Object of the same type as text_data with citations removed in text.

    '''
    return text_data_preprocessor(remove_citations_from_text, text_data)

################################################################################
# Case #########################################################################
################################################################################

def lower_case(text_data):
    return text_data_preprocessor(lambda text: text.lower(), text_data)

################################################################################
# Misc #########################################################################
################################################################################


def remove_patterns(text_data, pattern_list=None):

    class PatternRemover(object):
        default_patterns = ['.', ',', ':', '?', '!', '(', ')', ';', '[-,+]?[0-9]+']
        escape_patterns = ['.', ',', ':', '(', ')', '[', ']', '{', '}', '?', '!', '*']
        def __init__(self, pattern_list):
            if pattern_list is None:
                pattern_list = self.default_patterns
            self.pattern_list = pattern_list
        def __call__(self, text):
            pattern_list = ['\\'+pattern if pattern in self.escape_patterns else pattern
                                         for pattern in self.pattern_list]
            regex = '|'.join(pattern_list)
            return re.sub(regex, ' ', text)

    return text_data_preprocessor(PatternRemover(pattern_list), text_data)


def remove_words(text_data, word_list=None):

    class WordRemover(object):
        pass

    return text_data_preprocessor(WordRemover(word_list), text_data)
