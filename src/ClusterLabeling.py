"""Python implementation of the TextRank algoritm.

From this paper:
    https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf

Based on:
    https://gist.github.com/voidfiles/1646117
    https://github.com/davidadamojr/TextRank
"""
import logging, warnings
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import io
import itertools
import networkx as nx
import nltk
import os
from gensim import models
import numpy as np
import scipy as sp
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from Utils import *


def setup_environment():
    """Download required resources."""
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    print('Completed resource downloads.')

def filter_for_tags(tagged, tags=['NN', 'JJ', 'NNP']):
    """Apply syntactic filters based on POS tags."""
    return [item for item in tagged if item[1] in tags]

def normalize(tagged):
    """Return a list of tuples with the first item's periods removed."""
    return [(item[0].replace('.', ''), item[1]) for item in tagged]

def unique_everseen(iterable, key=None):
    """List unique elements in order of appearance.

    Examples:
        unique_everseen('AAAABBBCCDAABBB') --> A B C D
        unique_everseen('ABBCcAD', str.lower) --> A B C D
    """
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in [x for x in iterable if x not in seen]:
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element

def levenshtein_distance(first, second):
    """Return the Levenshtein distance between two strings.

    Based on:
        http://rosettacode.org/wiki/Levenshtein_distance#Python
    """
    if len(first) > len(second):
        first, second = second, first
    distances = range(len(first) + 1)
    for index2, char2 in enumerate(second):
        new_distances = [index2 + 1]
        for index1, char1 in enumerate(first):
            if char1 == char2:
                new_distances.append(distances[index1])
            else:
                new_distances.append(1 + min((distances[index1],
                                             distances[index1 + 1],
                                             new_distances[-1])))
        distances = new_distances
    return distances[-1]


def word2vec_similarity(first, second, w2v_model) :
    words = list(w2v_model.wv.vocab)

    if (first in words) and (second in words) :
        word_sim = w2v_model.wv.similarity(first, second)
    else :
        word_sim = 0
    return word_sim

def word2vec_distance(first, second, w2v_model) :
    words = list(w2v_model.wv.vocab)

    if (first in words) and (second in words) :
        word_dist = 1 - (w2v_model.wv.similarity(first, second))
    else :
        word_dist = 1
    return word_dist



def build_graph(nodes, model):
    """Return a networkx graph instance.

    :param nodes: List of hashables that represent the nodes of a graph.
    :param mode: Word2Vec model to calculate word similarity for edges weight
    """
    gr = nx.Graph()  # initialize an undirected graph
    gr.add_nodes_from(nodes)
    nodePairs = list(itertools.combinations(nodes, 2))

    # add edges to the graph (weighted by word2vec distance)
    for pair in tqdm(nodePairs, leave=False):
        firstString = pair[0]
        secondString = pair[1]
        #levDistance = levenshtein_distance(firstString, secondString)
        word2vecSim = word2vec_distance(firstString, secondString, model)
        #word2vecSim = word2vec_similarity(firstString, secondString, model)
        gr.add_edge(firstString, secondString, weight=word2vecSim)

    return gr

def build_graph_cooccurence(nodes, sentences, win) :
    """Generate word co-occurence count matrix, with windows n words.

    Parameter
    nodes : list
        List of (unique) words extracted from text source.
    sentences : list
        List of sentences extracted from text source. the sentences were
        preprocessed beforehand.
    n : integer
        Number of windows range of co-occurence.

    Return
    Gtext : networkx graph
        Graph generated from text
    """
    
    #Construct the graph
    Gtext = nx.Graph()
    Gtext.add_nodes_from(nodes)

    #generate co-occurence relation on graph
    #iteration on list of sentences
    for sen in sentences :
        sen_split = sen.split(' ')
        #iteration on list of words in a sentence
        for word_index in range(len(sen_split)) :
            word = sen_split[word_index]
            #if word not in nodes : #check if the sentence word is part of the nodes
            #    continue
            #iteration on neighbouring words, left & right
            #right neighbors
            for i in range(1,win) :
                neighword_index = word_index + i
                if neighword_index < len(sen_split) :
                    neighword = sen_split[neighword_index]
                    #if neighword not in nodes : #check if the sentence neighbour word is part of the nodes
                    #    continue
                    if (word,neighword) not in list(Gtext.edges()) :
                        Gtext.add_edge(word, neighword, weight=0.0)
                    #count matrix increment, for un-directed graph
                    Gtext[word][neighword]['weight'] += 1.0

    return Gtext



def extract_key_phrases(word_tokens, model):
    """Return a set of key phrases.

    :param text: A string.
    """

    # assign POS tags to the words in the text
    tagged = nltk.pos_tag(word_tokens)
    textlist = [x[0] for x in tagged]

    tagged = filter_for_tags(tagged)
    tagged = normalize(tagged)

    unique_word_set = unique_everseen([x[0] for x in tagged])
    word_set_list = list(unique_word_set)

    # this will be used to determine adjacent words in order to construct
    # keyphrases with two words

    graph = build_graph(word_set_list, model)

    # pageRank - initial value of 1.0, error tolerance of 0,0001,
    calculated_page_rank = nx.pagerank(graph, weight='weight')

    # most important words in ascending order of importance
    keyphrases = sorted(calculated_page_rank, key=calculated_page_rank.get,
                        reverse=True)

    # the number of keyphrases returned will be relative to the size of the
    # text (a fourth of the number of vertices)
    top_phrase = len(word_set_list) // 3
    keyphrases = keyphrases[0:top_phrase + 1]

    '''
    # take keyphrases with multiple words into consideration as done in the
    # paper - if two words are adjacent in the text and are selected as
    # keywords, join them together
    modified_key_phrases = set([])
    # keeps track of individual keywords that have been joined to form a
    # keyphrase
    dealt_with = set([])
    i = 0
    j = 1
    while j < len(textlist):
        first = textlist[i]
        second = textlist[j]
        if first in keyphrases and second in keyphrases:
            keyphrase = first + ' ' + second
            modified_key_phrases.add(keyphrase)
            dealt_with.add(first)
            dealt_with.add(second)
        else:
            if first in keyphrases and first not in dealt_with:
                modified_key_phrases.add(first)

            # if this is the last word in the text, and it is a keyword, it
            # definitely has no chance of being a keyphrase at this point
            if j == len(textlist) - 1 and second in keyphrases and \
                    second not in dealt_with:
                modified_key_phrases.add(second)

        i = i + 1
        j = j + 1
    '''
    return keyphrases

def extract_key_phrases_v2(word_set_list, model):
    """Return a set of key phrases.

    :param text: A string.
    """
    # this will be used to determine adjacent words in order to construct
    # keyphrases with two words

    graph = build_graph(word_set_list, model)

    # pageRank - initial value of 1.0, error tolerance of 0,0001,
    calculated_page_rank = nx.pagerank(graph, weight='weight')

    # most important words in ascending order of importance
    keyphrases = sorted(calculated_page_rank, key=calculated_page_rank.get,
                        reverse=True)

    return keyphrases

def extract_key_phrases_cooccurence(nodes, sentences, win) :
    """Return a set of key phrases.

    :param text: A string.
    """
    # this will be used to determine adjacent words in order to construct
    # keyphrases with two words

    graph = build_graph_cooccurence(nodes, sentences, win)

    # pageRank - initial value of 1.0, error tolerance of 0,0001,
    calculated_page_rank = nx.pagerank(graph, weight='weight')

    # most important words in ascending order of importance
    keyphrases = sorted(calculated_page_rank, key=calculated_page_rank.get,
                        reverse=True)

    return keyphrases



def cluster_labeling(clust_text, clust_phrases, w2v_model) :
    logging.info('Cluster labeling.....')
    clust_keyphrases = []
    clust_keywords = []

    for text in tqdm(clust_text, leave=False) :
        keywords = extract_key_phrases(text, w2v_model)
        clust_keywords.append(keywords)
    
    for i in range(len(clust_phrases)) :
        temp_phrases = []
        clust_keyphrases.append([])
        for phrase in clust_phrases[i] :
            phrase_tokens = phrase.split(' ')
            if (set(phrase_tokens) < set(clust_keywords[i])) and (phrase not in clust_keyphrases[i]) :
                clust_keyphrases[i].append(phrase)

    return clust_keywords, clust_keyphrases

def cluster_labeling_v2(clust_text, clust_phrases, w2v_model, max_phrase) :
    logging.info('Cluster labeling.....')
    clust_keyphrases = []
    clust_keywords = []

    #IF THE INPUT CLUST_ARTICLES_TOKENIZED
    clust_words = []
    for clust_index in range(len(clust_text)) :
        clust_words.append([])
        for article_tokens in clust_text[clust_index] :
            clust_words[clust_index].extend(article_tokens)
        clust_words[clust_index] = list(set(clust_words[clust_index]))

    #CHANGE THE ITERABLES DEPENDING ON INPUT
    for text in tqdm(clust_words, leave=False) :
        keywords = extract_key_phrases_v2(text, w2v_model)
        clust_keywords.append(keywords)
    
    for i in range(len(clust_phrases)) :
        temp_phrases = []
        clust_keyphrases.append([])
        temp_keywords = []
        n = 0
        while (True) :
            if len(clust_keywords[i]) == 0 :
                break
            temp_keywords.append(clust_keywords[i][n])
            for phrase in clust_phrases[i] :
                phrase_tokens = phrase.split(' ')
                if (len(clust_keyphrases[i]) < max_phrase) and \
                    (set(phrase_tokens) < set(temp_keywords)) and (phrase not in clust_keyphrases[i]) :
                    clust_keyphrases[i].append(phrase)
            n += 1
            if (n>len(clust_keywords[i]) - 1) :
                break
    
    return clust_keywords, clust_keyphrases

def cluster_labeling_cooccurence(clust_words, clust_phrases, clust_contents, max_phrase) :
    logging.info('Cluster labeling.....')
    clust_keyphrases = []
    clust_keywords = []

    for i in tqdm(range(len(clust_words)), leave=False) :
        words = clust_words[i]
        contents = clust_contents[i]
        #MERGE ALL CONTENTS IN A CLUSTER
        all_contents = ' '.join(contents)
        #EXTRACT SENTENCES IN A CLUSTER
        sentences = extract_sentences(all_contents)
        sentences = [preprocess_text(sen) for sen in tqdm(sentences, leave=False)]

        win = 2
        keywords = extract_key_phrases_cooccurence(words, sentences, win)
        clust_keywords.append(keywords)
    
    for i in tqdm(range(len(clust_phrases)), leave=False) :
        temp_phrases = []
        clust_keyphrases.append([])
        temp_keywords = []
        n = 0
        while (True) :
            if len(clust_keywords[i]) == 0 :
                break
            temp_keywords.append(clust_keywords[i][n])
            for phrase in clust_phrases[i] :
                phrase_tokens = phrase.split(' ')
                if (len(clust_keyphrases[i]) < max_phrase) and \
                    (set(phrase_tokens) < set(temp_keywords)) and (phrase not in clust_keyphrases[i]) :
                    clust_keyphrases[i].append(phrase)
            n += 1
            if (n>len(clust_keywords[i]) - 1) :
                break
    
    return clust_keywords, clust_keyphrases




def extract_sentences(text) :
    """Split text into sentences.

    Parameter
    text : string
        text source (multi-sentences) that will be splitted into list of sentences.
    
    Return
    sentences : list
        list of sentence (raw) extracted from the text.

    Note : Failed on abbreviation, i.e. "U.S.S. Callister"
    """
    # removes all 'lines' in the file
    sentences = re.sub(r'\n', ' ', text)

    # classes any period after Mr/Mrs/Dr as ('not sentence boundaries')
    sentences = re.sub(r'(?<!Mr)(?<!Mrs)(?<!Dr)\.\s([A-Z])', r'.\n\1', sentences)

    # creates new line after exclaimation mark
    sentences = re.sub(r'!\s', '!\n', sentences)

    # creates nenw line after question mark
    sentences = re.sub(r'\?\s', '?\n', sentences)

    sentences = sentences.split('\n')

    return sentences

def extract_imp_sentences(text, summary_length=100, clean_sentences=False, language='english'):
    """Return a paragraph formatted summary of the source text.

    :param text: A string.
    """
    sent_detector = nltk.data.load('tokenizers/punkt/'+language+'.pickle')
    sentence_tokens = sent_detector.tokenize(text.strip())
    graph = build_graph(sentence_tokens)

    calculated_page_rank = nx.pagerank(graph, weight='weight')

    # most important sentences in ascending order of importance
    sentences = sorted(calculated_page_rank, key=calculated_page_rank.get,
                       reverse=True)

    # return a 100 word summary
    summary = ' '.join(sentences)
    summary_words = summary.split()
    summary_words = summary_words[0:summary_length]
    dot_indices = [idx for idx, word in enumerate(summary_words) if word.find('.') != -1]
    if clean_sentences and dot_indices:
        last_dot = max(dot_indices) + 1
        summary = ' '.join(summary_words[0:last_dot])
    else:
        summary = ' '.join(summary_words)

    return summary

def write_files(summary, key_phrases, filename):
    """Write key phrases and summaries to a file."""
    print("Generating output to " + 'keywords/' + filename)
    key_phrase_file = io.open('keywords/' + filename, 'w')
    for key_phrase in key_phrases:
        key_phrase_file.write(key_phrase + '\n')
    key_phrase_file.close()

    print("Generating output to " + 'summaries/' + filename)
    summary_file = io.open('summaries/' + filename, 'w')
    summary_file.write(summary)
    summary_file.close()

    print("-")

def summarize_all():
    # retrieve each of the articles
    articles = os.listdir("articles")
    for article in articles:
        print('Reading articles/' + article)
        article_file = io.open('articles/' + article, 'r')
        text = article_file.read()
        keyphrases = extract_key_phrases(text)
        summary = extract_sentences(text)
        write_files(summary, keyphrases, article)


def cluster_word_filter(documents, min_avg_count) :
    '''Filtering words from a document cluster
    based on average count between documents.
    
    Parameter
    documents : list
        list of document text
    min_avg_count : float
        minimal count average of a word

    Return
    big_text : string
        a string containing "important" words.
    '''
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(documents)
    words = vectorizer.get_feature_names()
    print ("Words before filtering : ", len(words))
    count_matrix = np.array(count_matrix.todense())
    avg_count = np.mean(count_matrix, axis=0)
    imp_words = [words[i] for i in range(len(words)) if avg_count[i] > min_avg_count]
    print ("Words after filtering : ", len(imp_words))
    big_text = " ".join(imp_words)
    return big_text

