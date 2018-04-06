import mysql.connector
from Utils import *
from FrequentPhraseMining import *
from tqdm import tqdm
import itertools

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

def lev_ratio(first, second) :
    max_length = max(len(first), len(second))
    l_dist = levenshtein_distance(first, second)
    l_ratio = l_dist / float(max_length)
    return l_ratio

if __name__ == "__main__" :
    articles_id = []
    articles_text = []
    connA = mysql.connector.connect(user='root', password='admin', host='127.0.0.1', database='article')
    fout = open('output/article_label.txt','w')
    curA = connA.cursor(buffered=True)
    select_article = "SELECT article_id, article_title, article_abstract " + \
                     "FROM fg_m_article WHERE article_abstract != '' ORDER BY article_id"
    curA.execute(select_article)
    
    for (art_id, art_title, art_abstract) in curA :
        articles_id.append(art_id)
        articles_text.append(art_title + " " + art_abstract)
    
    #articles_tokens = [preprocess_text(a).split(' ') for a in tqdm(articles_text)]
    #SAVE FILE TO PICKLE
    #save_to_pickle('output/article_label_dump.pkl', articles_id, articles_tokens)
    #LOAD DATA FROM PICkLE
    unpack = load_from_pickle('output/article_label_dump.pkl')
    articles_id, articles_tokens = unpack[0], unpack[1]

    connB = mysql.connector.connect(user='root', password='admin', host='127.0.0.1', database='dblp_sigweb')
    curB = connB.cursor(buffered=True)
    select_tag = "select distinct author_tag from dblp_sigweb.author_tags"
    curB.execute(select_tag)
    tags = [t[0] for t in curB]

    ex_num = 5
    for i in tqdm(range(ex_num)) :
        docs_tokens = []
        docs_tokens.append(articles_tokens[i])
        phrase_count = extract_phrases_v2(docs_tokens, 2)
        phrases = [p for p in phrase_count]
        words = []
        for phrase in phrases :
            word_tokens = phrase.split(' ')
            for w in word_tokens :
                if w not in words :
                    words.append(w)
        
        match_tags = []
        '''
        pairs = itertools.product(phrases, tags)
        for p in tqdm(pairs, leave=False) :
            l_ratio = lev_ratio(p[0],p[1])
            if (l_ratio <= 0.25) :
                match_tags.append(p[1])
        '''
        for t in tags :
            t_tokens = t.split(' ')
            if (set(t_tokens) < set(words) and (t not in match_tags)) :
                match_tags.append(t)

        fout.write("article-" + str(i+1) + "\n")
        fout.write("TEXT :\n")
        fout.write(str(articles_text[i]) + "\n")
        fout.write("PHRASES :\n")
        fout.write(str(phrase_count) + "\n")
        fout.write("MATCH TAGS :\n")
        fout.write(str(match_tags) + "\n")

    fout.close()
    connB.close()
    connA.close()
