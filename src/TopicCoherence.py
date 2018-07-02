from palmettopy.palmetto import Palmetto
from palmettopy.exceptions import EndpointDown
from Utils import *

def coherence_v(keyphrases) :
    top_words = []
    for phrase in keyphrases :
        clean_phrase = preprocess_text(phrase)
        words = clean_phrase.split(' ')
        top_words.extend(words)
    
    top_words = list(set(top_words))
    
    #CEK KOHERENSI TOPIK PER FRASE
    top_words = keyphrases
    
    #print(top_words)
    #print('n before : ', len(top_words))
    flag = 0
    while(flag == 0) :
        try :
            palmetto = Palmetto()
            score = palmetto.get_coherence(top_words)
            flag = 1
            #print ('n after : ', len(top_words))
        except EndpointDown:
            top_words = top_words[:len(top_words)-1]
    return score

if __name__ == "__main__" :
    palmetto = Palmetto()
    #words = [u'real', u'task', u'algorithm', u'tasks', u'dynamic', u'periodic', u'systems', u'time', u'scheduling', u'problem', u'model']
    words = [u'real', u'task', u'algorithm', u'tasks', u'dynamic']
    words = [u'real', u'task', u'algorithm', u'tasks', u'dynamic', u'periodic', u'systems']
    words = [u'real', u'task', u'algorithm', u'tasks', u'dynamic', u'periodic', u'systems', u'time']
    words = [u'real', u'task', u'algorithm', u'tasks', u'dynamic', u'periodic', u'systems', u'time', u'scheduling']
    words = [u'real', u'task', u'algorithm', u'tasks', u'dynamic', u'periodic', u'systems', u'time', u'scheduling', u'problem']
    words = [u'real', u'task', u'algorithm', u'tasks', u'dynamic', u'periodic', u'systems', u'time', u'scheduling', u'problem','dice','hen']
    phrases = ['wireless network','network performance']
    score = palmetto.get_coherence(phrases)
    score = round(score,3)
    print (score)