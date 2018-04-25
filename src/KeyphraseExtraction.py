from pke.unsupervised import TopicalPageRank as keypex
from palmettopy.palmetto import Palmetto
from Utils import *
import re
import os

def extract_keyphrases(extractor) :
    
    # create a TopicRank extractor and set the input language to English (used for
    # the stoplist in the candidate selection method)
    #extractor = TopicRank(input_file=inputfile)

    # load the content of the document, here in CoreNLP XML format
    # the use_lemmas parameter allows to choose using CoreNLP lemmas or stems 
    # computed using nltk
    extractor.read_document(format='raw', stemmer=None)

    # select the keyphrase candidates, for TopicRank the longest sequences of 
    # nouns and adjectives
    #extractor.candidate_selection(pos=['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR',
    #                                'JJS'])
    #extractor.candidate_selection(lasf=2,cutoff=100)
    extractor.candidate_selection()

    # weight the candidates using a random walk. The threshold parameter sets the
    # minimum similarity for clustering, and the method parameter defines the 
    # linkage method
    #extractor.candidate_weighting(threshold=0.74,
    #                            method='average')
    extractor.candidate_weighting()

    #FOR SUPERVISED KEYPHRASE EXTRACTION
    #extractor.feature_extraction()

    # print the n-highest (10) scored candidates
    return [u for u, v in extractor.get_n_best(n=5)]

def natural_keys(text):
    atoi = lambda c : int(c) if c.isdigit() else c
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def generate_keyphrases_files(inputfolder, outputfile) :
    clust_files = []
    for file in os.listdir(inputfolder) :
        if file.endswith(".txt") :
            clust_files.append(os.path.join(inputfolder, file))
    
    keyphrases_list = []
    clust_files.sort(key=natural_keys)

    for fin in clust_files :
        basename = os.path.splitext(fin)[0]
        print(basename)
        clustname = basename.split('/')[1]
        extractor = keypex(input_file=fin)
        keyphrases = extract_keyphrases(extractor)
        keyphrases_list.append(keyphrases)
        print(topic_coherence(keyphrases))
    
    fout = open(outputfile,'w')
    for clustno in range(len(keyphrases_list)) :
        fout.write('Cluster' + str(clustno + 1) + '\n' + str(keyphrases_list[clustno]) + '\n\n')
    fout.close()

def topic_coherence(keyphrases) :

    top_words = []
    for phrase in keyphrases :
        clean_phrase = preprocess_text(phrase)
        words = clean_phrase.split(' ')
        top_words.extend(words)
    
    top_words = list(set(top_words))
    print(top_words)
    palmetto = Palmetto()
    score = palmetto.get_coherence(top_words)
    return score

def clear_txt(folderpath) :
    txtfiles = []
    for file in os.listdir(folderpath):
        if file.endswith(".txt"):
            txtfiles.append(os.path.join(folderpath, file))
    for f in txtfiles :
        os.remove(f)

def load_documents(dumpfile, outfolder) :
    unpack = load_from_pickle(dumpfile)
    clust_articles_id, clust_articles_tokenized, clust_articles_content = unpack[0], unpack[1], unpack[2]
    clear_txt(outfolder)

    for i in range(len(clust_articles_content)) :
        filename = outfolder + "Cluster" + str(i+1) + ".txt"
        f = open(filename,'w')
        for doc in clust_articles_content[i] :
            text = doc
            try :
                f.write(text + "\n")
            except UnicodeEncodeError :
                f.write(text.encode('utf-8') + b"n")
        f.close()

def load_documents_fpm(dumpfile, dumpkeytokensfile, outfolder) :
    unpack1 = load_from_pickle(dumpfile)
    clust_articles_id, clust_articles_tokenized = unpack1[0], unpack1[1]
    unpack2 = load_from_pickle(dumpkeytokensfile)
    clust_words = unpack2[0] 
    clear_txt(outfolder)

    for i in range(len(clust_articles_tokenized)) :
        filename = outfolder + "Cluster" + str(i+1) + ".txt"
        f = open(filename,'w')
        #KALO MAU FPM DILAKUKAN DISINI
        word_keep_list = clust_words[i] 
        for doc in clust_articles_tokenized[i] :
            filtered_word_tokens = [w for w in doc if w in word_keep_list]
            text = ' '.join(filtered_word_tokens)
            f.write(text + "\n")
        f.close()

if __name__ == "__main__" :
    db_name = 'article550'
    dumpfile = "output_" + db_name + "/clust_article_dump.pkl"
    folderpath = "documents/"
    outputfile = "output_" + db_name + "/keyphrase_pke.txt"
    
    #KEYPHRASE EXTRACTION BIASA
    load_documents(dumpfile, folderpath)
    generate_keyphrases_files(folderpath, outputfile)
    
    logging.info('Finished.')
    
