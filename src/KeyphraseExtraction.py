from pke.unsupervised import TopicalPageRank as keypex
from Utils import *
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

def generate_keyphrases_files(inputfolder, outputfile) :
    clust_files = []
    for file in os.listdir(inputfolder) :
        if file.endswith(".txt") :
            clust_files.append(os.path.join(inputfolder, file))
    
    keyphrases_dict = dict()
    for fin in clust_files :
        basename = os.path.splitext(fin)[0]
        clustname = basename.split('/')[1]
        #outputfile = basename + "_keyphrases.txt"
        #fout = open(outputfile,'w')
        extractor = keypex(input_file=fin)
        keyphrases = extract_keyphrases(extractor)
        keyphrases_dict[clustname] = keyphrases
        #fout.write(str(keyphrases))
        #fout.close()
    
    fout = open(outputfile,'w')
    for clustname in keyphrases_dict :
        fout.write(str(clustname) + '\n' + str(keyphrases_dict[clustname]) + '\n\n')
    fout.close()
 
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
    dumpfile = "output_600data/new_clust_article_dump.pkl"
    dumpkeytokensfile = "output_600data/new_clust_keytokens_dump.pkl"
    folderpath = "documents/"
    outputfile = "output_600data/keyphrase_pke.txt"
    outputfile_fpm = "output_600data/keyphrase_pke_fpm.txt"

    #KEYPHRASE EXTRACTION BIASA
    load_documents(dumpfile, folderpath)
    generate_keyphrases_files(folderpath, outputfile)

    #KEYPHRASE EXTRACTION PAKE FPM
    load_documents_fpm(dumpfile,dumpkeytokensfile,folderpath)
    generate_keyphrases_files(folderpath, outputfile_fpm)

    logging.info('Finished.')
    
