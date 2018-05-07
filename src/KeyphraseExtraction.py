from pke.unsupervised import TopicalPageRank as keypex
from tqdm import tqdm
from Utils import *
from TopicCoherence import coherence_v
import csv
import re
import os

#GLOBAL
db_name = 'article550'
input_folder = 'output_' + db_name + '/article_dumps/'
coherence_loop_output = 'output_' + db_name + '/tcoherence_loop.csv'
clustermapfile = 'output_' + db_name + '/cluster_mapping.csv'
max_phrase = 5

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
    #extractor.candidate_weighting(threshold=0.5,
    #                            method='average')
    extractor.candidate_weighting()

    #FOR SUPERVISED KEYPHRASE EXTRACTION
    #extractor.feature_extraction()

    # print the n-highest (10) scored candidates
    return [u for u, v in extractor.get_n_best(n=max_phrase)]

def natural_keys(text):
    atoi = lambda c : int(c) if c.isdigit() else c
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def generate_keyphrases_files(inputfolder, outputfile) :
    clust_files = []
    for file in os.listdir(inputfolder) :
        if file.endswith(".txt") :
            clust_files.append(os.path.join(inputfolder, file))
    
    keyphrases_list = []
    tc_scores = []
    clust_files.sort(key=natural_keys)

    for fin in tqdm(clust_files, leave=False) :
        basename = os.path.splitext(fin)[0]
        clustname = basename.split('/')[1]
        extractor = keypex(input_file=fin)
        keyphrases = extract_keyphrases(extractor)
        keyphrases_list.append(keyphrases)
        tc_scores.append(coherence_v(keyphrases))
    
    fout = open(outputfile,'w')
    for i in range(len(keyphrases_list)) :
        fout.write('Cluster' + str(i + 1) + '\n' + str(keyphrases_list[i]) + \
                    '\n'+ 'Coherence : ' + str(tc_scores[i]) +'\n\n')
    
    avg_tc_scores = sum(tc_scores)/float(len(tc_scores))
    fout.write('Mean Topic Coherence : ' + str(avg_tc_scores))

    fout.close()

    return keyphrases_list, tc_scores

def load_documents(dumpfile, outfolder) :
    unpack = load_from_pickle(dumpfile)
    clust_articles_id, clust_articles_tokenized, clust_articles_content = unpack[0], unpack[1], unpack[2]
    clear_folder(outfolder,'txt')

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
    clear_folder(outfolder,'txt')

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

def clustermapping_generate() :
    clustmap = dict()
    with open(clustermapfile) as csvfile :
        rowreader = csv.reader(csvfile, delimiter=';')
        for row in rowreader :
            if row[0].isdigit() :
                new_clust = int(row[0])
                old_clust = int(row[1])
                if new_clust not in clustmap :
                    clustmap[new_clust] = []
                clustmap[new_clust].append(old_clust)
    return clustmap

def output_topic_coherence(outputfile, keyphrases_list_ori, keyphrases_list_merged, 
                           tc_scores_ori, tc_scores_merged) :
    clustmap = clustermapping_generate()
    sorted(clustmap)
    merged_ori_cluster_tscores = []
    with open(outputfile, 'wb') as csvfile :
        csvwriter = csv.writer(csvfile, delimiter=';')
        csvwriter.writerow(['New Cluster','','Coherence','Original Cluster','','Coherence'])
        for new_clust in clustmap :
            n_old_clusts = len(clustmap[new_clust])
            old_clusts_tcscores = [tc_scores_ori[x-1] for x in clustmap[new_clust]]
            avg_tscores_ori = sum(old_clusts_tcscores) / float(len(old_clusts_tcscores))
            merged_ori_cluster_tscores.append(avg_tscores_ori)
            counter = 0
            for old_clust in clustmap[new_clust] :
                if counter == 0 :
                    csvwriter.writerow([str(new_clust),str(keyphrases_list_merged[new_clust-1]),
                                        str(tc_scores_merged[new_clust-1]),
                                        str(old_clust), str(keyphrases_list_ori[old_clust-1]),
                                        str(tc_scores_ori[old_clust-1]),
                                        str(avg_tscores_ori)])
                    counter += 1
                else :
                    csvwriter.writerow(['','','',
                                        str(old_clust),str(keyphrases_list_ori[old_clust-1]),
                                        str(tc_scores_ori[old_clust-1])])
        avg_merged = sum(tc_scores_merged) / float(len(tc_scores_merged))
        avg_ori = sum(tc_scores_ori) / float(len(tc_scores_ori))
        avg_merged_ori = sum(merged_ori_cluster_tscores) / float(len(merged_ori_cluster_tscores))
        csvwriter.writerow(['','',avg_merged,'','',avg_ori, avg_merged_ori])
        print 'avg_merged : ', avg_merged
        print 'avg_ori : ', avg_ori
        print 'avg_merged_ori : ', avg_merged_ori

        return avg_merged, avg_ori, avg_merged_ori

def main(dumpfile_ori, dumpfile_merged) :
    folderpath = "documents/"
    outputfile1 = "output_" + db_name + "/keyphrase_pke_ori.txt"
    outputfile2 = "output_" + db_name + "/keyphrase_pke_merged.txt"
    outputcomparison = "output_" + db_name + "/clustlabel_comparison.csv"
    
    #KEYPHRASE EXTRACTION BIASA
    load_documents(dumpfile_ori, folderpath)
    keyphrases_ori, tcscores_ori = generate_keyphrases_files(folderpath, outputfile1)
    logging.info('Cluster labeling original cluster [DONE]')
    load_documents(dumpfile_merged, folderpath)
    keyphrases_merged, tcscores_merged = generate_keyphrases_files(folderpath, outputfile2)
    logging.info('Cluster labeling merged cluster [DONE]')
    
    avg_merged, avg_ori, avg_merged_ori = output_topic_coherence(outputcomparison,
                                                                 keyphrases_ori, keyphrases_merged,
                                                                 tcscores_ori, tcscores_merged)
    logging.info('Finished.')

    return avg_merged, avg_ori, avg_merged_ori

def main_loop() :
    clust_dict = dict()
    for file in os.listdir(input_folder) :
        file_noext = (file.split('.')[0]).split('_')
        loopnumber = file_noext[len(file_noext)-1]
        if loopnumber not in clust_dict :
            clust_dict[loopnumber] = ['','','']
        if file.split('_')[0] == 'new' :
            clust_dict[loopnumber][1] = os.path.join(input_folder, file)
        elif file.split('_')[0] == 'clust' :
            clust_dict[loopnumber][0] = os.path.join(input_folder, file)
        else :
            clust_dict[loopnumber][2] = os.path.join(input_folder, file)
    
    csvfile = open(coherence_loop_output,'wb')
    csvwriter = csv.writer(csvfile, delimiter=';')
    csvwriter.writerow(['n Cluster','ID', 'Coherence Merged','Coherence Original', 'Coherence Ori-merged'])
    
    for loopnumber in clust_dict :
        print loopnumber
        dumpfile_ori = clust_dict[loopnumber][0]
        dumpfile_merged = clust_dict[loopnumber][1]
        global clustermapfile
        clustermapfile = clust_dict[loopnumber][2]

        avg_merged, avg_ori, avg_merged_ori = main(dumpfile_ori, dumpfile_merged)
        n_clust = loopnumber.split('-')[0]
        loop_id = loopnumber.split('-')[1]
        csvwriter.writerow([n_clust,loop_id, avg_merged, avg_ori, avg_merged_ori])
    csvfile.close()

if __name__ == "__main__" :
    dumpfile_ori = "output_" + db_name + "/clust_article_dump.pkl"
    dumpfile_merged = "output_" + db_name + "/new_clust_article_dump.pkl"
    main(dumpfile_ori, dumpfile_merged)
    #main_loop()
    
    
