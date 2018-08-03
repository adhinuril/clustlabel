from Utils import *
from Clustering import *
from ClusterMerging import *
from ClusterLabeling import *
from FrequentPhraseMining import *
from tqdm import tqdm
import mysql.connector
import os
import csv

#PREPARE THE OUTPUT
DB_NAME = 'article550'
modelname = 'w2v_model/all_articles.w2v'
output_folder = 'output_' + DB_NAME + '/'
os.makedirs(os.path.dirname(output_folder), exist_ok=True)
output_folder_artdumps = 'output_' + DB_NAME + '/article_dumps/'
os.makedirs(os.path.dirname(output_folder_artdumps), exist_ok=True)
#PRE-PROCESS OUTPUT
ARTICLE = output_folder + "article_dump.pkl"
#CLUSTERING OUTPUT
SILHSCORE_ORI = output_folder + "silhscore_ori.txt"
AVG_SILH_ORI = output_folder + "cluster_averagesilh_ori.csv"
CLUSTER_ORI = output_folder + "cluster_ori.csv"
CLUST_ARTICLE = output_folder + "clust_article_dump.pkl"
SILHFILE = output_folder + "silhouette_analysis.png"
SILH_CSV = output_folder + "silhouette_analysis.csv"
#HIERARCHICAL CLUSTER MERGING OUTPUT
DIST_MATRIX = output_folder + "dist_matrix.csv"
MERGED_CLUSTER = output_folder + "merged_cluster.pkl"
CLUSTER_MAPPING = output_folder + "cluster_mapping.csv"
SILHSCORE_MERGED = output_folder + "silhscore_merged.txt"
CLUSTER_MERGED = output_folder + "cluster_merged.csv"
CLUST_ARTICLE_MERGED = output_folder + "new_clust_article_dump.pkl"
CLUST_KEYTOKENS_MERGED = output_folder + "new_clust_keytokens_dump.pkl"
CLUSTER_GRAPH = output_folder + "cluster_graph_ori.pkl"
DENDOGRAM = output_folder + "dendogram.png"
MCSPERCENT_MATRIX = output_folder + "mcspercent_matrix.pkl"
#RE-CLUSTERING OUTPUT
SILHSCORE_RECLUST = output_folder + "silhscore_reclustering.txt"
#CLUSTER LABELING OUTPUT
KEYPHRASE = output_folder + "keyphrase_textrank.txt"

#PARAMETER
#n_clusters = 13
min_count = 3
min_count_phrase = 3
min_dist = 0.9
max_phrase = 5

def preprocess() :
    
    #LOAD DATA DIRECTLY FROM DATABASE
    articles_id, articles_text = collecting_data(DB_NAME)
    
    #PRE-PROCESS EVERY ARTICLE
    articles_tokenized = preprocess_articles(articles_text)

    #SAVE FILE TO PICKLE PRE-PROCESS
    save_to_pickle(ARTICLE, articles_id, articles_tokenized, articles_text)

    #return articles_id, articles_content, articles_tokenized


def clustering(w2v_model, n_clusters, articles_id, articles_tokenized, articles_text, looping=False) :
    
    #CLUSTERING
    #'''
    cluster_labels, centroids, silhscore_ori ,sample_silhouette_values = cluster_word2vec(w2v_model,
                                                                articles_tokenized,
                                                                n_clusters,
                                                                SILHSCORE_ORI,
                                                                False)
    #'''                                                                
    '''
    cluster_labels, centroids, silhscore_ori ,sample_silhouette_values = cluster_birch(w2v_model,
                                                                articles_tokenized,
                                                                n_clusters,
                                                                SILHSCORE_ORI,
                                                                False)
    #'''
    '''
    cluster_labels, centroids, silhscore_ori ,sample_silhouette_values = cluster_tfidf(articles_tokenized,
                                                                n_clusters,
                                                                SILHSCORE_ORI)
    #'''
    '''
    cluster_labels, centroids, silhscore_ori ,sample_silhouette_values = cluster_spectral(w2v_model,
                                                                articles_tokenized,
                                                                n_clusters,
                                                                SILHSCORE_ORI,
                                                                False)
    #'''

    #POST-PROCESSING CLUSTERING RESULT
    clust_articles_id, clust_articles_tokenized, clust_articles_text, clust_articles_silh = \
        postprocess_clustering(cluster_labels, articles_id, articles_tokenized, articles_text, sample_silhouette_values)

    #STORE CLUSTER RESULT TO CSV
    cluster_tocsv(CLUSTER_ORI, AVG_SILH_ORI, clust_articles_id, clust_articles_tokenized, clust_articles_text, clust_articles_silh)

    #SAVE FILE TO PICKLE CLUSTERING
    save_to_pickle(CLUST_ARTICLE , clust_articles_id, 
                                   clust_articles_tokenized,
                                   clust_articles_text,
                                   centroids,
                                   silhscore_ori)
    
    if (looping) :
        name = (CLUST_ARTICLE.split('/')[1]).split('.')[0]
        fname = output_folder_artdumps + name + '_' + loop_number +'.pkl'
        save_to_pickle(fname , clust_articles_id, 
                               clust_articles_tokenized,
                               clust_articles_text,
                               centroids,
                               silhscore_ori)


def clustmerging(w2v_model, clust_words, clust_phrases, clust_articles_id,
                 clust_articles_tokenized, clust_articles_content, centroids, looping=False) :
    
    #GENERATE GRAPH MATRIX
    #cluster_graph, cluster_graph_size = generate_cluster_graph(clust_words, w2v_model) #Word2Vec
    cluster_graph, cluster_graph_size = generate_cluster_graph_v2(clust_words, clust_articles_content) #Co-Occurence

    #GENERATE GRAPH DISTANCE MATRIX
    percentile = 5
    dist_matrix, adapt_threshold = generate_graphdist_matrix(cluster_graph, cluster_graph_size, percentile, DIST_MATRIX, MCSPERCENT_MATRIX)
    #dist_matrix, adapt_threshold = generate_centroiddist_matrix(centroids, DIST_MATRIX)
    
    save_to_pickle(CLUSTER_GRAPH, cluster_graph, dist_matrix)
    #unpack = load_from_pickle(CLUSTER_GRAPH)
    #cluster_graph, dist_matrix = unpack[0], unpack[1]

    #THE CLUSTER MERGING
    merged_cluster = hier_cluster_merging(dist_matrix, adapt_threshold, DENDOGRAM)
    
    #CLUSTER MAPPING
    new_n_clusters = output_cluster_mapping(merged_cluster, CLUSTER_MAPPING)

    #POST-PROCESSING HIERARCHICAL CLUSTER MERGING
    new_flat_articles_id, new_flat_articles_tokenized, new_flat_cluster_label, \
        new_clust_words, new_clust_phrases, new_clust_articles_id, new_clust_articles_tokenized, new_clust_articles_content = \
        postprocess_cluster_merging(merged_cluster, 
                                    clust_articles_id, 
                                    clust_articles_tokenized,
                                    clust_articles_content,
                                    clust_words,
                                    clust_phrases)

    #CALCULATE NEW SILHOUETTE SCORE
    new_article_matrix = generate_article_matrix(new_flat_articles_tokenized, w2v_model)
    new_avg_silh, new_samples_silh = calculate_silhouette(new_article_matrix, new_flat_cluster_label)
    output_new_avg_silh(new_n_clusters, new_avg_silh, SILHSCORE_MERGED)

    #STORE NEW CLUSTER LABEL TO DATABASE & CREATE CSV FILE
    #store_cluster_label(conn, new_flat_articles_id, new_flat_cluster_label, new_samples_silh)
    #cluster_tocsv(conn, CLUSTER_MERGED)

    #SAVE FILE TO PICKLE HIERARCHICAL CLUSTER MERGING
    save_to_pickle(CLUST_ARTICLE_MERGED, new_clust_articles_id, 
                                         new_clust_articles_tokenized, 
                                         new_clust_articles_content,
                                         new_avg_silh)
    save_to_pickle(CLUST_KEYTOKENS_MERGED, new_clust_words, new_clust_phrases)

    if (looping) :
        name = (CLUST_ARTICLE_MERGED.split('/')[1]).split('.')[0]
        fname = output_folder_artdumps + name + '_' + loop_number +'.pkl'
        save_to_pickle(fname, new_clust_articles_id, 
                              new_clust_articles_tokenized, 
                              new_clust_articles_content,
                              new_avg_silh)
        name = (CLUSTER_MAPPING.split('/')[1]).split('.')[0]
        fname = output_folder_artdumps + name + '_' + loop_number +'.csv'
        output_cluster_mapping(merged_cluster, fname)


def reclustering(w2v_model, articles_tokenized, new_clust_articles_id) :
    new_n_clusters = len(new_clust_articles_id)
    cluster_labels_reclust, centroids_reclust, \
    silhscore_reclust ,sample_silhouette_values_reclust = cluster_word2vec(w2v_model,
                                                                articles_tokenized,
                                                                new_n_clusters,
                                                                SILHSCORE_RECLUST,
                                                                False)
    return new_n_clusters, silhscore_reclust


def clustlabeling(new_clust_words, new_clust_phrases, new_clust_articles_content) :
    #clust_keywords2, clust_keyphrases2 = cluster_labeling_cooccurence(new_clust_words, 
    #                                                                  new_clust_phrases, 
    #                                                                  new_clust_articles_content,
    #                                                                  max_phrase)                                                      

    clust_keyphrases = cluster_labeling_topicrank(new_clust_articles_content, max_phrase)

    #OUTPUT
    fout = open(KEYPHRASE,'w')
    for i in range(len(clust_keyphrases)) :
        fout.write('Cluster-' + str(i+1) + ' :\n')
        fout.write('Phrases :\n')
        fout.write(str(clust_keyphrases[i]) + '\n\n')
    fout.close()


def main(n_clusters=11, looping=False) :

    print ("START")

    #PREPROCESS
    print ("PRE-PROCESS")
    preprocess()
    #LOAD PICKLE FROM PREPROCESS
    unpack = load_from_pickle(ARTICLE)
    articles_id, articles_tokenized, articles_text = unpack[0], unpack[1], unpack[2]

    #LOAD WORD2VEC MODEL
    #train_word2vec(articles_tokenized, modelname)   #if this on, then it ends here.
    w2v_model = load_word2vec(modelname)
    
    #CLUSTERING
    print ("CLUSTERING")
    #max_n_clusters = 30
    #silh_analysis(articles_tokenized,w2v_model,SILHFILE, SILH_CSV, max_n_clusters)     #if this on, then it ends here.
    clustering(w2v_model, n_clusters, articles_id, articles_tokenized, articles_text, looping)
    
    #LOAD PICKLE FROM CLUSTERING
    unpack = load_from_pickle(CLUST_ARTICLE)
    clust_articles_id, clust_articles_tokenized, clust_articles_content, centroids, silhscore_ori = \
       unpack[0], unpack[1], unpack[2], unpack[3], unpack[4]
    
    #FREQUENT PHRASE MINING
    print("FREQUENT PHRASE MINING")
    clust_words, clust_phrases = extract_clust_phrases(clust_articles_tokenized, min_count, min_count_phrase)

    #HIERARCHICAL CLUSTER MERGING
    print("HIERARCHICAL CLUSTER MERGING")
    clustmerging(w2v_model, clust_words, clust_phrases, clust_articles_id, 
                 clust_articles_tokenized,clust_articles_content, centroids, looping)
    
    #LOAD PICKLE FROM CLUSTER MERGING
    unpack = load_from_pickle(CLUST_ARTICLE_MERGED)
    new_clust_articles_id, new_clust_articles_tokenized, new_clust_articles_content, new_avg_silh = \
        unpack[0], unpack[1], unpack[2], unpack[3]
    unpack = load_from_pickle(CLUST_KEYTOKENS_MERGED)
    new_clust_words, new_clust_phrases = unpack[0], unpack[1]

    #RE-CLUSTERING WITH NEW CLUSTER NUMBER
    #print("RE-CLUSTERING")
    #new_n_clusters, silhscore_reclust = reclustering(w2v_model, articles_tokenized, new_clust_articles_id)

    #CLUSTER LABELING
    #print("CLUSTER LABELING")
    #clustlabeling(new_clust_words, new_clust_phrases, new_clust_articles_content)
    
    if (looping) :
        return silhscore_ori, new_n_clusters, new_avg_silh, silhscore_reclust
    

def main_n() :
    #PREPARE INPUT AND OUTPUT
    clear_folder(output_folder_artdumps,'pkl')
    clear_folder(output_folder_artdumps,'csv')
    #PREPARE CSV FILE
    filename = output_folder + 'silhscore_loop.csv'
    csvfile = open(filename, 'w', newline='')
    csvwriter = csv.writer(csvfile, delimiter=';')
    csvwriter.writerow(['No', '','Silhouette Score','','','','Delta','Delta (Abs)'])
    csvwriter.writerow(['', 'Original','','Merged','','Re-Clustering','',''])
    #PREPARE PARAMETER
    try_n_clusters = []
    n = 0

    for n_clust in try_n_clusters :
        acc_silh_ori = []
        acc_silh_merged = []
        acc_silh_reclust = []
        acc_delta_abs = []
        list_new_n = []
        for i in range(n) :
            global loop_number
            loop_number = str(n_clust) + '-' + str(i+1)
            print('N CLUSTER ' + str(n_clust) + ' RUNNING NO-' + str(i+1))
            silhscore_ori, new_n_clust, silhscore_merged, silhscore_reclust = main(n_clust, looping=True)
            delta = silhscore_reclust - silhscore_merged
            delta_abs = abs(delta)
            csvwriter.writerow([str(i+1), n_clust, silhscore_ori, 
                                new_n_clust, silhscore_merged, silhscore_reclust, 
                                delta, delta_abs])
            acc_silh_ori.append(silhscore_ori)
            acc_silh_merged.append(silhscore_merged)
            acc_silh_reclust.append(silhscore_reclust)
            acc_delta_abs.append(delta_abs)
            list_new_n.append(new_n_clust)
        avg_silh_ori = sum(acc_silh_ori) / float(n)
        avg_silh_merged = sum(acc_silh_merged) / float(n)
        avg_silh_reclust = sum(acc_silh_reclust) / float(n)
        avg_delta_abs = sum(acc_delta_abs) / float(n)
        frequent_new_n = max(set(list_new_n), key=list_new_n.count)
        csvwriter.writerow(['',n_clust,avg_silh_ori,frequent_new_n,avg_silh_merged,avg_silh_reclust,'',avg_delta_abs])
        csvwriter.writerow([''])
        csvwriter.writerow([''])

    csvfile.close()


def main_graphtrap(clustgraph, idxa, idxb) :

    #LOAD FILE FROM PICKLE
    logging.info('Traping Graph Commencing!')
    unpack = load_from_pickle(clustgraph)
    cluster_graph = unpack[0]

    #GENERATE MCS GRAPH
    graph_a = cluster_graph[idxa - 1]
    graph_b = cluster_graph[idxb - 1]
    logging.info('Graph Trapped Succesfully!')
    logging.info('Generating graph MCS.....')
    graph_mcs = generate_mcs(graph_a, graph_b)

    #OUTPUT FILE
    GRAPH_A_NODE = output_folder + 'G' + str(idxa) + '_nodes.csv'
    GRAPH_A_EDGE = output_folder + 'G' + str(idxa) + '_edges.csv'
    graph_to_csv(graph_a, GRAPH_A_NODE, GRAPH_A_EDGE)

    GRAPH_B_NODE = output_folder + 'G' + str(idxb) + '_nodes.csv'
    GRAPH_B_EDGE = output_folder + 'G' + str(idxb) + '_edges.csv'
    graph_to_csv(graph_b, GRAPH_B_NODE, GRAPH_B_EDGE)

    GRAPH_MCS_NODE = output_folder + 'Gmcs' + str(idxa) + '-'  + str(idxb) +  '_nodes.csv'
    GRAPH_MCS_EDGE = output_folder + 'Gmcs' + str(idxa) + '-'  + str(idxb) +  '_edges.csv'
    graph_to_csv(graph_mcs, GRAPH_MCS_NODE, GRAPH_MCS_EDGE)

    logging.info('Graph Traping Completed.')
    return 0


if __name__ == "__main__" :
    #main_n()
    main()
    #CLUSTER_GRAPH = output_folder + 'cluster_graph_ori.pkl'
    #main_graphtrap(CLUSTER_GRAPH, 2,11)
