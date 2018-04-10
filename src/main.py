from Utils import *
from Clustering import *
from ClusterMerging import *
from ClusterLabeling import *
from FrequentPhraseMining import *
from tqdm import tqdm
import mysql.connector
import os
import csv

def main(DB_NAME, n_clusters) :
    #PREPARE THE OUTPUT
    #DB_NAME = 'article550'
    conn = mysql.connector.connect(user='root', password='admin', host='127.0.0.1', database=DB_NAME)
    modelname = 'w2v_model/all_articles.w2v'
    output_folder = 'output_' + DB_NAME + '/'
    os.makedirs(os.path.dirname(output_folder), exist_ok=True)
    #PRE-PROCESS OUTPUT
    ARTICLE = output_folder + "article_dump.pkl"
    #CLUSTERING OUTPUT
    SILHSCORE_ORI = output_folder + "silhscore_ori.txt"
    CLUSTER_ORI = output_folder + "cluster_ori.csv"
    CLUST_ARTICLE = output_folder + "clust_article_dump.pkl"
    #HIERARCHICAL CLUSTER MERGING OUTPUT
    GRAPHDIST_MATRIX = output_folder + "graphdist_matrix.txt"
    MERGED_CLUSTER = output_folder + "merged_cluster.pkl"
    CLUSTER_MAPPING = output_folder + "cluster_mapping.csv"
    SILHSCORE_MERGED = output_folder + "silhscore_merged.txt"
    CLUSTER_MERGED = output_folder + "cluster_merged.csv"
    CLUST_ARTICLE_MERGED = output_folder + "new_clust_article_dump.pkl"
    CLUST_KEYTOKENS_MERGED = output_folder + "new_clust_keytokens_dump.pkl"
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

    print ("START")
    
    '''PRE-PROCESS OFF

    print ("PRE-PROCESS")
    
    #LOAD DATA DIRECTLY FROM DATABASE
    articles_id, articles_content = collecting_data(conn)
    
    #PRE-PROCESS EVERY ARTICLE
    articles_tokenized = preprocess_articles(articles_content)

    #SAVE FILE TO PICKLE PRE-PROCESS
    save_to_pickle(ARTICLE, articles_id, articles_tokenized)

    #PRE-PROCESS OFF'''

    #LOAD DATA FROM PICkLE PRE-PROCESS
    unpack = load_from_pickle(ARTICLE)
    articles_id, articles_tokenized = unpack[0], unpack[1]

    #TRAIN WORD2VEC
    #train_word2vec(articles_tokenized, modelname)
    #LOAD WORD2VEC MODEL
    w2v_model = load_word2vec(modelname)
    
    #'''CLUSTERING OFF

    #CLUSTERING
    print ("CLUSTERING")
    
    cluster_labels, silhscore_ori ,sample_silhouette_values = cluster_word2vec(w2v_model,
                                                                articles_tokenized,
                                                                n_clusters,
                                                                SILHSCORE_ORI,
                                                                False)
    #store_cluster_label(conn, articles_id, cluster_labels, sample_silhouette_values)
    #cluster_tocsv(conn, CLUSTER_ORI)
    #TAMBAHAN UNTUK NON-DATABASE
    clust_articles_id, clust_articles_content = postprocess_clustering(cluster_labels,articles_id, articles_content)
    
    #LOAD CLUSTERS FROM DATABASE
    clust_articles_id, clust_articles_content = collecting_cluster_data(conn)
    
    #RE-PRE-PROCESS CLUSTERS ARTICLES CONTENT
    clust_articles_tokenized = preprocess_clust_articles(clust_articles_content)
    
    #SAVE FILE TO PICKLE CLUSTERING
    save_to_pickle(CLUST_ARTICLE , clust_articles_id, 
                                   clust_articles_tokenized,
                                   clust_articles_content)

    #CLUSTERING OFF '''

    #LOAD DATA FROM PICkLE CLUSTERING
    #unpack = load_from_pickle(CLUST_ARTICLE)
    #clust_articles_id, clust_articles_tokenized, clust_articles_content = unpack[0], unpack[1], unpack[2]

    '''
    #EXPERIMENT FPM
    clust_artcontent_tokenized = []
    for i in range(len(clust_articles_content)) :
        clust_articles = clust_articles_content[i]
        clust_artcontent_tokenized.append([])
        for article_content in clust_articles :
            preprocessed_content = preprocess_text_experimental(article_content)
            clust_artcontent_tokenized[i].append(preprocessed_content.split(' '))
    #'''

    #''''FPM & HIERARCHICAL CLUSTER MERGING & RE-CLUSTERING OFF

    print("FREQUENT PHRASE MINING")
    #FREQUENT PHRASE MINING
    clust_words, clust_phrases = extract_clust_phrases(clust_articles_tokenized, min_count, min_count_phrase)

    #HIERARCHICAL CLUSTER MERGING
    print("HIERARCHICAL CLUSTER MERGING")

    #GENERATE & SAVE GRAPH FOR EVERY CLUSTER
    cluster_graph = generate_cluster_graph(clust_words, w2v_model)
    #cluster_graph = generate_cluster_graph_v2(clust_words, clust_articles_content)
    #save_to_pickle('output_600data/cluster_graph.pkl', cluster_graph)
    #LOAD CLUSTER GRAPH FOR HIERARCHICAL CLUSTER MERGING
    #unpack = load_from_pickle('output_600data/cluster_graph.pkl')
    #cluster_graph = unpack[0]

    #GENERATE GRAPH DISTANCE MATRIX
    graphdist_matrix = generate_graphdist_matrix(cluster_graph, GRAPHDIST_MATRIX)

    #THE CLUSTER MERGING
    merged_cluster = hier_cluster_merging(graphdist_matrix, min_dist, plot=False)
    save_to_pickle(MERGED_CLUSTER)
    
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

    #RE-CLUSTERING WITH NEW CLUSTER NUMBER
    print("RE-CLUSTERING")
    n_clusters = new_n_clusters
    cluster_labels_reclust, silhscore_reclust ,sample_silhouette_values_reclust = cluster_word2vec(w2v_model,
                                                                articles_tokenized,
                                                                n_clusters,
                                                                SILHSCORE_RECLUST,
                                                                False)

    #STORE NEW CLUSTER LABEL TO DATABASE & CREATE CSV FILE
    store_cluster_label(conn, new_flat_articles_id, new_flat_cluster_label, new_samples_silh)
    cluster_tocsv(conn, CLUSTER_MERGED)

    #SAVE FILE TO PICKLE HIERARCHICAL CLUSTER MERGING
    save_to_pickle(CLUST_ARTICLE_MERGED, new_clust_articles_id, 
                                                                new_clust_articles_tokenized, 
                                                                new_clust_articles_content)
    save_to_pickle(CLUST_KEYTOKENS_MERGED, new_clust_words, new_clust_phrases)

    #FPM & HIERARCHICAL CLUSTER MERGING OFF'''

    #LOAD DATA FROM PICkLE HIERARCHICAL CLUSTER MERGING
    #unpack = load_from_pickle(CLUST_ARTICLE_MERGED)
    #new_clust_articles_id, new_clust_articles_tokenized, new_clust_articles_content = unpack[0], unpack[1], unpack[2]
    #unpack = load_from_pickle(CLUST_KEYTOKENS_MERGED)
    #new_clust_words, new_clust_phrases = unpack[0], unpack[1]
    
    ''' CLUSTER LABELING OFF

    print("CLUSTER LABELING")
    #CLUSTER LABELING
    clust_keywords2, clust_keyphrases2 = cluster_labeling_cooccurence(new_clust_words, 
                                                                      new_clust_phrases, 
                                                                      new_clust_articles_content,
                                                                      max_phrase)
    #clust_keywords1, clust_keyphrases1 = cluster_labeling_v2(new_clust_articles_tokenized, 
    #                                                       new_clust_phrases, 
    #                                                       w2v_model,
    #                                                       max_phrase)                                                           

    #OUTPUT
    fout = open(KEYPHRASE,'w')
    for i in range(len(clust_keywords2)) :
        fout.write('Cluster-' + str(i+1) + ' :\n')
        fout.write('Phrases :\n')
        fout.write(str(new_clust_phrases[i]) + '\n')
        #fout.write('Words :' + str(len(new_clust_words[i])) + '\n')
        #fout.write(str(new_clust_words[i]) + '\n\n')

        #fout.write('Keywords (w2v) :' + str(len(clust_keywords1[i])) + '\n')
        #fout.write(str(clust_keywords1[i]) + '\n')
        #fout.write('Keyphrases (w2v) :\n')
        #fout.write(str(clust_keyphrases1[i]) + '\n\n')
        
        fout.write('Keywords (cooccurence) :\n')
        fout.write(str(clust_keywords2[i]) + '\n')
        fout.write('Keyphrases (cooccurence) :\n')
        fout.write(str(clust_keyphrases2[i]) + '\n\n')
    fout.close()

    # CLUSTER LABELING OFF '''
    conn.close()

    return silhscore_ori, new_n_clusters, new_avg_silh, silhscore_reclust

def main_n() :
    #PREPARE INPUT AND OUTPUT
    DB_NAME = 'article550'
    output_folder = 'output_' + DB_NAME + '/'
    os.makedirs(os.path.dirname(output_folder), exist_ok=True)
    #PREPARE CSV FILE
    filename = output_folder + 'silhscore_comparison.csv'
    csvfile = open(filename, 'w', newline='')
    csvwriter = csv.writer(csvfile, delimiter=';')
    csvwriter.writerow(['No', '','Silhouette Score','','','','Delta','Delta (Abs)'])
    csvwriter.writerow(['', 'Original','','Merged','','Re-Clustering','',''])
    #PREPARE PARAMETER
    try_n_clusters = [14]
    n = 10

    for n_clust in try_n_clusters :
        for i in range(n) :
            print('N CLUSTER ' + str(n_clust) + ' RUNNING NO-' + str(i+1))
            silhscore_ori, new_n_clust, silhscore_merged, silhscore_reclust = main(DB_NAME, n_clust)
            delta = silhscore_reclust - silhscore_merged
            delta_abs = abs(delta)
            csvwriter.writerow([str(i+1), n_clust, silhscore_ori, 
                                new_n_clust, silhscore_merged, silhscore_reclust, 
                                delta, delta_abs])

    csvfile.close()

if __name__ == "__main__" :
    main_n()