from Utils import *
from Clustering import *
from ClusterMerging import *
from ClusterLabeling import *
from FrequentPhraseMining import *
import mysql.connector

if __name__ == "__main__" :
    '''Jangan lupa kalo ganti database, diganti bagian :
        - collecting_data() - Clustering
        - store_cluster_label - Clustering
        - cluster_to_csv() - Clustering
    '''
    #PREPARE THE OUTPUT
    conn = mysql.connector.connect(user='root', password='admin', host='127.0.0.1', database='article600')

    print ("START")
    
    '''PRE-PROCESS OFF

    print ("PRE-PROCESS")
    
    #LOAD DATA DIRECTLY FROM DATABASE
    articles_id, articles_content = collecting_data(conn)
    
    #PRE-PROCESS EVERY ARTICLE
    articles_tokenized = preprocess_articles(articles_content)

    #SAVE FILE TO PICKLE PRE-PROCESS
    save_to_pickle('output_600data/article_dump.pkl', articles_id, articles_tokenized)

    #PRE-PROCESS OFF'''

    #LOAD DATA FROM PICkLE PRE-PROCESS
    unpack = load_from_pickle('output_600data/article_dump.pkl')
    articles_id, articles_tokenized = unpack[0], unpack[1]

    #TRAIN WORD2VEC
    modelname = 'w2v_model/all_articles.w2v'
    #train_word2vec(articles_tokenized, modelname)
    #LOAD WORD2VEC MODEL
    w2v_model = load_word2vec(modelname)
    
    '''CLUSTERING OFF

    #CLUSTERING
    print ("CLUSTERING")
    
    n_clusters = 12
    silhscorefile = 'output_600data/silhscore_ori.txt'
    cluster_labels, sample_silhouette_values = cluster_word2vec(w2v_model,
                                                                articles_tokenized,
                                                                n_clusters,
                                                                silhscorefile,
                                                                False)
    store_cluster_label(conn, articles_id, cluster_labels, sample_silhouette_values)
    cluster_tocsv(conn, 'output_600data/cluster.csv')
    
    #LOAD CLUSTERS FROM DATABASE
    clust_articles_id, clust_articles_content = collecting_cluster_data(conn)
    
    #RE-PRE-PROCESS CLUSTERS ARTICLES CONTENT
    clust_articles_tokenized = preprocess_clust_articles(clust_articles_content)
    
    #SAVE FILE TO PICKLE CLUSTERING
    save_to_pickle('output_600data/clust_article_dump.pkl', clust_articles_id, 
                                                            clust_articles_tokenized,
                                                            clust_articles_content)

    #CLUSTERING OFF '''

    #LOAD DATA FROM PICkLE CLUSTERING
    unpack = load_from_pickle('output_600data/clust_article_dump.pkl')
    clust_articles_id, clust_articles_tokenized, clust_articles_content = unpack[0], unpack[1], unpack[2]

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

    #''''FPM & HIERARCHICAL CLUSTER MERGING OFF

    print("FREQUENT PHRASE MINING")
    #FREQUENT PHRASE MINING
    min_count = 3
    min_count_phrase = 3
    clust_words, clust_phrases = extract_clust_phrases(clust_articles_tokenized, min_count, min_count_phrase)

    #HIERARCHICAL CLUSTER MERGING
    print("HIERARCHICAL CLUSTER MERGING")

    #GENERATE & SAVE GRAPH FOR EVERY CLUSTER
    cluster_graph = generate_cluster_graph(clust_words, w2v_model)
    #save_to_pickle('output_600data/cluster_graph.pkl', cluster_graph)
    #LOAD CLUSTER GRAPH FOR HIERARCHICAL CLUSTER MERGING
    #unpack = load_from_pickle('output_600data/cluster_graph.pkl')
    #cluster_graph = unpack[0]

    #GENERATE GRAPH DISTANCE MATRIX
    graphdistfile = 'output_600data/graphdist_matrix.txt'
    graphdist_matrix = generate_graphdist_matrix(cluster_graph, graphdistfile)

    #THE CLUSTER MERGING
    min_dist = 0.90
    merged_cluster = hier_cluster_merging(graphdist_matrix, min_dist, plot=False)
    save_to_pickle('output_600data/merged_cluster.pkl')
    
    #CLUSTER MAPPING
    clustmapfile = 'output_600data/cluster_mapping.csv'
    new_n_clusters = output_cluster_mapping(merged_cluster, clustmapfile)

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
    silhscorefile_new = 'output_600data/silhscore_new.txt'
    output_new_avg_silh(new_n_clusters, new_avg_silh, silhscorefile_new)

    #STORE NEW CLUSTER LABEL TO DATABASE & CREATE CSV FILE
    store_cluster_label(conn, new_flat_articles_id, new_flat_cluster_label, new_samples_silh)
    cluster_tocsv(conn, 'output_600data/new_cluster.csv')

    #SAVE FILE TO PICKLE HIERARCHICAL CLUSTER MERGING
    save_to_pickle('output_600data/new_clust_article_dump.pkl', new_clust_articles_id, 
                                                                new_clust_articles_tokenized, 
                                                                new_clust_articles_content)
    save_to_pickle('output_600data/new_clust_keytokens_dump.pkl', new_clust_words, new_clust_phrases)

    #FPM & HIERARCHICAL CLUSTER MERGING OFF'''

    #LOAD DATA FROM PICkLE HIERARCHICAL CLUSTER MERGING
    #unpack = load_from_pickle('output_600data/new_clust_article_dump.pkl')
    #new_clust_articles_id, new_clust_articles_tokenized, new_clust_articles_content = unpack[0], unpack[1], unpack[2]
    #unpack = load_from_pickle('output_600data/new_clust_keytokens_dump.pkl')
    #new_clust_words, new_clust_phrases = unpack[0], unpack[1]
    
    print("CLUSTER LABELING")
    #CLUSTER LABELING
    #clust_text = [' '.join(t) for t in new_clust_words]
    max_phrase = 5
    clust_keywords2, clust_keyphrases2 = cluster_labeling_cooccurence(new_clust_words, 
                                                                      new_clust_phrases, 
                                                                      new_clust_articles_content,
                                                                      max_phrase)
    #clust_keywords1, clust_keyphrases1 = cluster_labeling_v2(new_clust_articles_tokenized, 
    #                                                       new_clust_phrases, 
    #                                                       w2v_model,
    #                                                       max_phrase)                                                           

    #OUTPUT
    fout = open('output_600data/keyphrase_textrank.txt','w')
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
    conn.close()