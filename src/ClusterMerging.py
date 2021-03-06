import logging, warnings
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import networkx as nx
from ClusterLabeling import *
import csv
from tqdm import tqdm
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
from Utils import *

def generate_cluster_graph(clust_words, model) :
    logging.info('Generating Graph....')
    cluster_graph = []
    cluster_graph_size = []
    for i in tqdm(range(len(clust_words)), leave=False) :
        gr = build_graph(clust_words[i], model)
        nodes = gr.number_of_nodes()
        edges = gr.number_of_edges()
        cluster_graph.append(gr)
        cluster_graph_size.append((nodes, edges))
    return cluster_graph, cluster_graph_size


def generate_cluster_graph_v2(clust_words, clust_contents) :
    '''
    v2 : pake graph co-occurence
    '''
    logging.info('Generating Graph....')
    cluster_graph = []
    cluster_graph_size = []
    for i in tqdm(range(len(clust_words)), leave=False) :
        contents = clust_contents[i]
        #MERGE ALL CONTENTS IN A CLUSTER
        all_contents = ' '.join(contents)
        #EXTRACT SENTENCES IN A CLUSTER
        sentences = extract_sentences(all_contents)
        win = 2
        gr = build_graph_cooccurence(clust_words[i], sentences, win)
        nodes = gr.number_of_nodes()
        edges = gr.number_of_edges()
        cluster_graph.append(gr)
        cluster_graph_size.append((nodes, edges))
    return cluster_graph, cluster_graph_size


def graph_to_csv(G, nfilename, efilename) :
    #open node & edge csv file
    nodefile = open(nfilename, 'w', newline='')
    nodewriter = csv.writer(nodefile, delimiter=';')
    edgefile = open(efilename, 'w', newline='')
    edgewriter = csv.writer(edgefile, delimiter=';')

    #write node csv
    nodewriter.writerow(['Id','Label'])
    for node in list(G.nodes()) :
        nodewriter.writerow([node,node])

    #write edge csv
    edgewriter.writerow(['Source','Target','Type','Weight'])
    for edge in list(G.edges()) :
        source = edge[0]
        target = edge[1]
        weight = G[source][target]['weight']
        edgewriter.writerow([source,target,'Undirected',weight])
    
    #close
    nodefile.close()
    edgefile.close()
    logging.info("Graph CSV created successfully!")


def generate_mcs(G1,G2) :
    common_nodes = [x for x in tqdm(G1, leave=False, desc='MCS nodes') if x in G2]
    G3 = nx.Graph()
    G3.add_nodes_from(common_nodes)
    
    for node_i in tqdm(G3, leave=False, desc='MCS edges') :
        neighbours = list(G1.adj[node_i])
        for node_j in neighbours :
            if G2.has_edge(node_i,node_j) :
                #print (node_i, '-', node_j)
                weight_G1 = G1[node_i][node_j]['weight']
                #weight_G2 = G2[node_i][node_j]['weight']
                #print (weight_G1, ', ', weight_G2)
                #min_weight = min([weight_G1, weight_G2])
                #max_weight = max([weight_G1, weight_G2])
                #mcs_weight = min_weight / float(max_weight)
                mcs_weight = weight_G1
                G3.add_edge(node_i,node_j, weight=mcs_weight)
    
    return G3


def mcs_distance_score(G1,G2, alpha=0.7) :
    G3 = generate_mcs(G1,G2)

    g3_n = G3.number_of_nodes()
    g3_e = G3.number_of_edges()
    g2_n = G2.number_of_nodes()
    g2_e = G2.number_of_edges()
    g1_n = G1.number_of_nodes()
    g1_e = G1.number_of_edges()
    n_max = max(g1_n,g2_n)
    e_max = max(g1_e,g2_e)

    dist_score = 1-( alpha*(float(g3_n/n_max)) + (1-alpha)*(float(g3_e/e_max)) )
    return dist_score


def mcs_distance_score_v2(G1,G2, alpha=0.7) :
    '''
    v2 : Based on Wu et al. 2011; incorporating weight to the calculation.
    '''
    G3 = generate_mcs(G1,G2)

    g3_n = G3.number_of_nodes()
    g3_e = G3.number_of_edges()
    g2_n = G2.number_of_nodes()
    g2_e = G2.number_of_edges()
    g1_n = G1.number_of_nodes()
    g1_e = G1.number_of_edges()
    n_max = max(g1_n,g2_n)
    e_max = max(g1_e,g2_e)
    mcs_weights = [d['weight'] for (u,v,d) in G3.edges(data=True)]
    sum_weight = sum(mcs_weights)

    dist_score = 1-( alpha*(float(g3_n/n_max)) + (1-alpha)*(float(sum_weight/e_max)) )
    mcs_percent = (g3_n / n_max) * 100
    mcs_size = (g3_n, g3_e)
    return dist_score, mcs_percent, mcs_size

def threshold_std(distances) :
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    adapt_threshold = mean_dist - std_dist
    return adapt_threshold

def threshold_percentile(distances, k) :
    adapt_threshold = np.percentile(distances, k)
    return adapt_threshold

def generate_graphdist_matrix(cluster_graph, cluster_graph_size, percentile, graphdistfile, mcspercentfile) :
    logging.info("Generating Graph Distance Matrix....")
    n = len(cluster_graph)
    graphdist_matrix = np.zeros((n,n))
    mcspercent_matrix = np.zeros((n,n))
    mcssize_matrix = np.empty((n,n), object)
    for i in range(n) :
        mcssize_matrix[i][i] = (0,0)

    graphdistances = []

    for i in tqdm(range(n-1), leave=False, desc='Graph source') :
        for j in tqdm(range(i+1,n), leave=False, desc='Graph target') :
            #dist = mcs_distance_score(cluster_graph[i], cluster_graph[j])
            dist, mcs_percent, mcs_size = mcs_distance_score_v2(cluster_graph[i], cluster_graph[j])
            graphdistances.append(dist)
            graphdist_matrix[i][j] = dist
            graphdist_matrix[j][i] = dist
            mcspercent_matrix[i][j] = mcs_percent
            mcspercent_matrix[j][i] = mcs_percent
            mcssize_matrix[i][j] = mcs_size
            mcssize_matrix[j][i] = mcs_size

    #ADAPTIVE THRESHOLD
    #adapt_threshold = threshold_std(graphdistances)
    adapt_threshold = threshold_percentile(graphdistances, percentile)

    logging.info("Adaptive Threshold = " + str(adapt_threshold))
    #adapt_threshold = round(adapt_threshold,2)

    output_distmatrix_csv(graphdist_matrix, mcspercent_matrix, mcssize_matrix, cluster_graph_size, graphdistfile)
    save_to_pickle(mcspercentfile, mcspercent_matrix, mcssize_matrix, cluster_graph_size)

    return graphdist_matrix, adapt_threshold 


def generate_centroiddist_matrix(centroids, centroiddistfile) :
    logging.info("Generating Centroid Distance Matrix....")
    n = len(centroids)

    #DUMMY PARAMETER UNTUK FUNGSI output_distmatrix_csv()
    cluster_graph_size = np.zeros(n)
    mcspercent_matrix = np.zeros((n,n))
    mcssize_matrix = np.empty((n,n), object)
    for i in range(n) :
        mcssize_matrix[i][i] = (0,0)
    
    #KONSTRUKSI MATRIX SIMILARITAS KLASTER MENGGUNAKAN CENTROID
    centroiddist_matrix = cosine_distances(centroids)

    centroiddistances = []
    
    for i in tqdm(range(n-1), leave=False, desc='Centroid source') :
        for j in tqdm(range(i+1,n), leave=False, desc='Centroid target') :
            dist = centroiddist_matrix[i][j]
            centroiddistances.append(dist)

    mean_dist = np.mean(centroiddistances)
    std_dist = np.std(centroiddistances)
    adapt_threshold = mean_dist - std_dist
    logging.info("Adaptive Threshold = " + str(adapt_threshold))
    #adapt_threshold = round(adapt_threshold,2)
    #logging.info("Adaptive Threshold (round) = " + str(adapt_threshold))

    output_distmatrix_csv(centroiddist_matrix, mcspercent_matrix, mcssize_matrix, cluster_graph_size, centroiddistfile)
    #save_to_pickle(mcspercentfile, mcspercent_matrix, mcssize_matrix, cluster_graph_size)

    return centroiddist_matrix, adapt_threshold 
    

def hier_cluster_merging(graphdist_matrix, min_dist, mergefile) :
    logging.info('Cluster merging [START]')
    X = squareform(graphdist_matrix)
    Z = linkage(X,'average')
    fclust = fcluster(Z, min_dist, criterion='distance')
    
    plt.figure()
    dn = dendrogram(Z)
    #plt.show()
    plt.savefig(mergefile)
    
    logging.info('Cluster merging [DONE]')
    #logging.info('Merged cluster : ' + str(fclust))
    return fclust


def postprocess_cluster_merging(merged_cluster, 
                                clust_articles_id, 
                                clust_articles_tokenized,
                                clust_articles_content,
                                clust_words,
                                clust_phrases) :
    n_clust = len(merged_cluster)
    n_clust_new = max(merged_cluster)
    
    new_flat_articles_id = []            #MENYIMPAN URUTAN ID ARTIKEL SECARA FLAT (1 DIMENSI)
    new_flat_articles_tokenized = []     #MENYIMPAN URUTAN KONTEN ARTIKEL SECARA FLAT (1 DIMENSI)
    new_flat_cluster_label = []          #MENYIMPAN URUTAN LABEL KLASTER SECARA FLAT (1 DIMENSI)

    new_clust_articles_id = []            #MENYIMPAN ID ARTIKEL PER KLASTER (2 DIMENSI)
    new_clust_articles_tokenized = []     #MENYIMPAN KONTEN ARTIKEL (tokenized) PER KLASTER (2 DIMENSI)
    new_clust_articles_content = []     #MENYIMPAN KONTEN ARTIKEL (raw) PER KLASTER (2 DIMENSI)
    new_clust_words = []            #MENYIMPAN KATA PENTING PER KLASTER (2 DIMENSI)
    new_clust_phrases = []          #MENYIMPAN FRASE PENTING PER KLASTER (2 DIMENSI)
    
    for i in range(n_clust_new) :
        new_clust_words.append([])
        new_clust_phrases.append([])
        new_clust_articles_id.append([])
        new_clust_articles_tokenized.append([])
        new_clust_articles_content.append([])
    
    for i in range(n_clust) :
        current_cluster_label = merged_cluster[i] - 1
        n_data = len(clust_articles_id[i])
        for j in range(n_data) :
            temp_art_id = clust_articles_id[i][j]
            temp_art_tokenized = clust_articles_tokenized[i][j]
            new_flat_articles_id.append(temp_art_id)
            new_flat_articles_tokenized.append(temp_art_tokenized)
            new_flat_cluster_label.append(current_cluster_label)
        new_clust_words[current_cluster_label].extend([t for t in clust_words[i] if t not in new_clust_words[current_cluster_label]])
        new_clust_phrases[current_cluster_label].extend([t for t in clust_phrases[i] if t not in new_clust_phrases[current_cluster_label]])
        new_clust_articles_id[current_cluster_label].extend(clust_articles_id[i])
        new_clust_articles_tokenized[current_cluster_label].extend(clust_articles_tokenized[i])
        new_clust_articles_content[current_cluster_label].extend(clust_articles_content[i])
    
    return new_flat_articles_id, new_flat_articles_tokenized, new_flat_cluster_label, \
            new_clust_words, new_clust_phrases, new_clust_articles_id, new_clust_articles_tokenized, new_clust_articles_content


def output_new_avg_silh(new_n_clusters, new_avg_silh, silhscorefile_new) :

    new_avg_silh = round(new_avg_silh, 4)
    logging.info('For n_clusters = ' + str(new_n_clusters) + ' The new average silhouette score : ' + str(new_avg_silh))
    with open(silhscorefile_new,'w') as f :
        f.write('New Silhouette Score : ' + str(new_avg_silh))


def output_cluster_mapping(merged_cluster, clustmapfile) :
    n_clust = len(merged_cluster)
    new_n_clust = max(merged_cluster)

    #CREATING CLUSTER MAPPING
    cluster_map = dict()
    for i in range(n_clust) :
        new_cluster_label = merged_cluster[i]
        if new_cluster_label not in cluster_map :
            cluster_map[new_cluster_label] = []
        cluster_map[new_cluster_label].append(i+1)
    
    #OUTPUT CLUSTER MAPPING TO CSV
    with open(clustmapfile, 'w', newline='') as csvfile :
        csvwriter = csv.writer(csvfile, delimiter=';')
        csvwriter.writerow(["New Cluster","Ori Cluster"])
        for i in range(1, new_n_clust + 1) :
            for ori_label in cluster_map[i] :
                csvwriter.writerow([i,ori_label])
    
    #IDENTIFICATION NEWLY CREATED CLUSTERS
    new_clusters = [c for c in cluster_map if len(cluster_map[c]) > 1]
    merged_ori_clusters = [c for key in new_clusters for c in cluster_map[key]]
    n_new_clusters = len(new_clusters)
    n_merged_ori_clusters = len(merged_ori_clusters)

    logging.info(str(n_merged_ori_clusters) + " original clusters merged to " + str(n_new_clusters) + " new clusters")
    return new_n_clust


def output_distmatrix_csv(distmatrix, mcspmatrix, mcssmatrix, cluster_graph_size, distfile) :
    shape = distmatrix.shape[0]

    with open(distfile, 'w', newline='') as csvfile :
        csvwriter = csv.writer(csvfile, delimiter=';')

        #WRITING DISTANCE MATRIX
        toprow = ['']
        toprow.extend([str(i+1) for i in range(shape)])
        csvwriter.writerow(toprow)

        for i in range(shape) :
            row = [str(i+1)]
            row.extend([dist for dist in distmatrix[i]])
            csvwriter.writerow(row)
        
        csvwriter.writerow([''])
        csvwriter.writerow([''])

        #WRITING MCS PERCENTAGE MATRIX
        toprow = ['']
        toprow.extend([str(i+1) for i in range(shape)])
        csvwriter.writerow(toprow)

        for i in range(shape) :
            row = [str(i+1)]
            row.extend([percent for percent in mcspmatrix[i]])
            csvwriter.writerow(row)
        
        csvwriter.writerow([''])
        csvwriter.writerow([''])

        #WRITING MCS SIZE MATRIX
        toprow = ['']
        toprow.extend([str(i+1) for i in range(shape)])
        csvwriter.writerow(toprow)

        for i in range(shape) :
            row = [str(i+1)]
            row.extend([size_tuple for size_tuple in mcssmatrix[i]])
            row.extend(['', cluster_graph_size[i]])
            csvwriter.writerow(row)
        
        csvwriter.writerow([''])
        botrow = ['']
        botrow.extend([i for i in cluster_graph_size])
        csvwriter.writerow(botrow)

