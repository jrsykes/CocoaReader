#%%
import torch
import pandas as pd
import pymc as pm
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

def is_leaf_node(edge_list, node):
    sources = [source for source, target in edge_list]
    return node not in sources

def create_phylogenetic_dag(data):
    # Create a directed graph
    G = nx.DiGraph()

    for _, row in data.iterrows():
        G.add_edge(row['clade0'], row['clade1'])
        G.add_edge(row['clade1'], row['clade2'])
        G.add_edge(row['clade2'], row['clade3'])
        G.add_edge(row['clade3'], row['order'])
        G.add_edge(row['order'], row['family'])
        G.add_edge(row['family'], row['subfamily'])
        G.add_edge(row['subfamily'], row['genus'])
        G.add_edge(row['genus'], row['label'])
        
    # Remove these nodes from the graph
    nan_nodes = [node for node, data in G.nodes(data=True) if pd.isna(node)]
    for node in nan_nodes:
        G.remove_node(node)

    # Check if the graph has cycles and convert to DAG if necessary
    while True:
        try:
            # Finds a cycle and returns an iterator that can be used to remove the edge.
            cycle = nx.find_cycle(G, orientation='original')
            # Remove the last edge in the cycle, which should break the cycle
            G.remove_edge(cycle[-1][0], cycle[-1][1])
        except nx.NetworkXNoCycle:
            # If no cycle is found, break the loop
            break
           
    return G



#%%

# prior_dist = pd.read_csv('/home/jamiesykes/Documents/PhyloNet/taxonomy_sorted_Bayes.csv', header=0)
# labels = [5, 6, 6, 7, 15, 16]
# #subset row index with ids
# prior_data = prior_dist.loc[labels]


def make_graph(pooled_features, prior_data):
    prior_graph = create_phylogenetic_dag(prior_data)

    # Compute squared distance matrix using broadcasting
    square = np.sum(pooled_features**2, axis=1, keepdims=True)
    distance_squared = square + square.T - 2 * np.dot(pooled_features, pooled_features.T)
    distance_squared = np.clip(distance_squared, a_min=0, a_max=None)

    # Compute the distance matrix and set the diagonal to 0
    observed_distances = np.sqrt(distance_squared)
    np.fill_diagonal(observed_distances, 0)
    sigma = np.std(observed_distances)

    # Use nodes for indexing
    num_nodes = len(prior_graph.nodes())
    num_leaf_nodes = observed_distances.shape[0]

    prior_distance_matrix = np.zeros((num_nodes, num_nodes))

    # Create a mapping from node to index
    node_to_index = {node: i for i, node in enumerate(prior_graph.nodes())}

    # Fill the matrix with distances
    for u in prior_graph.nodes():
        for v in prior_graph.nodes():
            try:
                dist = nx.shortest_path_length(prior_graph, source=u, target=v)
                prior_distance_matrix[node_to_index[u], node_to_index[v]] = dist
            except nx.NetworkXNoPath:
                prior_distance_matrix[node_to_index[u], node_to_index[v]] = 100

    observed_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if prior_distance_matrix[i, j] == 100:
                observed_matrix[i,j] = 1


    index_to_node = {value: key for key, value in node_to_index.items()}


    with pm.Model() as model:
        beta = pm.Normal('beta', mu=0, sigma=10)

        # Existing model definition for edges
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_prob = pm.Deterministic(f'edge_prob_{i}_{j}', pm.math.sigmoid(-beta * prior_distance_matrix[i, j]))
                    edge = pm.Bernoulli(f'edge_{i}_{j}', p=edge_prob, observed=observed_matrix[i,j])

                    # If both i and j are leaf nodes, add the likelihood for observed distances
        for i in range(num_leaf_nodes):
            for j in range(num_leaf_nodes):
               if is_leaf_node(prior_graph.edges(), index_to_node[i]) and is_leaf_node(prior_graph.edges(), index_to_node[j]):
                   # Assuming a normal distribution for the observed distances
                   distance_likelihood = pm.Normal(f'distance_{i}_{j}', 
                                                   mu=edge_prob,  # directly using the edge probability
                                                   sd=sigma,  # sigma is a hyperparameter to be defined
                                                   observed=observed_distances[i, j])

        # Sample from the posterior using MCMC
        idata = pm.sample(1000, tune=100, target_accept=0.95, initvals={'beta': 0.1})


    # Extract the edge probabilities from the trace
    edge_probabilities = np.zeros((num_nodes, num_nodes))
    threshold = 0.994  # This is an example threshold

    # Extract the edge probabilities from the trace
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                # Extracting the edge probabilities from the trace
                edge_probabilities[i, j] = np.mean(idata.posterior[f'edge_prob_{i}_{j}'].values)


    # Create a directed graph
    post_graph = nx.DiGraph()
    # Add nodes
    for i in range(num_nodes):
        post_graph.add_node(i)

    # Add edges based on the consensus_edges
    for i in range(num_nodes):
        for j in range(num_nodes):
            if edge_probabilities[i, j] > threshold:
                post_graph.add_edge(i, j)
                

    return idata, post_graph

