#%%
import pandas as pd
from ete3 import Tree
import numpy as np
from Bio.Phylo.TreeConstruction import DistanceMatrix
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
from Bio import Phylo
from io import StringIO
from collections import Counter
import torch
import re

from collections import Counter
# pooled_features = torch.rand(42, 84)

# pooled_features = pooled_features.numpy()

# taxonomy = pd.read_csv('/home/jamiesykes/Documents/PhyloNet/taxonomy_sorted_GNN.csv', header=0)
# labels = [random.randint(0, 20) for _ in range(42)]

# %%


class Node:
    def __init__(self, name):
        self.name = name
        self.children = []

    def get_or_create_child(self, name):
        for child in self.children:
            if child.name == name:
                return child
        new_child = Node(name)
        self.children.append(new_child)
        return new_child

    def to_ete(self):
        t = Tree()
        t.name = self.name
        for child in self.children:
            t.add_child(child.to_ete())
        return t

def generate_newick(data):
    root = Node('Root')

    # Traverse the DataFrame to build the tree structure
    for _, row in data.iterrows():
        current = root
        for col in data.columns:
            taxon = row[col]
            if pd.notna(taxon):
                current = current.get_or_create_child(taxon)

    # Convert to ete3 tree
    ete_tree = root.to_ete()

    return ete_tree

def trees(taxonomy, labels, pooled_features):
    labels = labels.detach().cpu().numpy().tolist()
    batch_taxonomy = taxonomy.loc[labels]

    # Compute squared distance matrix using broadcasting
    pooled_features = pooled_features.detach().cpu().numpy()
    square = np.sum(pooled_features**2, axis=1, keepdims=True)
    distance_squared = square + square.T - 2 * np.dot(pooled_features, pooled_features.T)
    distance_squared = np.clip(distance_squared, a_min=0, a_max=None)
    
    # Compute the distance matrix and set the diagonal to 0
    observed_distances = np.sqrt(distance_squared)
    np.fill_diagonal(observed_distances, 0)

    names = taxonomy.apply(lambda row: row.dropna().iloc[-1], axis=1).tolist()
    names = [names[i] for i in labels]

    # Create a Counter object to count occurrences
    counts = Counter(names)

    dumby_taxonomy = batch_taxonomy
    name_to_base_name = {}  # Map from full name with suffix to base name

    # Process list to append suffixes for duplicates
    unique_names = []
    for item in names:
        if counts[item] > 1:  # Check if the item is a duplicate
            suffix = counts[item]
            new_item = f"{item}{suffix}"
            counts[item] -= 1  # Decrease the count for the next occurrence
            unique_names.append(new_item)
            name_to_base_name[new_item] = item  # Map to base name

            # Subset row from taxonomy based on item in the last position
            duplicate_row = taxonomy.loc[taxonomy.iloc[:, -1] == item].copy()
            # Append species name in last position of duplicate row with count
            duplicate_row.iloc[:, -1] = new_item

            # Append duplicate row to dumby_taxonomy
            dumby_taxonomy = pd.concat([dumby_taxonomy, duplicate_row], ignore_index=True)

        else:
            unique_names.append(item)
            name_to_base_name[item] = item  # Map to base name

    target_ete_tree = generate_newick(dumby_taxonomy)

    distances_list = observed_distances.tolist()
    matrix = [row[:i+1] for i, row in enumerate(distances_list)]

    # Create a DistanceMatrix object
    dm = DistanceMatrix(names=unique_names, matrix=matrix)

    # Create a DistanceTreeConstructor object
    constructor = DistanceTreeConstructor()
    # Build the tree
    out_tree = constructor.nj(dm)

    # Create an in-memory file-like object
    string_io = StringIO()

    # Write the tree to the in-memory file in Newick format
    Phylo.write(out_tree, string_io, 'newick')

    # Get the string from the in-memory file
    out_tree_newick = string_io.getvalue()

    ete_pred_tree = Tree(out_tree_newick, format=1)
    ete_pred_tree.set_outgroup(ete_pred_tree.get_midpoint_outgroup())

    trees = {'target_tree': target_ete_tree, 'pred_tree': ete_pred_tree}
    
    return trees, name_to_base_name  # Return both trees and the name mapping

def generate_relationship_matrix(tree_type, tree, name_to_base_name):
    """Generate a relationship matrix from an ete3 Tree object."""
    leaf_names = tree.get_leaf_names()
    num_leaves = len(leaf_names)
    
    # Initialize the relationship matrix with zeros
    relationship_matrix = np.zeros((num_leaves, num_leaves))
    
    # Populate the matrix with distances
    for i in range(num_leaves):
        for j in range(num_leaves):
            base_name_i = name_to_base_name[leaf_names[i]]
            base_name_j = name_to_base_name[leaf_names[j]]
            if base_name_i != base_name_j:  # Compare using base names
      
                node1 = tree&leaf_names[i]
                node2 = tree&leaf_names[j]
                distance = node1.get_distance(node2)
                # if i contains a number, add 1
                if bool(re.search(r'\d', leaf_names[i])):
                    distance -= 1
             
                relationship_matrix[i, j] = distance

            else:
                # If they are the same base name, treat them as identical
                relationship_matrix[i, j] = 0  # Maximum relatedness

    # Scale the matrix
    relationship_matrix = relationship_matrix / relationship_matrix.max()
    # Invert the matrix
    relationship_matrix = 1 - relationship_matrix

    return pd.DataFrame(relationship_matrix, index=leaf_names, columns=leaf_names)

def generate_matrices(trees, name_to_base_name):
    """Generate relationship matrices for both target and predicted trees."""
    target_tree = trees['target_tree']
    pred_tree = trees['pred_tree']
    
    target_matrix = generate_relationship_matrix('target', target_tree, name_to_base_name)
    pred_matrix = generate_relationship_matrix('pred', pred_tree, name_to_base_name)
    
    return {'target_matrix': torch.tensor(target_matrix.values, dtype=torch.float32), 'pred_matrix': torch.tensor(pred_matrix.values, requires_grad=True, dtype=torch.float32)}

################################################################################################

def common_edges(ref_tree, source_tree):
    def get_edges(tree):
        # Extract edges as sets of normalized leaf names
        edges = set()
        for node in tree.traverse():
            if not node.is_leaf():
                leaves = set(re.sub(r'\d+$', '', leaf.name) for leaf in node)
                edges.add(frozenset(leaves))
        return edges

    def calculate_ref_edges_in_source(ref_tree, source_tree):
        # Get edges from both trees
        ref_edges = get_edges(ref_tree)
        source_edges = get_edges(source_tree)

        # Find common edges
        common_edges = ref_edges.intersection(source_edges)

        # Calculate proportion
        if len(ref_edges) == 0:
            return 0
        return len(common_edges) / len(ref_edges)
    
    # Calculate the metric
    metric = (1- calculate_ref_edges_in_source(ref_tree, source_tree)) * 100
    metric = torch.tensor(metric, dtype=torch.float, requires_grad=True)
    
    return metric

def ESS(ref_tree, source_tree):
    def normalize_name(name):
        return re.sub(r'\d+$', '', name)

    def get_edges(tree):
        return {node: set(normalize_name(leaf.name) for leaf in node) for node in tree.traverse() if not node.is_leaf()}

    def edge_similarity(edge1, edge2):
        # Implement your similarity metric here, e.g., based on shared descendants
        shared_descendants = edge1.intersection(edge2)

        return len(shared_descendants) / max(len(edge1), len(edge2))

    def calculate_edge_similarity_score(tree1, tree2):
        edges1 = get_edges(tree1)
        edges2 = get_edges(tree2)

        total_score = 0
        for edge in edges1.values():

            best_score = max(edge_similarity(edge, e) for e in edges2.values())
            total_score += best_score
        # Normalize the score
        normalized_score = total_score / len(edges1)
        return normalized_score
    
    score = calculate_edge_similarity_score(ref_tree, source_tree)
    score = torch.tensor(score, dtype=torch.float, requires_grad=True)

    return -score



def LeafDist(output_tree):
    def normalize_name(name):
        # Remove numerical suffixes and lower case for case-insensitive grouping
        return re.sub(r'\d+$', '', name).lower().rsplit('_', 1)[0]

    def group_nodes_by_prefix(tree):
        groups = {}
        for leaf in tree.iter_leaves():
            prefix = normalize_name(leaf.name)
            groups.setdefault(prefix, []).append(leaf)
        return groups

    def sum_distances_within_groups(tree):
        groups = group_nodes_by_prefix(tree)
        total_distance = 0

        for group, leaves in groups.items():
            for i in range(len(leaves)):
                for j in range(i + 1, len(leaves)):
                    distance = tree.get_distance(leaves[i], leaves[j])
                    total_distance += distance
        return total_distance

    total_distance = sum_distances_within_groups(output_tree)
    LeafDist = torch.tensor(total_distance, dtype=torch.float, requires_grad=True)

    return LeafDist