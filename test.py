#%%
from ete3 import Tree
import pandas as pd

# Load the tree from the Newick file
tree = Tree("/users/jrs596/tree_pred.newick", format=1)

# Load the taxonomy CSV file
taxonomy = pd.read_csv('/users/jrs596/scratch/dat/flowers102_split/flowers_taxonomy.csv', header=0)

# List of taxonomy levels (headers)
headers = list(taxonomy.columns)


# Traverse the tree in reverse (postorder)
for node in tree.traverse("postorder"):
    # Only print nodes with children
    if node.children:
        print(f"\nParent: {node.name}")
        
        pred_parents = []
        for child in node.children:
            tax_level = None
            child_name = ''.join([i for i in child.name if not i.isdigit()])

            # Identify the correct taxonomic level for the child node
            for column in taxonomy.columns:
                #remove number characters from child name
                if child_name in taxonomy[column].values:
                    tax_level = column
                    break  # Exit the loop once tax level is found
            
            if tax_level is None:
                print(f"Taxonomy for child {child_name} not found in any taxonomic column.")
                continue
            
            # Get the index of the taxonomic level
            tax_level_index = headers.index(tax_level)
            
            # Get the parent taxonomic level (previous column in headers)
            if tax_level_index > 0:
                parent_level = headers[tax_level_index - 1]
            else:
                parent_level = None
            
            # Print the taxonomy row for the child
            row = taxonomy.loc[taxonomy[tax_level] == child_name]
            if not row.empty:
                print(f"Taxonomy row for child {child_name}:")
                print(row)
                
                if parent_level:
                    pred_parents.append(row[parent_level].values[0] if not row[parent_level].empty else 'N/A')
            else:
                print(f"No taxonomy row found for child {child_name}.")
        
        print("Predicted parents: ")
        print(pred_parents)

        if len(pred_parents) == 1:
            node.name = pred_parents[0]
        elif len(pred_parents) == 2 and pred_parents[0] == pred_parents[1]:
            node.name = pred_parents[0]

#show tree with internal node names
print(tree.get_ascii(attributes=["name"]))

#save tree to newick file
tree.write(outfile="/users/jrs596/scratch/dat/flowers102_split/tree_pred_named.newick", format=1)


# %%
