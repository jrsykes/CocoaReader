#%%
import re

def filter_slurm_log(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Pattern to identify the start of the section
    start_pattern = re.compile(r'^Relabeling images from:')
    # Pattern to identify the end of the section
    # end_pattern = re.compile(r'^Relabeled:  \d+  images$')
    end_pattern = re.compile(r'^Major epoch:  \d : \d$')

    in_section = False
    extracted_sections = []

    for line in lines:
        if start_pattern.match(line):
            in_section = True
            section = []

        if in_section:
            section.append(line)

        if end_pattern.match(line) and in_section:
            in_section = False
            extracted_sections.append(''.join(section))

    return extracted_sections

# Usage Example
file_path = '/users/jrs596/slurm-3096532.out'
extracted = filter_slurm_log(file_path)
for section in extracted:
    print(section)

# %%

# slurm-2321635.out
# slurm-2321720.out