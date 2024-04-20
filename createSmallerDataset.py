import os
import shutil
from random import sample

#path to the directory of complete dataset
source_dir = '/Users/pranavgupta/Documents/Synopsys_2024/dataset/[train/test]/[FAKE/REAL]'

#path to directory of smaller dataset
dest_dir = '/Users/pranavgupta/Documents/Synopsys_2024/dataset[number]/[train/test]/[FAKE/REAL]'

#get a list of filenames in the source directory,
#'[]' is replaced with the number of filenames in the destination directory
filenames = os.listdir(source_dir)
sampled_filenames = sample(filenames, [])

#create the destination directory if it is not there
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

#copy the randomly sampled images to the destination directory
for filename in sampled_filenames:
    src_path = os.path.join(source_dir, filename)
    dest_path = os.path.join(dest_dir, filename)
    shutil.copy(src_path, dest_path)

print(f'Copied {len(sampled_filenames)} images to {dest_dir}')