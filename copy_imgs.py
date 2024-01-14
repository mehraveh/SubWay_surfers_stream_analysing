import os
import shutil

s_folder_path = "dino2"  # Replace with the path to your folder
d_folder_path = "rs_img2"  # Replace with the path to your folder

# List all files in the folder
s_files = os.listdir(s_folder_path)
d_files = os.listdir(d_folder_path)

# Iterate through the list of files
for file in d_files:
    # Print the name of each file in the folder
    print(file.split('.')[0])
    file = file.split('.')[0]
    if 'output' in file:
        shutil.copy2(s_folder_path+'/'+file + '.jpg', d_folder_path)

