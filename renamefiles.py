import os

def rename_files(folder_path, new_prefix):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Iterate through each file and rename it
    for old_name in files:
        # Construct the new file name by adding a prefix
        new_name = new_prefix + old_name

        # Create the full file paths
        old_path = os.path.join(folder_path, old_name)
        new_path = os.path.join(folder_path, new_name)

        # Rename the file
        os.rename(old_path, new_path)
        print(f'Renamed: {old_name} to {new_name}')

# Example usage: replace 'folder_path' with the path to your folder and 'new_prefix' with your desired prefix
folder_path = 'rs_img4'
new_prefix = 'other_'
rename_files(folder_path, new_prefix)
