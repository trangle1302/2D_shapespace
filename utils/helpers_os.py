import os

def file_exists_and_size(file_path, extensions, min_size=4 * 1024 * 1024):
    for ext in extensions:
        full_path = f"{file_path}{ext}"
        if os.path.exists(full_path) and os.path.getsize(full_path) > min_size:
            return True
    return False

def find_files_with_patterns(directory, patterns):
    matching_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if all(pattern in file for pattern in patterns):
                matching_files.append(os.path.join(root, file))
    return matching_files
