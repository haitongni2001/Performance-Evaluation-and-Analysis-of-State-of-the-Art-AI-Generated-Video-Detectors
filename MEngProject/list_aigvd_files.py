from huggingface_hub import list_repo_files

files = list_repo_files("AIGVDBench/AIGVDBench", repo_type="dataset")

for f in files:
    print(f)