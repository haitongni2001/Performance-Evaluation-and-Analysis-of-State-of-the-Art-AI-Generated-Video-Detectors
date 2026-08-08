from huggingface_hub import HfApi

api = HfApi()
files = api.list_repo_tree(
    repo_id="AIGVDBench/AIGVDBench",
    repo_type="dataset",
    recursive=True
)

for f in files:
    if f.path.endswith(".zip"):
        size_gb = f.size / (1024**3) if f.size else 0
        print(f"{f.path}\t{size_gb:.2f} GB")