from huggingface_hub import HfApi

api = HfApi()
api.create_repo("ai-text-detector", exist_ok=True)
api.upload_folder(
    folder_path="models/roberta-hc3-best",
    repo_id="AyoMax00358/ai-text-detector",
)
print("Upload complete!")
