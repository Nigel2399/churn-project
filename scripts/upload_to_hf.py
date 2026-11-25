"""Upload the trained model to the Hugging Face Hub.

Run:  python scripts/upload_to_hf.py YOUR-HF-USERNAME/churn-model
"""
import json
import sys

from huggingface_hub import HfApi, create_repo

repo_id = sys.argv[1]
metrics = json.load(open("models/metrics.json"))

create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
api = HfApi()

card = f"""# Churn prediction model

Scikit-learn pipeline ({metrics['best_model']}) trained on a synthetic customer churn dataset.

- Test ROC-AUC: {metrics['test']['roc_auc']:.3f}
- Test recall: {metrics['test']['recall']:.3f}
- Trained with scikit-learn 1.9.1
"""

api.upload_file(path_or_fileobj="models/model.joblib", path_in_repo="model.joblib", repo_id=repo_id)
api.upload_file(path_or_fileobj="models/metrics.json", path_in_repo="metrics.json", repo_id=repo_id)
api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo_id)

print(f"Uploaded -> https://huggingface.co/{repo_id}")