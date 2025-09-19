import os
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset

# create a huggingface dataset for harem2 and push it to the hub
def main():
    load_dotenv()
    login(token=os.getenv("HF_TOKEN"))
    dataset = load_dataset("json", data_files={"selective" : "CDSegundoHAREM-selective.json", "total": "CDSegundoHAREM-total.json"})
    dataset.push_to_hub(repo_id=f"{os.getenv('HF_USER')}/SegundoHAREM")


if __name__ == '__main__':
    main()