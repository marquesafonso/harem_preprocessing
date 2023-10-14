import json, random, csv

# Opening JSON file
with open('./CDSegundoHAREMReRelEM-selective.json', encoding='utf-8')as f:
    data = json.load(f)

all_docs = []

for _ , data in enumerate(data):
    for _, example in enumerate(data["doc_ps"]):
        all_docs += [example]

random.shuffle(all_docs)
total_exemplos = len(all_docs)

train_set = all_docs[:round(total_exemplos*0.7)]
test_set = all_docs[round(total_exemplos*0.7):]

# create a train.csv and test.csv
# create a huggingface dataset for harem2