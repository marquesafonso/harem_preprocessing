import json, random

# Opening JSON file
def load_data():
    with open('./CDSegundoHAREMReRelEM-selective.json', encoding='utf-8')as f:
        data = json.load(f)
    return data

def write_csv(filename:str, data:list):
    with open(filename, mode='w', encoding='utf-8') as file:
        file.write(f'Text \t  Entities\n')
        for example in data:
            file.write(f"{example['p_text']} \t {example['entities']}\n")

def writer_json(docs:list, outfile:str):
    with open(outfile, 'w', encoding='utf-8') as f:
        json.dump(docs, f, ensure_ascii=False, indent=1)

all_docs = []
data = load_data()
for _ , data in enumerate(data):
    for _, example in enumerate(data["doc_ps"]):
        if len(example['p_text']) > 0:
            all_docs += [example]

writer_json(all_docs, "harem_groundtruth.json")

random.shuffle(all_docs)
total_exemplos = len(all_docs)

train_set = all_docs[:round(total_exemplos*0.7)]
test_set = all_docs[round(total_exemplos*0.7):]

# create a train.csv and test.csv
write_csv('train.csv', train_set)
write_csv('test.csv', test_set)

# create a huggingface dataset for harem2