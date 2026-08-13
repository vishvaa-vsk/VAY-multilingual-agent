import chromadb
client = chromadb.PersistentClient(path='./chroma_db')
col = client.get_collection('nexatel_kb')
data = col.get(include=['documents','metadatas'])
for i, (doc, meta) in enumerate(zip(data['documents'], data['metadatas'])):
    print(f\"[{i}] {meta}\")
    print(doc[:150], '...\n')