import json
import faiss
import numpy as np
import ollama
from neo4j import GraphDatabase

# Load FAISS and chunks
index = faiss.read_index("vector.index")
with open("chunks.json", "r") as f:
    data = json.load(f)
texts = data["texts"]
ids = data["ids"]

# Neo4j
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "neo4j"))

def embed_query_ollama(question):
    response = ollama.embeddings(model="mistral", prompt=question)
    return response['embedding']

def search_vectors(query_embedding, top_k=3):
    query = np.array(query_embedding).astype('float32').reshape(1, -1)
    D, I = index.search(query, top_k)
    return [texts[i] for i in I[0]]

def query_graph(keywords):
    with driver.session() as session:
        query = """
        MATCH (c:Class)-[:OWNS]->(m:Method)
        OPTIONAL MATCH (c)-[:HAS_ANNOTATION]->(ca:Annotation)
        OPTIONAL MATCH (m)-[:HAS_ANNOTATION]->(ma:Annotation)
        WHERE any(kw IN $kwlist WHERE toLower(c.name) CONTAINS kw OR toLower(m.name) CONTAINS kw)
        RETURN c.name AS class, collect(DISTINCT m.name) AS methods, collect(DISTINCT ca.name) AS class_annotations, collect(DISTINCT ma.name) AS method_annotations
        LIMIT 5
        """
        result = session.run(query, kwlist=[k.lower() for k in keywords])
        context = ""
        for row in result:
            context += f"\nClass: {row['class']}\n"
            context += f"  Class Annotations: {', '.join(a for a in row['class_annotations'] if a)}\n"
            context += f"  Methods: {', '.join(row['methods'])}\n"
            context += f"  Method Annotations: {', '.join(a for a in row['method_annotations'] if a)}\n"
        return context

def ask_mistral(question, context):
    messages = [
        {"role": "system", "content": "You are a helpful Java Spring Boot code explainer."},
        {"role": "user", "content": f"""Context:\n{context}\n\nQ: {question}\nA:"""}
    ]
    response = ollama.chat(model="mistral", messages=messages)
    return response['message']['content']

def main():
    question = input("Ask a code-related question: ")
    query_embedding = embed_query_ollama(question)
    vector_contexts = search_vectors(query_embedding)
    keywords = question.lower().split()
    graph_context = query_graph(keywords)
    combined_context = "\n".join(vector_contexts) + "\n\n" + graph_context
    answer = ask_mistral(question, combined_context)
    print("\n🔎 Answer:\n", answer)

if __name__ == "__main__":
    main()
