# ------------------------------
# File 2: query_graph_rag.py
# ------------------------------
import json
import faiss
import numpy as np
import ollama
from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "neo4j"

# Load FAISS and chunks
def load_faiss_and_chunks(codebase):
    with open(f"chunks_{codebase}.json", "r") as f:
        data = json.load(f)
    index = faiss.read_index(f"vector_{codebase}.index")
    return index, data["texts"], data["ids"]

# Neo4j
neo_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

def embed_query_ollama(question):
    response = ollama.embeddings(model="mistral", prompt=question)
    return response['embedding']

def search_vectors(query_embedding, index, texts, top_k=3):
    query = np.array(query_embedding).astype('float32').reshape(1, -1)
    D, I = index.search(query, top_k)
    return [texts[i] for i in I[0]]

def query_graph(keywords, codebase):
    with neo_driver.session() as session:
        query = """
        MATCH (c:Class {codebase: $codebase})-[:OWNS]->(m:Method)
        OPTIONAL MATCH (c)-[:IMPLEMENTS]->(i:Interface)
        OPTIONAL MATCH (c)-[:EXTENDS]->(p:Class)
        OPTIONAL MATCH (m)-[:CALLS]->(callee:Method)
        WHERE any(kw IN $kwlist WHERE toLower(c.name) CONTAINS kw OR toLower(m.name) CONTAINS kw)
        RETURN c.name AS class, collect(DISTINCT m.name) AS methods,
               collect(DISTINCT i.name) AS interfaces,
               collect(DISTINCT p.name) AS parents,
               collect(DISTINCT callee.name) AS method_calls
        LIMIT 5
        """
        result = session.run(query, kwlist=[k.lower() for k in keywords], codebase=codebase)
        context = ""
        for row in result:
            context += f"\nClass: {row['class']}\n"
            context += f"  Implements: {', '.join(i for i in row['interfaces'] if i)}\n"
            context += f"  Extends: {', '.join(p for p in row['parents'] if p)}\n"
            context += f"  Methods: {', '.join(row['methods'])}\n"
            context += f"  Calls: {', '.join(c for c in row['method_calls'] if c)}\n"
        return context

def ask_mistral(question, context):
    messages = [
        {"role": "system", "content": "You are a helpful Java Spring Boot code explainer."},
        {"role": "user", "content": f"""Context:\n{context}\n\nQ: {question}\nA:"""}
    ]
    response = ollama.chat(model="mistral", messages=messages)
    return response['message']['content']

def main():
    codebase = input("Enter codebase name: ")
    question = input("Ask a code-related question: ")
    
    index, texts, ids = load_faiss_and_chunks(codebase)
    query_embedding = embed_query_ollama(question)
    vector_contexts = search_vectors(query_embedding, index, texts)
    keywords = question.lower().split()
    graph_context = query_graph(keywords, codebase)
    combined_context = "\n".join(vector_contexts) + "\n\n" + graph_context
    answer = ask_mistral(question, combined_context)
    print("\n🔎 Answer:\n", answer)

if __name__ == "__main__":
    main()
