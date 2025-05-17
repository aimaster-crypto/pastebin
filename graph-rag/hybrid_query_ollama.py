import json
import faiss
import numpy as np
import ollama
from neo4j import GraphDatabase

# Load FAISS and chunks once at start (you can cache this in a class or globally)
index = faiss.read_index("vector.index")
with open("chunks.json", "r") as f:
    data = json.load(f)
texts = data["texts"]  # Assumed to include module/class info prepended for clarity
ids = data["ids"]

# Neo4j driver - reuse
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "neo4j"))

def embed_query_ollama(question):
    response = ollama.embeddings(model="mistral", prompt=question)
    return response['embedding']

def search_vectors(query_embedding, top_k=3):
    query = np.array(query_embedding).astype('float32').reshape(1, -1)
    D, I = index.search(query, top_k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        # Include similarity score for transparency
        snippet = texts[idx]
        results.append(f"[Score: {dist:.4f}] {snippet}")
    return results

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
        found = False
        for row in result:
            found = True
            context += f"\nClass: {row['class']}\n"
            class_anns = ', '.join(a for a in row['class_annotations'] if a) or "None"
            method_anns = ', '.join(a for a in row['method_annotations'] if a) or "None"
            methods = ', '.join(row['methods']) or "None"
            context += f"  Class Annotations: {class_anns}\n"
            context += f"  Methods: {methods}\n"
            context += f"  Method Annotations: {method_anns}\n"
        return context if found else "No graph context matches found."

def ask_mistral(question, context):
    messages = [
        {"role": "system", "content": "You are a helpful Java Spring Boot code explainer."},
        {"role": "user", "content": f"Context:\n{context}\n\nQ: {question}\nA:"}
    ]
    response = ollama.chat(model="mistral", messages=messages)
    return response['message']['content']

def extract_keywords(text):
    # Simple keyword extraction: remove common stopwords could be added here
    stopwords = {"the", "is", "at", "which", "on", "and", "a", "an", "of", "in", "to", "for"}
    tokens = text.lower().split()
    keywords = [t for t in tokens if t not in stopwords]
    return keywords if keywords else tokens

def main():
    question = input("Ask a code-related question: ").strip()
    if not question:
        print("Please enter a valid question.")
        return

    query_embedding = embed_query_ollama(question)
    vector_contexts = search_vectors(query_embedding, top_k=5)
    keywords = extract_keywords(question)
    graph_context = query_graph(keywords)
    combined_context = "\n".join(vector_contexts) + "\n\nGraph Data:\n" + graph_context
    answer = ask_mistral(question, combined_context)
    print("\n🔎 Answer:\n", answer)

if __name__ == "__main__":
    main()
