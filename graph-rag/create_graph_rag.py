import os
import javalang
import faiss
import numpy as np
import json
from neo4j import GraphDatabase
import ollama

# --- Config ---
JAVA_SOURCE_DIR = "./src"  # <-- change this to your actual Spring Boot source root
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "neo4j"

# --- Vector store ---
embeddings = []
texts = []
id_map = []

# --- Neo4j driver ---
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

def embed_with_ollama(text):
    response = ollama.embeddings(model="mistral", prompt=text)
    return response['embedding']

def parse_annotations(declaration):
    if hasattr(declaration, 'annotations'):
        return [a.name for a in declaration.annotations]
    return []

def parse_and_embed():
    for root, _, files in os.walk(JAVA_SOURCE_DIR):
        for file in files:
            if file.endswith(".java"):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r') as f:
                        code = f.read()
                    tree = javalang.parse.parse(code)
                    for _, node in tree:
                        if isinstance(node, javalang.tree.ClassDeclaration):
                            class_name = node.name
                            annotations = parse_annotations(node)
                            class_code = code[node.position.line - 1:]
                            full_text = f"Class: {class_name}\nAnnotations: {annotations}\n{class_code[:500]}"
                            vector = embed_with_ollama(full_text[:2048])
                            embeddings.append(vector)
                            texts.append(full_text)
                            id_map.append(class_name)
                            store_graph(class_name, annotations, node)
                except Exception as e:
                    print(f"Parse error in {file}: {e}")

def store_graph(class_name, annotations, class_node):
    with driver.session() as session:
        session.run("MERGE (c:Class {name: $class_name})", class_name=class_name)
        for ann in annotations:
            session.run("""
                MERGE (a:Annotation {name: $annotation})
                WITH a
                MATCH (c:Class {name: $class_name})
                MERGE (c)-[:HAS_ANNOTATION]->(a)
            """, annotation=ann, class_name=class_name)

        for method in class_node.methods:
            method_name = method.name
            session.run("""
                MERGE (m:Method {name: $method_name})
                WITH m
                MATCH (c:Class {name: $class_name})
                MERGE (c)-[:OWNS]->(m)
            """, method_name=method_name, class_name=class_name)

            for ann in parse_annotations(method):
                session.run("""
                    MERGE (a:Annotation {name: $annotation})
                    WITH a
                    MATCH (m:Method {name: $method_name})
                    MERGE (m)-[:HAS_ANNOTATION]->(a)
                """, annotation=ann, method_name=method_name)

def save_faiss():
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings).astype('float32'))
    faiss.write_index(index, "vector.index")
    with open("chunks.json", "w") as f:
        json.dump({"texts": texts, "ids": id_map}, f)
    print("✅ Saved FAISS index and metadata.")

if __name__ == "__main__":
    parse_and_embed()
    save_faiss()
    print("✅ Graph + Embeddings (with annotations) generated.")
