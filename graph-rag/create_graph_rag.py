# ------------------------------
# File 1: save_graph_rag.py
# ------------------------------
import os
import json
import javalang
import faiss
import numpy as np
from neo4j import GraphDatabase
import ollama

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "neo4j"

# Connect to Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

EMBEDDINGS = []
TEXTS = []
IDS = []

# Load or initialize FAISS
FAISS_INDEX_PATH = "vector.index"
DIMENSION = 4096  # Adjust based on Mistral's embedding output
if os.path.exists(FAISS_INDEX_PATH):
    index = faiss.read_index(FAISS_INDEX_PATH)
else:
    index = faiss.IndexFlatL2(DIMENSION)

def parse_java_file(filepath):
    with open(filepath, "r") as f:
        code = f.read()
    try:
        tree = javalang.parse.parse(code)
    except:
        return []

    parsed_classes = []
    for path, node in tree:
        if isinstance(node, javalang.tree.ClassDeclaration):
            class_name = node.name
            annotations = [a.name for a in node.annotations]
            implements = [i.name for i in node.implements]
            extends = node.extends.name if node.extends else None
            methods = []
            for member in node.body:
                if isinstance(member, javalang.tree.MethodDeclaration):
                    method_annotations = [a.name for a in member.annotations]
                    method_calls = []
                    if member.body:
                        for path2, node2 in member:
                            if isinstance(node2, javalang.tree.MethodInvocation):
                                method_calls.append(node2.member)
                    methods.append({
                        "name": member.name,
                        "annotations": method_annotations,
                        "calls": method_calls
                    })
            parsed_classes.append({
                "name": class_name,
                "annotations": annotations,
                "implements": implements,
                "extends": extends,
                "methods": methods,
                "filepath": filepath,
                "text": code
            })
    return parsed_classes

def embed_text(text):
    response = ollama.embeddings(model="mistral", prompt=text)
    return response['embedding']

def save_to_neo4j(parsed_classes, codebase):
    with driver.session() as session:
        for i, cls in enumerate(parsed_classes):
            chunk_text = cls['text']
            embedding = embed_text(chunk_text)
            EMBEDDINGS.append(embedding)
            TEXTS.append(chunk_text)
            IDS.append(f"{codebase}_{cls['name']}_{i}")

            session.run("""
                MERGE (c:Class {name: $name, codebase: $codebase})
                SET c.annotations = $annotations, c.chunk = $chunk, c.filepath = $filepath
            """, name=cls['name'], annotations=cls['annotations'], codebase=codebase,
                chunk=json.dumps(cls), filepath=cls['filepath'])

            for interface in cls['implements']:
                session.run("MERGE (i:Interface {name: $name})", name=interface)
                session.run("""
                    MATCH (c:Class {name: $cls, codebase: $codebase}), (i:Interface {name: $iface})
                    MERGE (c)-[:IMPLEMENTS]->(i)
                """, cls=cls['name'], iface=interface, codebase=codebase)

            if cls['extends']:
                session.run("MERGE (super:Class {name: $parent})", parent=cls['extends'])
                session.run("""
                    MATCH (child:Class {name: $child, codebase: $codebase}), (super:Class {name: $parent})
                    MERGE (child)-[:EXTENDS]->(super)
                """, child=cls['name'], parent=cls['extends'], codebase=codebase)

            for method in cls['methods']:
                session.run("""
                    MERGE (m:Method {name: $mname})
                    WITH m MATCH (c:Class {name: $cname, codebase: $codebase})
                    MERGE (c)-[:OWNS]->(m)
                """, mname=method['name'], cname=cls['name'], codebase=codebase)

                for call in method['calls']:
                    session.run("MERGE (callee:Method {name: $called})", called=call)
                    session.run("""
                        MATCH (caller:Method {name: $caller}), (callee:Method {name: $called})
                        MERGE (caller)-[:CALLS]->(callee)
                    """, caller=method['name'], called=call)

def process_directory(directory, codebase):
    all_classes = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".java"):
                parsed = parse_java_file(os.path.join(root, file))
                all_classes.extend(parsed)
    save_to_neo4j(all_classes, codebase)

    # Save embeddings
    emb_array = np.array(EMBEDDINGS).astype('float32')
    index.add(emb_array)
    faiss.write_index(index, FAISS_INDEX_PATH)

    with open("chunks.json", "w") as f:
        json.dump({"texts": TEXTS, "ids": IDS}, f)

    print(f"✅ Saved {len(all_classes)} classes and embeddings to Neo4j and FAISS under codebase '{codebase}'")

if __name__ == "__main__":
    directory = input("Enter path to Java codebase: ")
    codebase = input("Enter codebase name (used for tagging): ")
    process_directory(directory, codebase)
