import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from neo4j import GraphDatabase
import faiss
from dataclasses import dataclass
import requests
import time
import re
import xml.etree.ElementTree as ET
import uuid
from tree_sitter import Language, Parser

# Initialize tree-sitter for Java
JAVA_LANGUAGE = Language('build/java.so', 'java')
parser = Parser()
parser.set_language(JAVA_LANGUAGE)

@dataclass
class CodeChunk:
    file_path: str
    content: str
    start_line: int
    end_line: int
    chunk_type: str  # 'class', 'method', 'interface', 'module', 'config'
    name: str
    package: str = ""
    annotations: List[str] = None
    embedding: np.ndarray = None
    node_id: str = None  # Neo4j node ID

class OllamaEmbedding:
    def __init__(self, model_name='nomic-embed-text', base_url='http://localhost:11434'):
        self.model_name = model_name
        self.base_url = base_url
        self.session = requests.Session()
    
    def encode(self, texts: List[str], batch_size: int = 32, show_progress_bar: bool = True) -> np.ndarray:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            if show_progress_bar:
                print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            batch_embeddings = []
            for text in batch:
                try:
                    response = self.session.post(
                        f"{self.base_url}/api/embeddings",
                        json={"model": self.model_name, "prompt": text},
                        timeout=30
                    )
                    if response.status_code == 200:
                        embedding = response.json()['embedding']
                        batch_embeddings.append(embedding)
                    else:
                        print(f"Error getting embedding: {response.text}")
                        batch_embeddings.append([0.0] * 768)
                except Exception as e:
                    print(f"Error processing text: {e}")
                    batch_embeddings.append([0.0] * 768)
                time.sleep(0.1)
            embeddings.extend(batch_embeddings)
        return np.array(embeddings, dtype=np.float32)

class OllamaLLM:
    def __init__(self, model_name='mistral:7b', base_url='http://localhost:11434'):
        self.model_name = model_name
        self.base_url = base_url
        self.session = requests.Session()
    
    def generate(self, prompt: str, max_tokens: int = 1500, temperature: float = 0.3) -> str:
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                        "stop": ["Human:", "Assistant:", "\n\n---"]
                    },
                    "stream": False
                },
                timeout=180
            )
            if response.status_code == 200:
                return response.json()['response']
            else:
                return f"Error: {response.text}"
        except Exception as e:
            return f"Error generating response: {e}"

class SpringBootIndexer:
    def __init__(self, codebase_name: str = None, embedding_model='nomic-embed-text', 
                 neo4j_uri="bolt://localhost:7687", neo4j_user="neo4j", neo4j_password="password"):
        self.embedding_model = OllamaEmbedding(embedding_model)
        self.chunks = []
        self.index = None
        self.codebase_name = codebase_name
        self.modules = {}
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.setup_database()
    
    def setup_database(self):
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:CodeChunk) REQUIRE n.node_id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Module) REQUIRE n.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Codebase) REQUIRE n.name IS UNIQUE")
    
    def discover_modules(self, root_path: str) -> Dict[str, str]:
        modules = {}
        for root, dirs, files in os.walk(root_path):
            if 'pom.xml' in files:
                pom_path = os.path.join(root, 'pom.xml')
                try:
                    tree = ET.parse(pom_path)
                    root_elem = tree.getroot()
                    artifact_id = None
                    for elem in root_elem:
                        if elem.tag.endswith('artifactId'):
                            artifact_id = elem.text
                            break
                    if artifact_id:
                        modules[artifact_id] = root
                        with self.driver.session() as session:
                            session.run(
                                "MERGE (m:Module {name: $name, codebase_name: $codebase_name, path: $path})",
                                name=artifact_id, codebase_name=self.codebase_name, path=root
                            )
                        print(f"Found module: {artifact_id} at {root}")
                except Exception as e:
                    print(f"Error parsing pom.xml at {pom_path}: {e}")
        self.modules = modules
        return modules
    
    def parse_java_file(self, file_path: str, module_name: str = "") -> List[CodeChunk]:
        chunks = []
        relationships = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            lines = content.split('\n')
            
            # Skip test files
            if '/src/test/' in file_path.replace('\\', '/') or 'test' in file_path.lower():
                print(f"Skipping test file: {file_path}")
                return chunks
            
            # Parse with tree-sitter
            tree = parser.parse(content.encode('utf-8'))
            root_node = tree.root_node
            
            # Extract package
            package_name = ""
            for node in root_node.children:
                if node.type == 'package_declaration':
                    package_name = content[node.start_byte:node.end_byte].split('package ')[1].strip(';\n')
            
            spring_annotations = [
                'SpringBootApplication', 'RestController', 'Controller', 'Service', 
                'Repository', 'Component', 'Configuration', 'Entity', 'Table',
                'RequestMapping', 'GetMapping', 'PostMapping', 'PutMapping', 'DeleteMapping',
                'Autowired', 'Value', 'ConfigurationProperties', 'EnableAutoConfiguration'
            ]
            
            def get_node_text(node):
                return content[node.start_byte:node.end_byte]
            
            def get_line_numbers(node):
                start_line = content[:node.start_byte].count('\n') + 1
                end_line = content[:node.end_byte].count('\n') + 1
                return start_line, end_line
            
            # Process class, interface, enum declarations
            for node in root_node.walk():
                if node.type in ['class_declaration', 'interface_declaration', 'enum_declaration']:
                    class_name = ""
                    extends = []
                    implements = []
                    annotations = []
                    
                    # Get class/interface/enum name
                    for child in node.children:
                        if child.type == 'identifier':
                            class_name = get_node_text(child)
                        elif child.type == 'superclass':
                            for ext in child.children:
                                if ext.type == 'identifier':
                                    extends.append(get_node_text(ext))
                        elif child.type == 'super_interfaces':
                            for impl in child.children:
                                if impl.type == 'identifier':
                                    implements.append(get_node_text(impl))
                        elif child.type == 'annotation':
                            ann_name = get_node_text(child).split('(')[0].lstrip('@')
                            if ann_name in spring_annotations:
                                annotations.append(f'@{ann_name}')
                    
                    # Skip test classes
                    if 'test' in class_name.lower():
                        print(f"Skipping test class: {class_name} in {file_path}")
                        continue
                    
                    chunk_type = 'class' if node.type == 'class_declaration' else 'interface' if node.type == 'interface_declaration' else 'enum'
                    start_line, end_line = get_line_numbers(node)
                    class_content = get_node_text(node)
                    
                    enhanced_content = f"// Module: {module_name}\n// Package: {package_name}\n"
                    if annotations:
                        enhanced_content += f"// Annotations: {', '.join(annotations)}\n"
                    enhanced_content += f"// {chunk_type.title()}: {class_name}\n\n"
                    enhanced_content += class_content
                    
                    node_id = str(uuid.uuid4())
                    chunk = CodeChunk(
                        file_path=file_path,
                        content=enhanced_content,
                        start_line=start_line,
                        end_line=end_line,
                        chunk_type=chunk_type,
                        name=class_name,
                        package=package_name,
                        annotations=annotations,
                        embedding=None,
                        node_id=node_id
                    )
                    chunks.append(chunk)
                    
                    for ext in extends:
                        relationships.append(('EXTENDS', class_name, ext))
                    for impl in implements:
                        relationships.append(('IMPLEMENTS', class_name, impl))
            
            # Process methods and method calls
            for node in root_node.walk():
                if node.type == 'method_declaration':
                    method_name = ""
                    annotations = []
                    for child in node.children:
                        if child.type == 'identifier':
                            method_name = get_node_text(child)
                        elif child.type == 'annotation':
                            ann_name = get_node_text(child).split('(')[0].lstrip('@')
                            if ann_name in spring_annotations:
                                annotations.append(f'@{ann_name}')
                    
                    if method_name in ['get', 'set', 'is'] or method_name[0].isupper():
                        continue
                    
                    start_line, end_line = get_line_numbers(node)
                    method_content = get_node_text(node)
                    if len(method_content) > 50:
                        enhanced_content = f"// Module: {module_name}\n// Package: {package_name}\n"
                        if annotations:
                            enhanced_content += f"// Annotations: {', '.join(annotations)}\n"
                        enhanced_content += f"// Method: {method_name}\n\n"
                        enhanced_content += method_content
                        
                        node_id = str(uuid.uuid4())
                        chunk = CodeChunk(
                            file_path=file_path,
                            content=enhanced_content,
                            start_line=start_line,
                            end_line=end_line,
                            chunk_type='method',
                            name=method_name,
                            package=package_name,
                            annotations=annotations,
                            embedding=None,
                            node_id=node_id
                        )
                        chunks.append(chunk)
                        
                        # Find method calls
                        for child in node.walk():
                            if child.type == 'method_invocation':
                                called_method = ""
                                for subchild in child.children:
                                    if subchild.type == 'identifier':
                                        called_method = get_node_text(subchild)
                                        break
                                if called_method:
                                    relationships.append(('CALLS', method_name, called_method))
            
            # File summary
            file_summary = f"// Module: {module_name}\n// File: {os.path.basename(file_path)}\n"
            file_summary += f"// Package: {package_name}\n// Path: {file_path}\n\n"
            file_summary += content[:3000]
            
            node_id = str(uuid.uuid4())
            chunks.append(CodeChunk(
                file_path=file_path,
                content=file_summary,
                start_line=1,
                end_line=len(lines),
                chunk_type='module',
                name=os.path.basename(file_path),
                package=package_name,
                annotations=[],
                embedding=None,
                node_id=node_id
            ))
            
            # Store in Neo4j
            with self.driver.session() as session:
                for chunk in chunks:
                    session.run(
                        """
                        MERGE (c:CodeChunk {node_id: $node_id})
                        SET c.codebase_name = $codebase_name,
                            c.file_path = $file_path,
                            c.content = $content,
                            c.start_line = $start_line,
                            c.end_line = $end_line,
                            c.chunk_type = $chunk_type,
                            c.name = $name,
                            c.package = $package,
                            c.annotations = $annotations
                        """,
                        node_id=chunk.node_id,
                        codebase_name=self.codebase_name,
                        file_path=chunk.file_path,
                        content=chunk.content,
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                        chunk_type=chunk.chunk_type,
                        name=chunk.name,
                        package=chunk.package,
                        annotations=json.dumps(chunk.annotations or [])
                    )
                for rel_type, source, target in relationships:
                    session.run(
                        """
                        MATCH (source:CodeChunk {codebase_name: $codebase_name, name: $source_name})
                        MATCH (target:CodeChunk {codebase_name: $codebase_name, name: $target_name})
                        MERGE (source)-[:$rel_type]->(target)
                        """,
                        codebase_name=self.codebase_name,
                        source_name=source,
                        target_name=target,
                        rel_type=rel_type
                    )
        
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
        
        return chunks
    
    def parse_config_file(self, file_path: str, module_name: str = "") -> List[CodeChunk]:
        chunks = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            file_type = "properties" if file_path.endswith('.properties') else "yaml"
            enhanced_content = f"# Module: {module_name}\n# Configuration File ({file_type}): {os.path.basename(file_path)}\n"
            enhanced_content += f"# Path: {file_path}\n\n"
            enhanced_content += content
            
            node_id = str(uuid.uuid4())
            chunk = CodeChunk(
                file_path=file_path,
                content=enhanced_content,
                start_line=1,
                end_line=len(content.split('\n')),
                chunk_type='config',
                name=os.path.basename(file_path),
                package="",
                annotations=[],
                embedding=None,
                node_id=node_id
            )
            chunks.append(chunk)
            
            with self.driver.session() as session:
                session.run(
                    """
                    MERGE (c:CodeChunk {node_id: $node_id})
                    SET c.codebase_name = $codebase_name,
                        c.file_path = $file_path,
                        c.content = $content,
                        c.start_line = $start_line,
                        c.end_line = $end_line,
                        c.chunk_type = $chunk_type,
                        c.name = $name,
                        c.package = $package,
                        c.annotations = $annotations
                    """,
                    node_id=chunk.node_id,
                    codebase_name=self.codebase_name,
                    file_path=chunk.file_path,
                    content=chunk.content,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    chunk_type=chunk.chunk_type,
                    name=chunk.name,
                    package=chunk.package,
                    annotations=json.dumps(chunk.annotations or [])
                )
        
        except Exception as e:
            print(f"Error parsing config file {file_path}: {e}")
        
        return chunks
    
    def index_codebase(self, root_path: str):
        print(f"Indexing codebase '{self.codebase_name}' from: {root_path}")
        modules = self.discover_modules(root_path)
        if not modules:
            print("No Spring Boot modules found. Indexing as single module...")
            modules = {"main": root_path}
        
        for module_name, module_path in modules.items():
            print(f"Processing module: {module_name}")
            for root, dirs, files in os.walk(module_path):
                dirs[:] = [d for d in dirs if d not in ['.git', 'target', '.idea', '.vscode', 'node_modules']]
                for file in files:
                    file_path = os.path.join(root, file)
                    if file.endswith('.java'):
                        chunks = self.parse_java_file(file_path, module_name)
                        self.chunks.extend(chunks)
                    elif file in ['application.yml', 'application.properties', 'application-dev.yml', 
                                  'application-prod.yml', 'bootstrap.yml', 'bootstrap.properties']:
                        chunks = self.parse_config_file(file_path, module_name)
                        self.chunks.extend(chunks)
                    elif file.endswith(('.yml', '.yaml', '.properties')) and 'application' in file:
                        chunks = self.parse_config_file(file_path, module_name)
                        self.chunks.extend(chunks)
                    elif file == 'pom.xml':
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                pom_content = f.read()
                            enhanced_content = f"# Module: {module_name}\n# Maven POM: {file_path}\n\n"
                            enhanced_content += pom_content
                            node_id = str(uuid.uuid4())
                            chunk = CodeChunk(
                                file_path=file_path,
                                content=enhanced_content,
                                start_line=1,
                                end_line=len(pom_content.split('\n')),
                                chunk_type='config',
                                name='pom.xml',
                                package="",
                                annotations=[],
                                embedding=None,
                                node_id=node_id
                            )
                            self.chunks.append(chunk)
                            with self.driver.session() as session:
                                session.run(
                                    """
                                    MERGE (c:CodeChunk {node_id: $node_id})
                                    SET c.codebase_name = $codebase_name,
                                        c.file_path = $file_path,
                                        c.content = $content,
                                        c.start_line = $start_line,
                                        c.end_line = $end_line,
                                        c.chunk_type = $chunk_type,
                                        c.name = $name,
                                        c.package = $package,
                                        c.annotations = $annotations
                                    """,
                                    node_id=chunk.node_id,
                                    codebase_name=self.codebase_name,
                                    file_path=chunk.file_path,
                                    content=chunk.content,
                                    start_line=chunk.start_line,
                                    end_line=chunk.end_line,
                                    chunk_type=chunk.chunk_type,
                                    name=chunk.name,
                                    package=chunk.package,
                                    annotations=json.dumps(chunk.annotations or [])
                                )
                        except Exception as e:
                            print(f"Error parsing pom.xml at {file_path}: {e}")
        
        print(f"Found {len(self.chunks)} code chunks across {len(modules)} modules")
        if not self.chunks:
            print("No code chunks found. Check your path.")
            return
        
        contents = [chunk.content for chunk in self.chunks]
        embeddings = self.embedding_model.encode(contents, batch_size=8, show_progress_bar=True)
        
        for i, embedding in enumerate(embeddings):
            self.chunks[i].embedding = embedding
            with self.driver.session() as session:
                session.run(
                    """
                    MATCH (c:CodeChunk {node_id: $node_id})
                    SET c.embedding = $embedding
                    """,
                    node_id=self.chunks[i].node_id,
                    embedding=embedding.tolist()
                )
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        with self.driver.session() as session:
            session.run(
                """
                MERGE (cb:Codebase {name: $codebase_name})
                SET cb.root_path = $root_path,
                    cb.indexed_at = datetime(),
                    cb.total_chunks = $total_chunks
                """,
                codebase_name=self.codebase_name,
                root_path=root_path,
                total_chunks=len(self.chunks)
            )
        
        print(f"Indexing complete for codebase '{self.codebase_name}'!")
    
    def load_from_database(self):
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (c:CodeChunk {codebase_name: $codebase_name})
                RETURN c.node_id, c.file_path, c.content, c.start_line, c.end_line, 
                       c.chunk_type, c.name, c.package, c.annotations, c.embedding
                """,
                codebase_name=self.codebase_name
            )
            rows = result.values()
        
        if not rows:
            print(f"No existing index found for codebase '{self.codebase_name}'.")
            return False
        
        self.chunks = []
        embeddings = []
        
        for row in rows:
            embedding = np.array(row[9], dtype=np.float32) if row[9] else None
            chunk = CodeChunk(
                file_path=row[1],
                content=row[2],
                start_line=row[3],
                end_line=row[4],
                chunk_type=row[5],
                name=row[6],
                package=row[7],
                annotations=json.loads(row[8]),
                embedding=embedding,
                node_id=row[0]
            )
            self.chunks.append(chunk)
            if embedding is not None:
                embeddings.append(embedding)
        
        result = session.run(
            """
            MATCH (m:Module {codebase_name: $codebase_name})
            RETURN m.name, m.path
            """,
            codebase_name=self.codebase_name
        )
        self.modules = {row[0]: row[1] for row in result.values()}
        
        if embeddings:
            embeddings = np.vstack(embeddings)
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings.astype('float32'))
        
        print(f"Loaded {len(self.chunks)} chunks from {len(self.modules)} modules for codebase '{self.codebase_name}'")
        return True

class SpringBootAssistant:
    def __init__(self, indexer: SpringBootIndexer, llm_model='mistral:7b'):
        self.indexer = indexer
        self.llm = OllamaLLM(llm_model)
    
    def search_relevant_code(self, query: str, top_k: int = 8) -> List[Tuple[CodeChunk, float]]:
        if not self.indexer.index:
            print("No index available. Please run index_codebase() first.")
            return []
        
        query_embedding = self.indexer.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        scores, indices = self.indexer.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.indexer.chunks) and idx >= 0:
                results.append((self.indexer.chunks[idx], score))
        
        # Enrich with DFS traversal
        enriched_results = []
        visited = set()
        with self.indexer.driver.session() as session:
            for chunk, score in results:
                if chunk.node_id not in visited:
                    enriched_results.append((chunk, score))
                    visited.add(chunk.node_id)
                    # DFS to find related nodes (CALLS, EXTENDS, IMPLEMENTS)
                    result = session.run(
                        """
                        MATCH (start:CodeChunk {node_id: $node_id})
                        MATCH path = (start)-[:CALLS|EXTENDS|IMPLEMENTS*1..3]->(related:CodeChunk)
                        WHERE related.codebase_name = $codebase_name
                        RETURN related.node_id, related.file_path, related.content, 
                               related.start_line, related.end_line, related.chunk_type,
                               related.name, related.package, related.annotations, related.embedding
                        """,
                        node_id=chunk.node_id,
                        codebase_name=self.indexer.codebase_name
                    )
                    for record in result:
                        related_node_id = record[0]
                        if related_node_id not in visited:
                            related_chunk = CodeChunk(
                                file_path=record[1],
                                content=record[2],
                                start_line=record[3],
                                end_line=record[4],
                                chunk_type=record[5],
                                name=record[6],
                                package=record[7],
                                annotations=json.loads(record[8]),
                                embedding=np.array(record[9], dtype=np.float32) if record[9] else None,
                                node_id=related_node_id
                            )
                            enriched_results.append((related_chunk, score * 0.8))  # Lower score for related nodes
                            visited.add(related_node_id)
        
        return enriched_results[:top_k]
    
    def build_context(self, query: str, current_file: str = None, max_chars: int = 15000) -> str:
        relevant_chunks = self.search_relevant_code(query, top_k=12)
        context_parts = []
        total_chars = 0
        
        context_parts.append(f"=== SPRING BOOT PROJECT OVERVIEW ({self.indexer.codebase_name}) ===\n")
        context_parts.append(f"Modules: {', '.join(self.indexer.modules.keys())}\n\n")
        
        if current_file and os.path.exists(current_file):
            try:
                with open(current_file, 'r', encoding='utf-8') as f:
                    current_content = f.read()
                current_chars = len(current_content)
                if current_chars < max_chars // 3:
                    context_parts.append(f"=== CURRENT FILE ({current_file}) ===\n{current_content}\n\n")
                    total_chars += current_chars
            except Exception as e:
                print(f"Error reading current file: {e}")
        
        context_parts.append("=== RELEVANT CODE FROM CODEBASE ===\n")
        chunks_by_type = {}
        for chunk, score in relevant_chunks:
            if chunk.chunk_type not in chunks_by_type:
                chunks_by_type[chunk.chunk_type] = []
            chunks_by_type[chunk.chunk_type].append((chunk, score))
        
        type_order = ['config', 'class', 'interface', 'method', 'module']
        for chunk_type in type_order:
            if chunk_type in chunks_by_type:
                for chunk, score in chunks_by_type[chunk_type]:
                    annotations_str = f" [{', '.join(chunk.annotations)}]" if chunk.annotations else ""
                    chunk_text = f"\n--- {chunk.chunk_type.upper()}: {chunk.name}{annotations_str} ---\n"
                    chunk_text += f"File: {chunk.file_path}\n"
                    if chunk.package:
                        chunk_text += f"Package: {chunk.package}\n"
                    chunk_text += f"{chunk.content}\n"
                    chunk_chars = len(chunk_text)
                    if total_chars + chunk_chars > max_chars:
                        break
                    context_parts.append(chunk_text)
                    total_chars += chunk_chars
        
        return ''.join(context_parts)
    
    def ask_question(self, query: str, current_file: str = None) -> str:
        context = self.build_context(query, current_file)
        prompt = f"""You are an expert Spring Boot developer assistant. Answer the user's question based on the provided codebase context.

Focus on:
- Spring Boot specific patterns and annotations
- Multi-module project structure
- Configuration management
- RESTful API design
- Dependency injection patterns
- Database integration
- Security configurations

{context}

Question: {query}

Provide a detailed answer with specific references to the code, files, and Spring Boot concepts. If you mention specific classes or methods, include their package names."""
        print("Generating response with Mistral...")
        response = self.llm.generate(prompt, max_tokens=2000, temperature=0.2)
        return response

def add_index(codebase_name: str, path: str, embedding_model: str = 'nomic-embed-text', 
              neo4j_uri: str = "bolt://localhost:7687", neo4j_user: str = "neo4j", neo4j_password: str = "password") -> bool:
    """
    Index a Spring Boot codebase with a given name and path.
    
    Args:
        codebase_name: Unique identifier for the codebase
        path: Full path to the Spring Boot project root
        embedding_model: Ollama embedding model (default: 'nomic-embed-text')
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
    
    Returns:
        True if indexing was successful, False otherwise
    """
    try:
        if not os.path.exists(path):
            print(f"Error: Path '{path}' does not exist.")
            return False
        
        print(f"Starting indexing process for codebase: {codebase_name}")
        indexer = SpringBootIndexer(codebase_name, embedding_model, neo4j_uri, neo4j_user, neo4j_password)
        indexer.index_codebase(path)
        print(f"Successfully indexed codebase '{codebase_name}'")
        return True
    except Exception as e:
        print(f"Error indexing codebase '{codebase_name}': {e}")
        return False

def query(codebase_name: str, query_str: str, llm_model: str = 'mistral:7b', 
          embedding_model: str = 'nomic-embed-text', neo4j_uri: str = "bolt://localhost:7687", 
          neo4j_user: str = "neo4j", neo4j_password: str = "password") -> str:
    """
    Query a specific Spring Boot codebase with a question.
    
    Args:
        codebase_name: Name of the indexed codebase
        query_str: Question to ask about the codebase
        llm_model: Ollama LLM model (default: 'mistral:7b')
        embedding_model: Ollama embedding model (default: 'nomic-embed-text')
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
    
    Returns:
        Answer to the question as a string
    """
    try:
        indexer = SpringBootIndexer(codebase_name, embedding_model, neo4j_uri, neo4j_user, neo4j_password)
        if not indexer.load_from_database():
            return f"Error: No indexed codebase found with name '{codebase_name}'. Please index it first using add_index()."
        
        assistant = SpringBootAssistant(indexer, llm_model)
        response = assistant.ask_question(query_str)
        return response
    except Exception as e:
        return f"Error processing query for codebase '{codebase_name}': {e}"

def list_codebases(neo4j_uri: str = "bolt://localhost:7687", neo4j_user: str = "neo4j", neo4j_password: str = "password") -> List[Dict[str, str]]:
    """
    List all indexed codebases.
    
    Returns:
        List of dictionaries containing codebase information
    """
    codebases = []
    try:
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        with driver.session() as session:
            result = session.run(
                """
                MATCH (cb:Codebase)
                RETURN cb.name, cb.root_path, cb.indexed_at, cb.total_chunks
                """
            )
            for record in result:
                codebases.append({
                    'name': record[0],
                    'root_path': record[1],
                    'indexed_at': record[2],
                    'total_chunks': record[3]
                })
        driver.close()
    except Exception as e:
        print(f"Error listing codebases: {e}")
    return codebases

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Spring Boot Code Assistant with Neo4j and Tree-Sitter')
    parser.add_argument('--index', nargs=2, metavar=('CODEBASE_NAME', 'PATH'), 
                        help='Index a codebase: --index my_project /path/to/project')
    parser.add_argument('--query', nargs=2, metavar=('CODEBASE_NAME', 'QUESTION'), 
                        help='Ask a question: --query my_project "How does authentication work?"')
    parser.add_argument('--list', action='store_true', help='List all indexed codebases')
    parser.add_argument('--file', type=str, help='Current file for context')
    parser.add_argument('--embedding-model', default='nomic-embed-text', help='Ollama embedding model')
    parser.add_argument('--llm-model', default='mistral:7b', help='Ollama LLM model')
    parser.add_argument('--neo4j-uri', default='bolt://localhost:7687', help='Neo4j connection URI')
    parser.add_argument('--neo4j-user', default='neo4j', help='Neo4j username')
    parser.add_argument('--neo4j-password', default='password', help='Neo4j password')
    
    args = parser.parse_args()
    
    if args.list:
        codebases = list_codebases(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
        if not codebases:
            print("No indexed codebases found.")
        else:
            print("Indexed Codebases:")
            print("-" * 60)
            for cb in codebases:
                print(f"Name: {cb['name']}")
                print(f"Path: {cb['root_path']}")
                print(f"Indexed: {cb['indexed_at']}")
                print(f"Chunks: {cb['total_chunks']}")
                print("-" * 60)
        return
    
    if args.index:
        codebase_name, path = args.index
        success = add_index(codebase_name, path, args.embedding_model, args.neo4j_uri, args.neo4j_user, args.neo4j_password)
        if success:
            print(f"Successfully indexed codebase '{codebase_name}'")
        else:
            print(f"Failed to index codebase '{codebase_name}'")
        return
    
    if args.query:
        codebase_name, question = args.query
        print(f"Querying codebase '{codebase_name}'...")
        response = query(codebase_name, question, args.llm_model, args.embedding_model, 
                        args.neo4j_uri, args.neo4j_user, args.neo4j_password)
        print("\n" + "="*60)
        print(f"SPRING BOOT ASSISTANT ANSWER ({codebase_name}):")
 print("="*60)
        print(response)
        return
    
    codebases = list_codebases(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
    if not codebases:
        print("No indexed codebases found. Please index a codebase first:")
        print("python script.py --index <codebase_name> <path_to_project>")
        return
    
    print("Available codebases:")
    for i, cb in enumerate(codebases, 1):
        print(f"{i}. {cb['name']} ({cb['total_chunks']} chunks)")
    
    try:
        choice = int(input("\nSelect a codebase (number): ")) - 1
        if choice < 0 or choice >= len(codebases):
            print("Invalid selection.")
            return
        
        selected_codebase = codebases[choice]['name']
        print(f"\nSelected codebase: {selected_codebase}")
        
        indexer = SpringBootIndexer(selected_codebase, args.embedding_model, args.neo4j_uri, args.neo4j_user, args.neo4j_password)
        if not indexer.load_from_database():
            print(f"Error loading codebase '{selected_codebase}'")
            return
        
        assistant = SpringBootAssistant(indexer, args.llm_model)
        print(f"\nSpring Boot Assistant - Interactive Mode ({selected_codebase})")
        print(f"Modules: {', '.join(indexer.modules.keys())}")
        print("Type 'quit' to exit, 'file: /path/to/file.java' to set current file context")
        print("Type 'switch' to switch to another codebase")
        
        current_file = args.file
        while True:
            query = input(f"\n[{selected_codebase}] Question: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            if query.lower() == 'switch':
                print("\nAvailable codebases:")
                for i, cb in enumerate(codebases, 1):
                    print(f"{i}. {cb['name']} ({cb['total_chunks']} chunks)")
                try:
                    choice = int(input("\nSelect a codebase (number): ")) - 1
                    if 0 <= choice < len(codebases):
                        selected_codebase = codebases[choice]['name']
                        indexer = SpringBootIndexer(selected_codebase, args.embedding_model, args.neo4j_uri, args.neo4j_user, args.neo4j_password)
                        if indexer.load_from_database():
                            assistant = SpringBootAssistant(indexer, args.llm_model)
                            print(f"Switched to codebase: {selected_codebase}")
                        else:
                            print(f"Error loading codebase '{selected_codebase}'")
                    else:
                        print("Invalid selection.")
                except ValueError:
                    print("Invalid input.")
                continue
            if query.startswith('file:'):
                current_file = query[5:].strip()
                print(f"Current file set to: {current_file}")
                continue
            if not query:
                continue
            response = assistant.ask_question(query, current_file)
            print("\n" + "="*60)
            print(f"ANSWER ({selected_codebase}):")
            print("="*60)
            print(response)
    except ValueError:
        print("Invalid input.")
    except KeyboardInterrupt:
        print("\nGoodbye!")

if __name__ == "__main__":
    # Build tree-sitter Java language library if not already built
    if not os.path.exists('build/java.so'):
        from tree_sitter import Language
        Language.build_library('build/java.so', ['vendor/tree-sitter-java'])
    main()

# Example usage:
"""
# Index a codebase
add_index("my_ecommerce_app", "/path/to/ecommerce/project")

# Query a codebase
response = query("my_ecommerce_app", "How is user authentication implemented?")
print(response)

# List all codebases
codebases = list_codebases()
for cb in codebases:
    print(f"Codebase: {cb['name']}, Chunks: {cb['total_chunks']}")
"""
