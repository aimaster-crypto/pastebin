import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import sqlite3
import faiss
from dataclasses import dataclass
import requests
import time
import re
import xml.etree.ElementTree as ET

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

class OllamaEmbedding:
    def __init__(self, model_name='nomic-embed-text', base_url='http://localhost:11434'):
        self.model_name = model_name
        self.base_url = base_url
        self.session = requests.Session()
    
    def encode(self, texts: List[str], batch_size: int = 32, show_progress_bar: bool = True) -> np.ndarray:
        """Generate embeddings for a list of texts"""
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
                        json={
                            "model": self.model_name,
                            "prompt": text
                        },
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
        """Generate response using Ollama"""
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
    def __init__(self, embedding_model='nomic-embed-text'):
        self.embedding_model = OllamaEmbedding(embedding_model)
        self.chunks = []
        self.index = None
        self.db_path = "springboot_codebase.db"
        self.modules = {}  # Track Spring Boot modules
        self.setup_database()
    
    def setup_database(self):
        """Initialize SQLite database for metadata storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS code_chunks (
                id INTEGER PRIMARY KEY,
                file_path TEXT,
                content TEXT,
                start_line INTEGER,
                end_line INTEGER,
                chunk_type TEXT,
                name TEXT,
                package TEXT,
                annotations TEXT,
                module_name TEXT,
                embedding BLOB
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS modules (
                id INTEGER PRIMARY KEY,
                module_name TEXT UNIQUE,
                module_path TEXT,
                pom_content TEXT,
                dependencies TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def discover_modules(self, root_path: str) -> Dict[str, str]:
        """Discover Spring Boot modules by finding pom.xml files"""
        modules = {}
        
        for root, dirs, files in os.walk(root_path):
            if 'pom.xml' in files:
                pom_path = os.path.join(root, 'pom.xml')
                try:
                    tree = ET.parse(pom_path)
                    root_elem = tree.getroot()
                    
                    # Extract artifactId as module name
                    artifact_id = None
                    for elem in root_elem:
                        if elem.tag.endswith('artifactId'):
                            artifact_id = elem.text
                            break
                    
                    if artifact_id:
                        modules[artifact_id] = root
                        print(f"Found module: {artifact_id} at {root}")
                
                except Exception as e:
                    print(f"Error parsing pom.xml at {pom_path}: {e}")
        
        self.modules = modules
        return modules
    
    def parse_java_file(self, file_path: str, module_name: str = "") -> List[CodeChunk]:
        """Parse Java files with Spring Boot specific annotations"""
        chunks = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # Extract package declaration
            package_pattern = r'package\s+([\w\.]+);'
            package_match = re.search(package_pattern, content)
            package_name = package_match.group(1) if package_match else ""
            
            # Find Spring Boot annotations
            spring_annotations = [
                '@SpringBootApplication', '@RestController', '@Controller', '@Service', 
                '@Repository', '@Component', '@Configuration', '@Entity', '@Table',
                '@RequestMapping', '@GetMapping', '@PostMapping', '@PutMapping', '@DeleteMapping',
                '@Autowired', '@Value', '@ConfigurationProperties', '@EnableAutoConfiguration'
            ]
            
            # Find classes and interfaces
            class_pattern = r'(?:@[\w\(\)=",\s]+\s*)*(?:public\s+|private\s+|protected\s+)?(?:abstract\s+)?(class|interface|enum)\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w\s,]+)?\s*\{'
            
            for match in re.finditer(class_pattern, content, re.MULTILINE | re.DOTALL):
                class_type = match.group(1)
                class_name = match.group(2)
                start_pos = match.start()
                start_line = content[:start_pos].count('\n') + 1
                
                # Find matching closing brace (simplified)
                brace_count = 0
                end_pos = start_pos
                for i, char in enumerate(content[start_pos:], start_pos):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i
                            break
                
                end_line = content[:end_pos].count('\n') + 1
                class_content = content[start_pos:end_pos + 1]
                
                # Extract annotations for this class
                annotations_before = content[max(0, start_pos - 500):start_pos]
                found_annotations = []
                for annotation in spring_annotations:
                    if annotation in annotations_before:
                        found_annotations.append(annotation)
                
                # Enhanced content with context
                enhanced_content = f"// Module: {module_name}\n"
                enhanced_content += f"// Package: {package_name}\n"
                if found_annotations:
                    enhanced_content += f"// Annotations: {', '.join(found_annotations)}\n"
                enhanced_content += f"// {class_type.title()}: {class_name}\n\n"
                enhanced_content += class_content
                
                chunks.append(CodeChunk(
                    file_path=file_path,
                    content=enhanced_content,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type=class_type,
                    name=class_name,
                    package=package_name,
                    annotations=found_annotations,
                    embedding=None
                ))
            
            # Find methods within classes
            method_pattern = r'(?:@[\w\(\)=",\s]+\s*)*(?:public\s+|private\s+|protected\s+)?(?:static\s+)?(?:final\s+)?[\w<>\[\]]+\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+[\w\s,]+)?\s*\{'
            
            for match in re.finditer(method_pattern, content, re.MULTILINE):
                method_name = match.group(1)
                # Skip constructors and common getters/setters
                if method_name in ['get', 'set', 'is'] or method_name[0].isupper():
                    continue
                
                start_pos = match.start()
                start_line = content[:start_pos].count('\n') + 1
                
                # Find method end (simplified)
                brace_count = 0
                end_pos = start_pos
                for i, char in enumerate(content[start_pos:], start_pos):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i
                            break
                
                end_line = content[:end_pos].count('\n') + 1
                method_content = content[start_pos:end_pos + 1]
                
                # Extract method annotations
                annotations_before = content[max(0, start_pos - 200):start_pos]
                found_annotations = []
                for annotation in spring_annotations:
                    if annotation in annotations_before:
                        found_annotations.append(annotation)
                
                if len(method_content) > 50:  # Skip very short methods
                    enhanced_content = f"// Module: {module_name}\n"
                    enhanced_content += f"// Package: {package_name}\n"
                    if found_annotations:
                        enhanced_content += f"// Annotations: {', '.join(found_annotations)}\n"
                    enhanced_content += f"// Method: {method_name}\n\n"
                    enhanced_content += method_content
                    
                    chunks.append(CodeChunk(
                        file_path=file_path,
                        content=enhanced_content,
                        start_line=start_line,
                        end_line=end_line,
                        chunk_type='method',
                        name=method_name,
                        package=package_name,
                        annotations=found_annotations,
                        embedding=None
                    ))
            
            # Add file summary
            file_summary = f"// Module: {module_name}\n"
            file_summary += f"// File: {os.path.basename(file_path)}\n"
            file_summary += f"// Package: {package_name}\n"
            file_summary += f"// Path: {file_path}\n\n"
            file_summary += content[:3000]  # First 3000 chars
            
            chunks.append(CodeChunk(
                file_path=file_path,
                content=file_summary,
                start_line=1,
                end_line=len(lines),
                chunk_type='module',
                name=os.path.basename(file_path),
                package=package_name,
                annotations=[],
                embedding=None
            ))
                    
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
        
        return chunks
    
    def parse_config_file(self, file_path: str, module_name: str = "") -> List[CodeChunk]:
        """Parse configuration files (application.yml, application.properties, etc.)"""
        chunks = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            file_type = "properties" if file_path.endswith('.properties') else "yaml"
            
            enhanced_content = f"# Module: {module_name}\n"
            enhanced_content += f"# Configuration File ({file_type}): {os.path.basename(file_path)}\n"
            enhanced_content += f"# Path: {file_path}\n\n"
            enhanced_content += content
            
            chunks.append(CodeChunk(
                file_path=file_path,
                content=enhanced_content,
                start_line=1,
                end_line=len(content.split('\n')),
                chunk_type='config',
                name=os.path.basename(file_path),
                package="",
                annotations=[],
                embedding=None
            ))
            
        except Exception as e:
            print(f"Error parsing config file {file_path}: {e}")
        
        return chunks
    
    def index_codebase(self, root_path: str):
        """Index Spring Boot multi-module project"""
        print("Discovering Spring Boot modules...")
        modules = self.discover_modules(root_path)
        
        if not modules:
            print("No Spring Boot modules found. Indexing as single module...")
            modules = {"main": root_path}
        
        print("Scanning codebase...")
        
        for module_name, module_path in modules.items():
            print(f"Processing module: {module_name}")
            
            for root, dirs, files in os.walk(module_path):
                # Skip common non-code directories
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
                            
                            enhanced_content = f"# Module: {module_name}\n"
                            enhanced_content += f"# Maven POM: {file_path}\n\n"
                            enhanced_content += pom_content
                            
                            self.chunks.append(CodeChunk(
                                file_path=file_path,
                                content=enhanced_content,
                                start_line=1,
                                end_line=len(pom_content.split('\n')),
                                chunk_type='config',
                                name='pom.xml',
                                package="",
                                annotations=[],
                                embedding=None
                            ))
                        except Exception as e:
                            print(f"Error parsing pom.xml at {file_path}: {e}")
        
        print(f"Found {len(self.chunks)} code chunks across {len(modules)} modules")
        
        if not self.chunks:
            print("No code chunks found. Check your path.")
            return
        
        # Generate embeddings
        print("Generating embeddings with Ollama...")
        contents = [chunk.content for chunk in self.chunks]
        embeddings = self.embedding_model.encode(contents, batch_size=8, show_progress_bar=True)
        
        for i, embedding in enumerate(embeddings):
            self.chunks[i].embedding = embedding
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        # Save to database
        self.save_to_database()
        
        print("Indexing complete!")
    
    def save_to_database(self):
        """Save chunks and embeddings to SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clear existing data
        cursor.execute('DELETE FROM code_chunks')
        cursor.execute('DELETE FROM modules')
        
        # Save modules
        for module_name, module_path in self.modules.items():
            cursor.execute('''
                INSERT INTO modules (module_name, module_path)
                VALUES (?, ?)
            ''', (module_name, module_path))
        
        # Save chunks
        for chunk in self.chunks:
            cursor.execute('''
                INSERT INTO code_chunks 
                (file_path, content, start_line, end_line, chunk_type, name, package, annotations, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                chunk.file_path,
                chunk.content,
                chunk.start_line,
                chunk.end_line,
                chunk.chunk_type,
                chunk.name,
                chunk.package,
                json.dumps(chunk.annotations or []),
                chunk.embedding.tobytes()
            ))
        
        conn.commit()
        conn.close()
    
    def load_from_database(self):
        """Load existing index from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM code_chunks')
        rows = cursor.fetchall()
        
        if not rows:
            print("No existing index found. Please run index_codebase() first.")
            conn.close()
            return
        
        self.chunks = []
        embeddings = []
        
        for row in rows:
            embedding = np.frombuffer(row[9], dtype=np.float32)
            chunk = CodeChunk(
                file_path=row[1],
                content=row[2],
                start_line=row[3],
                end_line=row[4],
                chunk_type=row[5],
                name=row[6],
                package=row[7],
                annotations=json.loads(row[8]),
                embedding=embedding
            )
            self.chunks.append(chunk)
            embeddings.append(embedding)
        
        # Load modules
        cursor.execute('SELECT module_name, module_path FROM modules')
        module_rows = cursor.fetchall()
        self.modules = {row[0]: row[1] for row in module_rows}
        
        if embeddings:
            embeddings = np.vstack(embeddings)
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings.astype('float32'))
        
        conn.close()
        print(f"Loaded {len(self.chunks)} chunks from {len(self.modules)} modules")

class SpringBootAssistant:
    def __init__(self, indexer: SpringBootIndexer, llm_model='mistral:7b'):
        self.indexer = indexer
        self.llm = OllamaLLM(llm_model)
    
    def search_relevant_code(self, query: str, top_k: int = 8) -> List[Tuple[CodeChunk, float]]:
        """Search for relevant code chunks"""
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
        
        return results
    
    def build_context(self, query: str, current_file: str = None, max_chars: int = 15000) -> str:
        """Build context for LLM prompt with Spring Boot specific information"""
        relevant_chunks = self.search_relevant_code(query, top_k=12)
        
        context_parts = []
        total_chars = 0
        
        # Add project overview
        context_parts.append("=== SPRING BOOT PROJECT OVERVIEW ===\n")
        context_parts.append(f"Modules: {', '.join(self.indexer.modules.keys())}\n\n")
        
        # Add current file context if provided
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
        
        # Add relevant chunks grouped by type
        context_parts.append("=== RELEVANT CODE FROM CODEBASE ===\n")
        
        # Group chunks by type for better organization
        chunks_by_type = {}
        for chunk, score in relevant_chunks:
            if chunk.chunk_type not in chunks_by_type:
                chunks_by_type[chunk.chunk_type] = []
            chunks_by_type[chunk.chunk_type].append((chunk, score))
        
        # Add chunks in order of importance: config, class, method, module
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
        """Answer question about Spring Boot codebase"""
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

# CLI Interface
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Spring Boot Code Assistant with Ollama')
    parser.add_argument('--index', type=str, help='Path to Spring Boot project root to index')
    parser.add_argument('--query', type=str, help='Question to ask about the codebase')
    parser.add_argument('--file', type=str, help='Current file for context')
    parser.add_argument('--embedding-model', default='nomic-embed-text', help='Ollama embedding model')
    parser.add_argument('--llm-model', default='mistral:7b', help='Ollama LLM model')
    
    args = parser.parse_args()
    
    # Initialize the system
    print(f"Initializing Spring Boot indexer with embedding model: {args.embedding_model}")
    indexer = SpringBootIndexer(args.embedding_model)
    
    if args.index:
        # Index the codebase
        print(f"Indexing Spring Boot project at: {args.index}")
        indexer.index_codebase(args.index)
        print("Indexing complete! You can now ask questions about your Spring Boot application.")
        return
    
    # Load existing index
    indexer.load_from_database()
    
    if not indexer.chunks:
        print("No indexed codebase found. Please run with --index /path/to/your/springboot/project first.")
        return
    
    # Create assistant
    print(f"Initializing Spring Boot assistant with LLM model: {args.llm_model}")
    assistant = SpringBootAssistant(indexer, args.llm_model)
    
    if args.query:
        # Answer single query
        response = assistant.ask_question(args.query, args.file)
        print("\n" + "="*60)
        print("SPRING BOOT ASSISTANT ANSWER:")
        print("="*60)
        print(response)
    else:
        # Interactive mode
        print(f"\nSpring Boot Assistant - Interactive Mode")
        print(f"Modules: {', '.join(indexer.modules.keys())}")
        print("Type 'quit' to exit, 'file: /path/to/file.java' to set current file context")
        current_file = None
        
        while True:
            query = input("\nSpring Boot Question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if query.startswith('file:'):
                current_file = query[5:].strip()
                print(f"Current file set to: {current_file}")
                continue
            
            if not query:
                continue
            
            response = assistant.ask_question(query, current_file)
            print("\n" + "="*60)
            print("ANSWER:")
            print("="*60)
            print(response)

if __name__ == "__main__":
    main()
