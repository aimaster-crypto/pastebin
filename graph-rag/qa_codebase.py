import os
import ast
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import sqlite3
import faiss
from dataclasses import dataclass
import requests
import time

@dataclass
class CodeChunk:
    file_path: str
    content: str
    start_line: int
    end_line: int
    chunk_type: str  # 'function', 'class', 'module'
    name: str
    embedding: np.ndarray = None

class OllamaEmbedding:
    def __init__(self, model_name='nomic-embed-text', base_url='http://localhost:11434'):
        self.model_name = model_name
        self.base_url = base_url
        self.session = requests.Session()
        self.ensure_model_available()
    
    def ensure_model_available(self):
        """Ensure the embedding model is pulled and available"""
        try:
            # Check if model exists
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = [model['name'] for model in response.json().get('models', [])]
                if self.model_name not in models:
                    print(f"Pulling {self.model_name} model...")
                    pull_response = self.session.post(
                        f"{self.base_url}/api/pull",
                        json={"name": self.model_name}
                    )
                    if pull_response.status_code != 200:
                        raise Exception(f"Failed to pull model: {pull_response.text}")
                    print(f"Model {self.model_name} pulled successfully")
        except Exception as e:
            print(f"Warning: Could not verify model availability: {e}")
    
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
                        # Use zero vector as fallback
                        batch_embeddings.append([0.0] * 768)  # Default dimension
                        
                except Exception as e:
                    print(f"Error processing text: {e}")
                    batch_embeddings.append([0.0] * 768)
                
                # Small delay to avoid overwhelming Ollama
                time.sleep(0.1)
            
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings, dtype=np.float32)

class OllamaLLM:
    def __init__(self, model_name='mistral:7b', base_url='http://localhost:11434'):
        self.model_name = model_name
        self.base_url = base_url
        self.session = requests.Session()
        self.ensure_model_available()
    
    def ensure_model_available(self):
        """Ensure the LLM model is pulled and available"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = [model['name'] for model in response.json().get('models', [])]
                if self.model_name not in models:
                    print(f"Pulling {self.model_name} model...")
                    # pull_response = self.session.post(
                    #     f"{self.base_url}/api/pull",
                    #     json={"name": self.model_name}
                    # )
                    if pull_response.status_code != 200:
                        raise Exception(f"Failed to pull model: {pull_response.text}")
                    print(f"Model {self.model_name} pulled successfully")
        except Exception as e:
            print(f"Warning: Could not verify model availability: {e}")
    
    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.3) -> str:
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
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json()['response']
            else:
                return f"Error: {response.text}"
                
        except Exception as e:
            return f"Error generating response: {e}"

class CodebaseIndexer:
    def __init__(self, embedding_model='nomic-embed-text'):
        self.embedding_model = OllamaEmbedding(embedding_model)
        self.chunks = []
        self.index = None
        self.db_path = "codebase.db"
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
                embedding BLOB
            )
        ''')
        conn.commit()
        conn.close()
    
    def parse_python_file(self, file_path: str) -> List[CodeChunk]:
        """Extract functions and classes from Python files"""
        chunks = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            lines = content.split('\n')
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    start_line = node.lineno
                    end_line = node.end_lineno or start_line
                    chunk_content = '\n'.join(lines[start_line-1:end_line])
                    
                    # Add docstring and context
                    docstring = ast.get_docstring(node) or ""
                    enhanced_content = f"# {chunk.chunk_type}: {node.name}\n"
                    if docstring:
                        enhanced_content += f'"""{docstring}"""\n'
                    enhanced_content += chunk_content
                    
                    chunk = CodeChunk(
                        file_path=file_path,
                        content=enhanced_content,
                        start_line=start_line,
                        end_line=end_line,
                        chunk_type='function' if isinstance(node, ast.FunctionDef) else 'class',
                        name=node.name
                    )
                    chunks.append(chunk)
            
            # Add whole file as a chunk for context
            file_summary = f"# File: {os.path.basename(file_path)}\n"
            file_summary += f"# Path: {file_path}\n"
            file_summary += content[:2000]  # First 2000 chars
            
            chunks.append(CodeChunk(
                file_path=file_path,
                content=file_summary,
                start_line=1,
                end_line=len(lines),
                chunk_type='module',
                name=os.path.basename(file_path)
            ))
                    
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
        
        return chunks
    
    def parse_javascript_file(self, file_path: str) -> List[CodeChunk]:
        """Basic JavaScript/TypeScript parsing"""
        chunks = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # Simple regex-based parsing for functions and classes
            import re
            
            # Find function declarations
            func_pattern = r'(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>))'
            class_pattern = r'class\s+(\w+)'
            
            for match in re.finditer(func_pattern, content, re.MULTILINE):
                func_name = match.group(1) or match.group(2)
                start_line = content[:match.start()].count('\n') + 1
                
                # Find end of function (simplified)
                end_line = min(start_line + 50, len(lines))  # Limit to 50 lines
                
                chunk_content = '\n'.join(lines[start_line-1:end_line])
                
                chunks.append(CodeChunk(
                    file_path=file_path,
                    content=f"// Function: {func_name}\n{chunk_content}",
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type='function',
                    name=func_name
                ))
            
            # Add file summary
            file_summary = f"// File: {os.path.basename(file_path)}\n{content[:2000]}"
            chunks.append(CodeChunk(
                file_path=file_path,
                content=file_summary,
                start_line=1,
                end_line=len(lines),
                chunk_type='module',
                name=os.path.basename(file_path)
            ))
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
        
        return chunks
    
    def index_codebase(self, root_path: str, extensions: List[str] = ['.py', '.js', '.ts', '.java']):
        """Index entire codebase"""
        print("Scanning codebase...")
        
        for root, dirs, files in os.walk(root_path):
            # Skip common non-code directories
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', '.venv', 'dist', 'build']]
            
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    
                    if file.endswith('.py'):
                        chunks = self.parse_python_file(file_path)
                    elif file.endswith(('.js', '.ts')):
                        chunks = self.parse_javascript_file(file_path)
                    else:
                        # Generic file handling
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()[:2000]
                            chunks = [CodeChunk(
                                file_path=file_path,
                                content=f"# File: {os.path.basename(file_path)}\n{content}",
                                start_line=1,
                                end_line=content.count('\n') + 1,
                                chunk_type='module',
                                name=os.path.basename(file_path)
                            )]
                        except:
                            chunks = []
                    
                    self.chunks.extend(chunks)
        
        print(f"Found {len(self.chunks)} code chunks")
        
        if not self.chunks:
            print("No code chunks found. Check your path and file extensions.")
            return
        
        # Generate embeddings
        print("Generating embeddings with Ollama...")
        contents = [chunk.content for chunk in self.chunks]
        embeddings = self.embedding_model.encode(contents, batch_size=8, show_progress_bar=True)
        
        for i, embedding in enumerate(embeddings):
            self.chunks[i].embedding = embedding
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
        
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
        
        for chunk in self.chunks:
            cursor.execute('''
                INSERT INTO code_chunks 
                (file_path, content, start_line, end_line, chunk_type, name, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                chunk.file_path,
                chunk.content,
                chunk.start_line,
                chunk.end_line,
                chunk.chunk_type,
                chunk.name,
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
            embedding = np.frombuffer(row[7], dtype=np.float32)
            chunk = CodeChunk(
                file_path=row[1],
                content=row[2],
                start_line=row[3],
                end_line=row[4],
                chunk_type=row[5],
                name=row[6],
                embedding=embedding
            )
            self.chunks.append(chunk)
            embeddings.append(embedding)
        
        if embeddings:
            embeddings = np.vstack(embeddings)
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings.astype('float32'))
        
        conn.close()
        print(f"Loaded {len(self.chunks)} chunks from database")

class CodeAssistant:
    def __init__(self, indexer: CodebaseIndexer, llm_model='mistral:7b'):
        self.indexer = indexer
        self.llm = OllamaLLM(llm_model)
    
    def search_relevant_code(self, query: str, top_k: int = 5) -> List[Tuple[CodeChunk, float]]:
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
    
    def build_context(self, query: str, current_file: str = None, max_chars: int = 12000) -> str:
        """Build context for LLM prompt"""
        relevant_chunks = self.search_relevant_code(query, top_k=10)
        
        context_parts = []
        total_chars = 0
        
        # Add current file context if provided
        if current_file and os.path.exists(current_file):
            try:
                with open(current_file, 'r', encoding='utf-8') as f:
                    current_content = f.read()
                current_chars = len(current_content)
                if current_chars < max_chars // 2:
                    context_parts.append(f"=== CURRENT FILE ({current_file}) ===\n{current_content}\n\n")
                    total_chars += current_chars
            except Exception as e:
                print(f"Error reading current file: {e}")
        
        # Add relevant chunks
        context_parts.append("=== RELEVANT CODE FROM CODEBASE ===\n")
        
        for chunk, score in relevant_chunks:
            chunk_text = f"\n--- {chunk.file_path} ({chunk.chunk_type}: {chunk.name}) ---\n{chunk.content}\n"
            chunk_chars = len(chunk_text)
            
            if total_chars + chunk_chars > max_chars:
                break
                
            context_parts.append(chunk_text)
            total_chars += chunk_chars
        
        return ''.join(context_parts)
    
    def ask_question(self, query: str, current_file: str = None) -> str:
        """Answer question about codebase"""
        context = self.build_context(query, current_file)
        
        prompt = f"""You are a helpful code assistant with deep knowledge of software engineering. Answer the user's question based on the provided codebase context.

{context}

Question: {query}

Please provide a clear, detailed answer based on the code shown above. If you reference specific functions or files, mention their names clearly."""
        
        print("Generating response...")
        response = self.llm.generate(prompt, max_tokens=1500, temperature=0.3)
        return response

# CLI Interface
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Local Code Assistant with Ollama')
    parser.add_argument('--index', type=str, help='Path to codebase to index')
    parser.add_argument('--query', type=str, help='Question to ask about the codebase')
    parser.add_argument('--file', type=str, help='Current file for context')
    parser.add_argument('--embedding-model', default='nomic-embed-text', help='Ollama embedding model')
    parser.add_argument('--llm-model', default='mistral:7b', help='Ollama LLM model')
    
    args = parser.parse_args()
    
    # Initialize the system
    print(f"Initializing with embedding model: {args.embedding_model}")
    indexer = CodebaseIndexer(args.embedding_model)
    
    if args.index:
        # Index the codebase
        print(f"Indexing codebase at: {args.index}")
        indexer.index_codebase(args.index)
        print("Indexing complete! You can now ask questions.")
        return
    
    # Load existing index
    indexer.load_from_database()
    
    if not indexer.chunks:
        print("No indexed codebase found. Please run with --index /path/to/your/code first.")
        return
    
    # Create assistant
    print(f"Initializing assistant with LLM model: {args.llm_model}")
    assistant = CodeAssistant(indexer, args.llm_model)
    
    if args.query:
        # Answer single query
        response = assistant.ask_question(args.query, args.file)
        print("\n" + "="*50)
        print("ANSWER:")
        print("="*50)
        print(response)
    else:
        # Interactive mode
        print("\nInteractive mode - type 'quit' to exit")
        print("You can also specify a current file with 'file: /path/to/file.py'")
        current_file = None
        
        while True:
            query = input("\nQuestion: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if query.startswith('file:'):
                current_file = query[5:].strip()
                print(f"Current file set to: {current_file}")
                continue
            
            if not query:
                continue
            
            response = assistant.ask_question(query, current_file)
            print("\n" + "="*50)
            print("ANSWER:")
            print("="*50)
            print(response)

if __name__ == "__main__":
    main()
