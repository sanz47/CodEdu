import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model_name = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    BitsAndBytesConfig
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
from docx import Document
import warnings
warnings.filterwarnings("ignore")

class LocalEmbeddingFunction:
    """Custom embedding function using sentence-transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with a local sentence transformer model"""
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
    def __call__(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

class LocalPersonalFileAgent:
    def __init__(self, folder_path: str, person_name: str, model_name: str = "microsoft/DialoGPT-medium"):
        """
        Initialize the Local Personal File Agent
        
        Args:
            folder_path: Path to the folder containing files to index
            person_name: Name of the person this agent represents  
            model_name: HuggingFace model name for the LLM (default: DialoGPT-medium)
        """
        self.folder_path = Path(folder_path)
        self.person_name = person_name
        self.model_name = model_name
        
        # Initialize local LLM
        print("Loading local language model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Configure model for efficient loading
        if self.device == "cuda":
            # Use 8-bit quantization to reduce memory usage
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Initialize ChromaDB for vector storage with local embeddings
        self.chroma_client = chromadb.PersistentClient(path=f"./chroma_db_{person_name}")
        self.collection_name = f"{person_name}_files"
        
        # Initialize local embedding function
        self.embedding_function = LocalEmbeddingFunction()
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(
                name=self.collection_name
            )
        except:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name
            )
        
        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Supported file extensions
        self.supported_extensions = {'.txt', '.md', '.py', '.js', '.html', '.css', 
                                   '.json', '.xml', '.csv', '.pdf', '.docx', '.log'}
        
        # File hash tracking for updates
        self.file_hashes = self.load_file_hashes()
        
        print("Local Personal File Agent initialized successfully!")
        
    def load_file_hashes(self) -> Dict[str, str]:
        """Load existing file hashes to track changes"""
        hash_file = Path(f"file_hashes_{self.person_name}.json")
        if hash_file.exists():
            with open(hash_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_file_hashes(self):
        """Save file hashes to track changes"""
        hash_file = Path(f"file_hashes_{self.person_name}.json")
        with open(hash_file, 'w') as f:
            json.dump(self.file_hashes, f)
    
    def get_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except:
            return ""
    
    def load_document(self, file_path: Path) -> str:
        """Load content from various file types"""
        try:
            file_ext = file_path.suffix.lower()
            
            if file_ext == '.pdf':
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text()
                    return text
                        
            elif file_ext == '.docx':
                doc = Document(file_path)
                return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                
            else:
                # For text-based files
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    return file.read()
                    
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return ""
    
    def process_files(self):
        """Process all files in the folder and add them to the vector database"""
        print(f"Processing files for {self.person_name}...")
        
        files_processed = 0
        files_updated = 0
        
        for file_path in self.folder_path.rglob('*'):
            if not file_path.is_file():
                continue
                
            if file_path.suffix.lower() not in self.supported_extensions:
                continue
                
            # Check if file has been modified
            current_hash = self.get_file_hash(file_path)
            file_key = str(file_path.relative_to(self.folder_path))
            
            if file_key in self.file_hashes and self.file_hashes[file_key] == current_hash:
                continue  # File hasn't changed, skip
            
            # Load and process the file
            content = self.load_document(file_path)
            if not content.strip():
                continue
                
            # Split content into chunks
            chunks = self.text_splitter.split_text(content)
            
            # Remove old entries for this file if they exist
            try:
                existing_docs = self.collection.get(where={"file_path": file_key})
                if existing_docs['ids']:
                    self.collection.delete(ids=existing_docs['ids'])
            except:
                pass
            
            # Generate embeddings for chunks
            print(f"Processing {file_path.name}...")
            chunk_embeddings = self.embedding_function(chunks)
            
            # Add new chunks to the collection
            for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                doc_id = f"{file_key}_chunk_{i}"
                metadata = {
                    "file_path": file_key,
                    "file_name": file_path.name,
                    "file_type": file_path.suffix.lower(),
                    "chunk_index": i,
                    "person": self.person_name,
                    "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }
                
                self.collection.add(
                    documents=[chunk],
                    metadatas=[metadata],
                    ids=[doc_id],
                    embeddings=[embedding]
                )
            
            # Update file hash
            self.file_hashes[file_key] = current_hash
            files_processed += 1
            if file_key in self.file_hashes:
                files_updated += 1
                
            print(f"✓ Processed: {file_path.name}")
        
        self.save_file_hashes()
        print(f"Processing complete! {files_processed} files processed, {files_updated} updated.")
    
    def search_files(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search through the indexed files using local embeddings"""
        # Generate embedding for the query
        query_embedding = self.embedding_function([query])[0]
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        search_results = []
        if results['ids'][0]:  # Check if we have results
            for i in range(len(results['ids'][0])):
                search_results.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
        
        return search_results
    
    def generate_response_local(self, query: str, context_results: List[Dict[str, Any]]) -> str:
        """Generate a response using the local LLM"""
        
        # Prepare context from search results
        context_text = ""
        file_names = set()
        
        for result in context_results[:3]:  # Limit context to avoid token limits
            file_names.add(result['metadata']['file_name'])
            content = result['content'][:300] + "..." if len(result['content']) > 300 else result['content']
            context_text += f"\nFrom {result['metadata']['file_name']}:\n{content}\n"
        
        # Create a focused prompt for the local model
        prompt = f"""Context from {self.person_name}'s files:
{context_text}

Question: {query}

Based on the files mentioned above, here's what I found:"""

        try:
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            if self.device == "cuda":
                inputs = inputs.to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=150,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs)
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (after the prompt)
            response = full_response[len(prompt):].strip()
            
            # Add file references if response is too short or generic
            if len(response) < 20 or not response:
                if file_names:
                    response = f"Based on the information in {', '.join(file_names)}, I found relevant content. However, you may want to check these files directly for more detailed information."
                else:
                    response = "I couldn't find specific information about that in the available files."
            
            return response
            
        except Exception as e:
            return f"Error generating response: {e}"
    
    def ask(self, question: str) -> str:
        """Main method to ask questions about the person's files"""
        print(f"🔍 Searching {self.person_name}'s files for: {question}")
        
        # Search for relevant content
        search_results = self.search_files(question, n_results=5)
        
        if not search_results:
            return f"I couldn't find any relevant information in {self.person_name}'s files for your question."
        
        # Show which files were found
        found_files = set(result['metadata']['file_name'] for result in search_results)
        print(f"📁 Found relevant content in: {', '.join(found_files)}")
        
        # Generate response using local LLM
        response = self.generate_response_local(question, search_results)
        
        return response
    
    def list_files(self) -> List[str]:
        """List all indexed files"""
        try:
            all_docs = self.collection.get()
            files = set()
            for metadata in all_docs['metadatas']:
                files.add(metadata['file_name'])
            return sorted(list(files))
        except:
            return []
    
    def get_file_summary(self) -> str:
        """Get a summary of indexed files"""
        files = self.list_files()
        file_count = len(files)
        
        # Get file type distribution
        file_types = {}
        try:
            all_docs = self.collection.get()
            for metadata in all_docs['metadatas']:
                file_type = metadata.get('file_type', 'unknown')
                file_types[file_type] = file_types.get(file_type, 0) + 1
        except:
            pass
        
        summary = f"🤖 Local Personal Agent for {self.person_name}\n"
        summary += f"📊 Total files indexed: {file_count}\n"
        summary += f"📋 File types: {', '.join([f'{k}: {v}' for k, v in file_types.items()])}\n"
        summary += f"🧠 Using local model: {self.model_name}\n"
        summary += f"💻 Device: {self.device}\n"
        
        return summary
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


# Lightweight alternative using a smaller model
class LightweightLocalAgent(LocalPersonalFileAgent):
    """A lighter version using smaller models for lower-end hardware"""
    
    def __init__(self, folder_path: str, person_name: str):
        # Use smaller, faster models
        print("Initializing lightweight local agent...")
        self.folder_path = Path(folder_path)
        self.person_name = person_name
        
        # Initialize ChromaDB with local embeddings
        self.chroma_client = chromadb.PersistentClient(path=f"./chroma_db_{person_name}")
        self.collection_name = f"{person_name}_files"
        
        # Use a smaller embedding model
        self.embedding_function = LocalEmbeddingFunction("all-MiniLM-L6-v2")
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
        except:
            self.collection = self.chroma_client.create_collection(name=self.collection_name)
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        
        # Supported extensions
        self.supported_extensions = {'.txt', '.md', '.py', '.js', '.html', '.css', 
                                   '.json', '.xml', '.csv', '.pdf', '.docx', '.log'}
        
        # File hashes
        self.file_hashes = self.load_file_hashes()
        
        print("Lightweight agent ready!")
    
    def generate_response_simple(self, query: str, context_results: List[Dict[str, Any]]) -> str:
        """Generate a simple template-based response without heavy LLM"""
        if not context_results:
            return f"No relevant information found in {self.person_name}'s files."
        
        # Extract key information
        relevant_files = []
        key_content = []
        
        for result in context_results[:3]:
            file_name = result['metadata']['file_name']
            content = result['content'][:200].strip()
            
            relevant_files.append(file_name)
            key_content.append(f"From {file_name}: {content}")
        
        response = f"Found information in {len(relevant_files)} file(s): {', '.join(set(relevant_files))}\n\n"
        response += "Key content:\n" + '\n\n'.join(key_content)
        
        return response
    
    def ask(self, question: str) -> str:
        """Simplified ask method without heavy LLM processing"""
        print(f"🔍 Searching {self.person_name}'s files for: {question}")
        
        search_results = self.search_files(question, n_results=5)
        response = self.generate_response_simple(question, search_results)
        
        return response


# Example usage
def main():
    print("Choose your agent type:")
    print("1. Full Local Agent (requires more memory, better responses)")
    print("2. Lightweight Agent (less memory, simpler responses)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    # Get user inputs
    folder_path = input("Enter path to files folder: ").strip()
    person_name = input("Enter person name: ").strip()
    
    if choice == "2":
        # Use lightweight version
        agent = LightweightLocalAgent(folder_path, person_name)
    else:
        # Use full version with local LLM
        model_name = input("Enter model name (press Enter for default 'microsoft/DialoGPT-medium'): ").strip()
        if not model_name:
            model_name = "microsoft/DialoGPT-medium"
        agent = LocalPersonalFileAgent(folder_path, person_name, model_name)
    
    # Process files
    print("\n" + "="*50)
    agent.process_files()
    
    # Print summary
    print("\n" + agent.get_file_summary())
    
    # Interactive query loop
    print("\n" + "="*50)
    print("Agent ready! Ask questions about the files.")
    print("Commands: 'files' to list files, 'quit' to exit")
    
    try:
        while True:
            question = input(f"\n💬 Ask {agent.person_name}'s agent: ").strip()
            
            if question.lower() == 'quit':
                break
            elif question.lower() == 'files':
                files = agent.list_files()
                print(f"📁 Indexed files ({len(files)}):")
                for file in files:
                    print(f"  • {file}")
                continue
            elif not question:
                continue
                
            print("\n🤖 Agent:")
            response = agent.ask(question)
            print(response)
            
    except KeyboardInterrupt:
        print("\n\nExiting...")
    finally:
        if hasattr(agent, 'cleanup'):
            agent.cleanup()

if __name__ == "__main__":
    main()