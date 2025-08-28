import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM,
    pipeline
)
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import json
from typing import List, Dict, Tuple, Optional
import pyttsx3
from gtts import gTTS
import io
import pygame
import threading
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentStore:
    """Handles document storage and retrieval for RAG with Mistral tokenization"""
    
    def __init__(self, 
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 mistral_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Load Mistral tokenizer for consistent text processing
        self.mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model_name)
        if self.mistral_tokenizer.pad_token is None:
            self.mistral_tokenizer.pad_token = self.mistral_tokenizer.eos_token
            
        self.documents = []
        self.document_metadata = []
        self.embeddings = None
        self.index = None
        
        logger.info(f"DocumentStore initialized with {embedding_model_name} embeddings and {mistral_model_name} tokenizer")
        
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """Add documents to the store with Mistral-aware preprocessing"""
        # Preprocess documents using Mistral tokenizer for consistency
        processed_documents = []
        for doc in documents:
            # Tokenize and decode to normalize text format
            tokens = self.mistral_tokenizer.encode(doc, max_length=2048, truncation=True)
            normalized_doc = self.mistral_tokenizer.decode(tokens, skip_special_tokens=True)
            processed_documents.append(normalized_doc)
        
        self.documents.extend(processed_documents)
        
        # Handle metadata
        if metadata:
            self.document_metadata.extend(metadata)
        else:
            # Add default metadata
            default_metadata = [{"index": len(self.documents) + i, "source": "unknown"} 
                              for i in range(len(processed_documents))]
            self.document_metadata.extend(default_metadata)
        
        # Generate embeddings for processed documents
        new_embeddings = self.embedding_model.encode(processed_documents)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
            
        # Build FAISS index
        self._build_index()
        logger.info(f"Added {len(processed_documents)} documents. Total: {len(self.documents)}")
        
    def _build_index(self):
        """Build FAISS index for fast similarity search"""
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings.astype('float32'))
        
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Retrieve top-k most similar documents with metadata"""
        if self.index is None:
            return []
        
        # Preprocess query using Mistral tokenizer for consistency
        query_tokens = self.mistral_tokenizer.encode(query, max_length=512, truncation=True)
        normalized_query = self.mistral_tokenizer.decode(query_tokens, skip_special_tokens=True)
            
        query_embedding = self.embedding_model.encode([normalized_query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                metadata = self.document_metadata[idx] if idx < len(self.document_metadata) else {}
                results.append((self.documents[idx], float(score), metadata))
                
        return results
    
    def save(self, filepath: str):
        """Save document store to disk"""
        data = {
            'documents': self.documents,
            'document_metadata': self.document_metadata,
            'embeddings': self.embeddings,
            'mistral_model_name': getattr(self, 'mistral_model_name', 'mistralai/Mistral-7B-Instruct-v0.2')
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
    def load(self, filepath: str):
        """Load document store from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.documents = data['documents']
        self.document_metadata = data.get('document_metadata', [])
        self.embeddings = data['embeddings']
        
        # Reinitialize Mistral tokenizer
        mistral_model = data.get('mistral_model_name', 'mistralai/Mistral-7B-Instruct-v0.2')
        self.mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model)
        if self.mistral_tokenizer.pad_token is None:
            self.mistral_tokenizer.pad_token = self.mistral_tokenizer.eos_token
            
        self._build_index()

class RAGGenerator:
    """Handles text generation using retrieved context with Mistral/Mixtral models"""
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """
        Initialize with Mistral/Mixtral models:
        - mistralai/Mistral-7B-Instruct-v0.2 (7B parameters, fast inference)
        - mistralai/Mixtral-8x7B-Instruct-v0.1 (8x7B MoE, higher quality)
        """
        self.model_name = model_name
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with appropriate settings for large models
        if "Mixtral" in model_name:
            # Mixtral 8x7B requires more memory management
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,  # Use half precision to save memory
                device_map="auto",  # Automatically distribute across GPUs if available
                load_in_8bit=True,  # Use 8-bit quantization to reduce memory
                trust_remote_code=True
            )
        else:
            # Mistral 7B can run with standard settings
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set model to evaluation mode
        self.model.eval()
        
        logger.info(f"Loaded {model_name} successfully")
            
    def generate_response(self, query: str, context_docs: List[str], max_new_tokens: int = 256) -> str:
        """Generate response using query and retrieved context with Mistral/Mixtral"""
        
        # Prepare context from retrieved documents
        context = "\n\n".join(context_docs[:3])  # Use top 3 documents
        
        # Create Mistral-style instruction prompt with token length awareness
        prompt = f"""<s>[INST] Use the following context to answer the question. Be concise and accurate.

Context:
{context}

Question: {query} [/INST]"""
        
        # Check prompt length and truncate context if needed
        prompt_tokens = self.tokenizer.encode(prompt)
        max_context_tokens = 1800  # Leave room for response
        
        if len(prompt_tokens) > max_context_tokens:
            # Truncate context while keeping the instruction format
            context_tokens = self.tokenizer.encode(context)
            available_tokens = max_context_tokens - 200  # Reserve for instruction template
            truncated_context = self.tokenizer.decode(context_tokens[:available_tokens], skip_special_tokens=True)
            
            prompt = f"""<s>[INST] Use the following context to answer the question. Be concise and accurate.

Context:
{truncated_context}

Question: {query} [/INST]"""
        
        # Tokenize with attention to max length
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            max_length=2048,  # Mistral max context length
            truncation=True,
            padding=True
        )
        
        # Move inputs to same device as model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,  # Nucleus sampling for better quality
                top_k=50,   # Top-k sampling
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,  # Reduce repetition
                no_repeat_ngram_size=3   # Prevent 3-gram repetition
            )
            
        # Decode response
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after the prompt)
        if "[/INST]" in generated_text:
            # For Mistral format, extract after [/INST]
            response = generated_text.split("[/INST]")[-1].strip()
        else:
            # Fallback: extract after the original prompt
            response = generated_text[len(prompt):].strip()
        
        # Clean up response
        response = self._clean_response(response)
        
        return response
    
    def _clean_response(self, response: str) -> str:
        """Clean and post-process the generated response"""
        # Remove any remaining special tokens or artifacts
        response = response.replace("<s>", "").replace("</s>", "")
        
        # Remove excessive whitespace
        response = " ".join(response.split())
        
        # Truncate at natural sentence boundaries if too long
        sentences = response.split('. ')
        if len(sentences) > 1 and len(response) > 500:
            # Keep first few sentences if response is too long
            response = '. '.join(sentences[:3]) + '.'
        
        # Ensure response ends properly
        if response and not response.endswith(('.', '!', '?')):
            response += '.'
            
        return response
    
    def generate_with_conversation_history(self, query: str, context_docs: List[str], 
                                         conversation_history: List[Dict], max_new_tokens: int = 256) -> str:
        """Generate response considering conversation history"""
        
        # Prepare context
        context = "\n\n".join(context_docs[:3])
        
        # Build conversation context
        conversation_context = ""
        if conversation_history:
            recent_history = conversation_history[-3:]  # Last 3 exchanges
            for item in recent_history:
                conversation_context += f"Q: {item['query']}\nA: {item['response']}\n\n"
        
        # Create enhanced prompt with history
        prompt = f"""<s>[INST] Use the following context and conversation history to answer the question. Be concise and accurate.

Context:
{context}

Previous Conversation:
{conversation_context}

Current Question: {query} [/INST]"""
        
        # Use similar generation logic as main method
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            max_length=3072,  # Longer for conversation history
            truncation=True,
            padding=True
        )
        
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3
            )
            
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text.split("[/INST]")[-1].strip()
        response = self._clean_response(response)
        
        return response

class TTSEngine:
    """Text-to-Speech engine with multiple backends"""
    
    def __init__(self, engine_type: str = "gtts"):
        self.engine_type = engine_type
        
        if engine_type == "pyttsx3":
            self.tts_engine = pyttsx3.init()
            self._setup_pyttsx3()
        elif engine_type == "gtts":
            pygame.mixer.init()
            
    def _setup_pyttsx3(self):
        """Setup pyttsx3 engine parameters"""
        voices = self.tts_engine.getProperty('voices')
        if voices:
            self.tts_engine.setProperty('voice', voices[0].id)
        self.tts_engine.setProperty('rate', 150)
        self.tts_engine.setProperty('volume', 0.9)
        
    def speak(self, text: str, language: str = 'en') -> Optional[bytes]:
        """Convert text to speech and play/return audio"""
        if self.engine_type == "pyttsx3":
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            return None
            
        elif self.engine_type == "gtts":
            try:
                tts = gTTS(text=text, lang=language, slow=False)
                audio_buffer = io.BytesIO()
                tts.write_to_fp(audio_buffer)
                audio_buffer.seek(0)
                
                # Play audio using pygame
                pygame.mixer.music.load(audio_buffer)
                pygame.mixer.music.play()
                
                # Wait for audio to finish
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
                    
                return audio_buffer.getvalue()
                
            except Exception as e:
                logger.error(f"TTS error: {e}")
                return None
                
    def save_audio(self, text: str, filepath: str, language: str = 'en'):
        """Save speech audio to file"""
        if self.engine_type == "gtts":
            tts = gTTS(text=text, lang=language, slow=False)
            tts.save(filepath)
        else:
            logger.warning("Audio saving only supported with gTTS engine")

class RAGTTSSystem:
    """Complete RAG-based Text-to-Speech system with Mistral integration"""
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 mistral_model: str = "mistralai/Mistral-7B-Instruct-v0.2",
                 tts_engine: str = "gtts"):
        
        # Initialize all components with consistent Mistral model
        self.document_store = DocumentStore(embedding_model, mistral_model)
        self.rag_generator = RAGGenerator(mistral_model)
        self.tts_engine = TTSEngine(tts_engine)
        
        # Store model info for reference
        self.mistral_model = mistral_model
        self.embedding_model = embedding_model
        
        self.conversation_history = []
        
        logger.info(f"RAGTTSSystem initialized with:")
        logger.info(f"  - Embedding model: {embedding_model}")
        logger.info(f"  - Generation model: {mistral_model}")
        logger.info(f"  - TTS engine: {tts_engine}")
        
    def add_knowledge_base(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """Add documents to the knowledge base"""
        self.document_store.add_documents(documents, metadata)
        
    def load_knowledge_from_file(self, filepath: str):
        """Load knowledge base from text file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Split into chunks (simple approach - can be improved)
        chunks = self._split_text(content)
        self.add_knowledge_base(chunks)
        
    def _split_text(self, text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks using Mistral tokenizer for accurate length"""
        # Use Mistral tokenizer to get accurate token count
        tokenizer = self.document_store.mistral_tokenizer
        
        # Tokenize the entire text
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        chunks = []
        start_idx = 0
        
        while start_idx < len(tokens):
            # Define chunk end
            end_idx = min(start_idx + chunk_size, len(tokens))
            
            # Extract chunk tokens
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode back to text
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
            
            # Move start index with overlap
            start_idx = end_idx - overlap
            
            # Break if we're at the end
            if end_idx >= len(tokens):
                break
                
        logger.info(f"Split text into {len(chunks)} chunks using Mistral tokenizer")
        return chunks
        
    def query_with_voice(self, query: str, speak: bool = True, save_audio: Optional[str] = None) -> Dict:
        """Process query and return both text and audio response"""
        logger.info(f"Processing query: {query}")
        
        # Retrieve relevant documents
        retrieved_docs = self.document_store.retrieve(query, top_k=5)
        
        if not retrieved_docs:
            response_text = "I don't have enough information to answer your question."
        else:
            # Extract document texts (now includes metadata)
            doc_texts = [doc for doc, score, metadata in retrieved_docs]
            doc_metadata = [metadata for doc, score, metadata in retrieved_docs]
            
            # Generate response with conversation history if available
            if hasattr(self.rag_generator, 'generate_with_conversation_history'):
                response_text = self.rag_generator.generate_with_conversation_history(
                    query, doc_texts, self.conversation_history
                )
            else:
                response_text = self.rag_generator.generate_response(query, doc_texts)
            
        # Store in conversation history with enhanced metadata
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response_text,
            'retrieved_docs': len(retrieved_docs),
            'model_used': self.mistral_model,
            'doc_scores': [score for doc, score, metadata in retrieved_docs] if retrieved_docs else []
        })
        
        # Convert to speech
        audio_data = None
        if speak:
            audio_data = self.tts_engine.speak(response_text)
            
        # Save audio if requested
        if save_audio:
            self.tts_engine.save_audio(response_text, save_audio)
            
        return {
            'query': query,
            'response_text': response_text,
            'retrieved_documents': retrieved_docs,
            'audio_generated': speak,
            'audio_saved': save_audio is not None,
            'audio_data': audio_data
        }
        
    def batch_process_queries(self, queries: List[str], output_dir: str = "audio_responses"):
        """Process multiple queries and save audio responses"""
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        for i, query in enumerate(queries):
            audio_path = os.path.join(output_dir, f"response_{i+1}.mp3")
            result = self.query_with_voice(query, speak=False, save_audio=audio_path)
            results.append(result)
            
        return results
        
    def save_system_state(self, filepath: str):
        """Save the complete system state"""
        self.document_store.save(f"{filepath}_docs.pkl")
        
        # Save system configuration and history
        system_data = {
            'mistral_model': self.mistral_model,
            'embedding_model': self.embedding_model,
            'conversation_history': self.conversation_history
        }
        
        with open(f"{filepath}_system.json", 'w') as f:
            json.dump(system_data, f, indent=2)
            
    def load_system_state(self, filepath: str):
        """Load the complete system state"""
        self.document_store.load(f"{filepath}_docs.pkl")
        
        try:
            with open(f"{filepath}_system.json", 'r') as f:
                system_data = json.load(f)
            self.conversation_history = system_data.get('conversation_history', [])
            
            # Verify model compatibility
            if system_data.get('mistral_model') != self.mistral_model:
                logger.warning(f"Model mismatch: loaded {system_data.get('mistral_model')}, "
                             f"current {self.mistral_model}")
                             
        except FileNotFoundError:
            logger.warning("No system state file found")
            
    def get_system_info(self) -> Dict:
        """Get detailed system information"""
        return {
            'mistral_model': self.mistral_model,
            'embedding_model': self.embedding_model,
            'total_documents': len(self.document_store.documents),
            'total_conversations': len(self.conversation_history),
            'mistral_tokenizer_vocab_size': len(self.document_store.mistral_tokenizer.get_vocab()),
            'embedding_dimension': self.document_store.embeddings.shape[1] if self.document_store.embeddings is not None else 0
        }

# Demo and usage example
def main():
    """Demo of the RAG-TTS system with Mistral models"""
    print("Initializing RAG-TTS System with Mistral/Mixtral...")
    print("Note: First run will download large models (~13-26GB)")
    
    # Initialize system with Mistral models
    rag_tts = RAGTTSSystem(
        embedding_model="all-MiniLM-L6-v2",
        mistral_model="mistralai/Mistral-7B-Instruct-v0.2",
        tts_engine="gtts"  # Use "pyttsx3" for offline TTS
    )
    
    # Display system information
    system_info = rag_tts.get_system_info()
    print(f"System Configuration:")
    for key, value in system_info.items():
        print(f"  - {key}: {value}")
    
    # Sample knowledge base
    sample_documents = [
        "Artificial Intelligence is a branch of computer science that aims to create machines that mimic human intelligence.",
        "Machine learning is a subset of AI that enables computers to learn and make decisions from data without explicit programming.",
        "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
        "Natural language processing (NLP) is a field of AI that focuses on the interaction between computers and human language.",
        "Computer vision is an AI field that trains computers to interpret and understand visual information from the world.",
        "Robotics combines AI, engineering, and other fields to create machines that can perform tasks autonomously.",
        "The Turing Test is a measure of a machine's ability to exhibit intelligent behavior equivalent to human intelligence."
    ]
    
    # Add knowledge to the system
    print("Adding knowledge base...")
    rag_tts.add_knowledge_base(sample_documents)
    
    # Interactive query loop
    print("\nRAG-TTS System Ready!")
    print("Ask questions about AI, ML, or related topics.")
    print("Type 'quit' to exit, 'save' to save audio, 'history' to see conversation history")
    
    while True:
        query = input("\nYour question: ").strip()
        
        if query.lower() == 'quit':
            break
        elif query.lower() == 'history':
            print("\nConversation History:")
            for i, item in enumerate(rag_tts.conversation_history, 1):
                print(f"{i}. Q: {item['query']}")
                print(f"   A: {item['response'][:100]}...")
            continue
        elif query.lower().startswith('save'):
            # Save last response as audio
            if rag_tts.conversation_history:
                last_response = rag_tts.conversation_history[-1]['response']
                rag_tts.tts_engine.save_audio(last_response, "last_response.mp3")
                print("Audio saved as 'last_response.mp3'")
            continue
            
        if query:
            # Process query with voice output
            result = rag_tts.query_with_voice(query, speak=True)
            
            print(f"\nResponse: {result['response_text']}")
            print(f"Retrieved {len(result['retrieved_documents'])} relevant documents")

if __name__ == "__main__":
    # Install required packages:
    # pip install torch transformers sentence-transformers faiss-cpu pyttsx3 gtts pygame numpy accelerate bitsandbytes
    # 
    # For GPU support (recommended for Mistral/Mixtral):
    # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    #
    # Hardware Requirements:
    # - Mistral 7B: 16GB+ RAM/VRAM (8GB with quantization)
    # - Mixtral 8x7B: 32GB+ RAM/VRAM (16GB with quantization)
    main()
    