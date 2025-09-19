#!/usr/bin/env python3
"""
Clear the vector database completely and show status
This will force the system to rebuild embeddings when documents are re-added
"""

import sys
import logging
from pathlib import Path
import json
import chromadb
from chromadb.utils import embedding_functions
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clear_vector_database():
    """Clear the vector database completely"""
    try:
        # Initialize ChromaDB client
        vector_db_path = Path("./data/chroma_db")
        logger.info(f"Connecting to vector database at: {vector_db_path}")
        
        chroma_client = chromadb.PersistentClient(path=str(vector_db_path))
        
        # Check current model configuration
        hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN')
        if hf_token:
            model_name = "google/embeddinggemma-300m"
            logger.info(f"HF token found - would use: {model_name}")
        else:
            model_name = "all-MiniLM-L6-v2"
            logger.info(f"No HF token - using: {model_name}")
        
        # Get the collection
        try:
            collection = chroma_client.get_collection(name="documents")
            count_before = collection.count()
            logger.info(f"Current document count: {count_before}")
            
            if count_before > 0:
                # Delete the entire collection
                chroma_client.delete_collection(name="documents")
                logger.info("✅ Deleted existing collection")
                
                # Recreate with new embedding function
                embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=model_name
                )
                new_collection = chroma_client.create_collection(
                    name="documents",
                    embedding_function=embedding_func
                )
                logger.info(f"✅ Created new collection with {model_name} embeddings")
                logger.info(f"New collection count: {new_collection.count()}")
                
                return {
                    "status": "success",
                    "model": model_name,
                    "documents_cleared": count_before,
                    "message": "Vector database cleared and ready for new embeddings"
                }
            else:
                logger.info("Vector database is already empty")
                return {
                    "status": "already_empty",
                    "model": model_name,
                    "message": "Vector database was already empty"
                }
                
        except Exception as e:
            logger.info(f"Collection doesn't exist or error accessing it: {e}")
            # Create new collection
            embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name
            )
            new_collection = chroma_client.create_collection(
                name="documents",
                embedding_function=embedding_func
            )
            logger.info(f"✅ Created new collection with {model_name} embeddings")
            
            return {
                "status": "created_new",
                "model": model_name,
                "message": "Created new vector database collection"
            }
            
    except Exception as e:
        logger.error(f"❌ Error clearing database: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    logger.info("Clearing vector database to force embedding rebuild...")
    result = clear_vector_database()
    
    logger.info(f"Result: {result}")
    
    if result.get('status') in ['success', 'created_new', 'already_empty']:
        logger.info("🎉 Vector database is now ready!")
        logger.info(f"📋 Model configured: {result.get('model', 'unknown')}")
        logger.info("💡 When you next upload documents, they will use the new embedding model")
        sys.exit(0)
    else:
        logger.error("💥 Failed to clear database")
        sys.exit(1)