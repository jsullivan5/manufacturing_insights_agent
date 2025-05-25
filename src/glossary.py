#!/usr/bin/env python3
"""
Manufacturing Copilot (MCP) - Tag Glossary with Semantic Search

This module provides semantic search over PI System tag metadata using OpenAI embeddings
and Chroma vector database. Enables natural language queries to be translated into
relevant PI tags for downstream analysis in the MCP CLI.

The glossary loads tag descriptions, embeds them using OpenAI's text-embedding-3-small
model, and stores vectors in-memory for fast semantic similarity search.
"""

import csv
import os
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass

# Suppress tokenizer warnings and model loading verbosity for cleaner demo output
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

import chromadb
from chromadb.config import Settings
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Suppress verbose third-party logs for cleaner demo output
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)

@dataclass
class TagInfo:
    """Represents a PI System tag with metadata."""
    tag: str
    description: str
    unit: str
    category: str

class TagGlossary:
    """
    In-memory semantic search over PI System tag metadata.
    
    Uses OpenAI embeddings and Chroma vector database to enable natural language
    queries to find relevant tags for manufacturing data analysis.
    """
    
    def __init__(self, glossary_path: str = "data/tag_glossary.csv"):
        """
        Initialize the TagGlossary with semantic search capabilities.
        
        Args:
            glossary_path: Path to the CSV file containing tag definitions
        """
        self.glossary_path = glossary_path
        self.tags = []
        self.collection = None
        
        # Load tags and initialize vector database
        self._load_tags()
        self._initialize_vector_db()
    
    def _load_tags(self) -> None:
        """Load tag definitions from CSV file."""
        try:
            df = pd.read_csv(self.glossary_path)
            
            for _, row in df.iterrows():
                tag = TagInfo(
                    tag=row['tag'],
                    description=row['description'],
                    unit=row['unit'],
                    category=row['category']
                )
                self.tags.append(tag)
            
            print(f"Loaded {len(self.tags)} manufacturing tags")
            
        except Exception as e:
            logger.error(f"Error loading tag glossary: {e}")
            raise
    
    def _initialize_vector_db(self) -> None:
        """Initialize ChromaDB with tag embeddings for semantic search."""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.Client()
            
            # Create or get collection
            collection_name = "manufacturing_tags"
            try:
                self.collection = self.client.get_collection(collection_name)
            except:
                self.collection = self.client.create_collection(collection_name)
            
            # Check if collection is empty and needs to be populated
            if self.collection.count() == 0:
                print("Building semantic search index...")
                self._populate_embeddings()
            
        except Exception as e:
            logger.error(f"Error initializing vector database: {e}")
            raise
    
    def _populate_embeddings(self) -> None:
        """Generate and store embeddings for all tags."""
        try:
            # Prepare data for embedding
            documents = []
            metadatas = []
            ids = []
            
            for i, tag in enumerate(self.tags):
                # Create searchable text combining tag name and description
                searchable_text = f"{tag.tag} {tag.description} {tag.category}"
                documents.append(searchable_text)
                
                metadatas.append({
                    "tag": tag.tag,
                    "description": tag.description,
                    "unit": tag.unit,
                    "category": tag.category
                })
                
                ids.append(f"tag_{i}")
            
            # Add to collection (ChromaDB will generate embeddings automatically)
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"✅ Semantic search ready")
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def search_tags(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Search for tags semantically similar to the query.
        
        Args:
            query: Natural language query describing desired tags
            top_k: Number of top similar tags to return
            
        Returns:
            List of dictionaries containing tag information and similarity scores
        """
        try:
            # Use ChromaDB's built-in query functionality
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=['metadatas', 'documents', 'distances']
            )
            
            # Format results
            search_results = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    result = {
                        'tag': results['metadatas'][0][i]['tag'],
                        'description': results['metadatas'][0][i]['description'],
                        'unit': results['metadatas'][0][i]['unit'],
                        'similarity_score': 1.0 - results['distances'][0][i]  # Convert distance to similarity
                    }
                    search_results.append(result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching tags: {e}")
            raise
    
    def get_tag_info(self, tag_name: str) -> Optional[TagInfo]:
        """
        Get detailed information for a specific tag.
        
        Args:
            tag_name: Exact tag name to look up
            
        Returns:
            TagInfo object if found, None otherwise
        """
        for tag in self.tags:
            if tag.tag == tag_name:
                return tag
        return None
    
    def list_all_tags(self) -> List[Dict]:
        """
        Get list of all available tags with their metadata.
        
        Returns:
            List of dictionaries containing tag information
        """
        return [
            {
                'tag': tag.tag,
                'description': tag.description,
                'unit': tag.unit,
                'category': tag.category
            }
            for tag in self.tags
        ]


def search_tags(query: str, top_k: int = 3) -> List[Dict]:
    """
    Convenience function for semantic tag search.
    
    Creates a TagGlossary instance and performs search. For production use,
    consider maintaining a single glossary instance to avoid reloading.
    
    Args:
        query: Natural language query describing desired tags
        top_k: Number of top similar tags to return
        
    Returns:
        List of dictionaries containing tag information and similarity scores
    """
    glossary = TagGlossary()
    return glossary.search_tags(query, top_k)


def main():
    """
    Demonstration of tag glossary semantic search capabilities.
    
    Shows how natural language queries can be translated into relevant
    PI System tags for manufacturing data analysis.
    """
    print("Manufacturing Copilot - Tag Glossary Demo")
    print("=" * 50)
    
    try:
        # Initialize glossary
        print("Initializing tag glossary with OpenAI embeddings...")
        glossary = TagGlossary()
        
        # Demo queries
        demo_queries = [
            "freezer temperature inside",
            "power consumption and energy usage", 
            "door open status and alarms",
            "compressor running state",
            "temperature control setpoint"
        ]
        
        print(f"\nLoaded {len(glossary.list_all_tags())} tags for semantic search")
        print("\nDemo: Natural Language Tag Search")
        print("-" * 40)
        
        for query in demo_queries:
            print(f"\nQuery: '{query}'")
            results = glossary.search_tags(query, top_k=3)
            
            for i, result in enumerate(results, 1):
                similarity_pct = result['similarity_score'] * 100
                print(f"  {i}. {result['tag']} (similarity: {similarity_pct:.1f}%)")
                print(f"     Description: {result['description']}")
                print(f"     Unit: {result['unit']}")
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("The glossary is ready for MCP CLI integration.")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"Error: {e}")
        print("Make sure OPENAI_API_KEY is set in your .env file")


if __name__ == "__main__":
    main() 