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

import chromadb
from chromadb.config import Settings
from openai import OpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TagInfo:
    """Represents a PI System tag with metadata."""
    tag: str
    description: str
    unit: str

class TagGlossary:
    """
    In-memory semantic search over PI System tag metadata.
    
    Uses OpenAI embeddings and Chroma vector database to enable natural language
    queries to find relevant tags for manufacturing data analysis.
    """
    
    def __init__(self, glossary_file: str = "data/tag_glossary.csv"):
        """
        Initialize the tag glossary with semantic search capabilities.
        
        Args:
            glossary_file: Path to CSV file containing tag metadata
        """
        # Load environment variables
        load_dotenv()
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.openai_client = OpenAI(api_key=api_key)
        
        # Initialize Chroma in-memory database
        self.chroma_client = chromadb.Client()
        
        # Create collection for tag embeddings
        self.collection = self.chroma_client.create_collection(
            name="tag_embeddings",
            metadata={"description": "PI System tag descriptions for semantic search"}
        )
        
        # Load and process glossary
        self.tags: List[TagInfo] = []
        self.glossary_file = glossary_file
        self._load_glossary()
        self._embed_descriptions()
        
        logger.info(f"Loaded {len(self.tags)} tags into semantic search index")
    
    def _load_glossary(self) -> None:
        """Load tag glossary from CSV file."""
        logger.info(f"Loading tag glossary from {self.glossary_file}")
        
        try:
            with open(self.glossary_file, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    tag_info = TagInfo(
                        tag=row['tag'].strip(),
                        description=row['description'].strip(), 
                        unit=row['unit'].strip()
                    )
                    self.tags.append(tag_info)
                    
        except FileNotFoundError:
            raise FileNotFoundError(f"Tag glossary file not found: {self.glossary_file}")
        except Exception as e:
            raise RuntimeError(f"Error loading tag glossary: {e}")
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        Get OpenAI embedding for text using text-embedding-3-small model.
        
        Args:
            text: Text to embed
            
        Returns:
            List of float values representing the embedding vector
        """
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding for text: {e}")
            raise
    
    def _embed_descriptions(self) -> None:
        """Generate embeddings for all tag descriptions and store in Chroma."""
        logger.info("Generating embeddings for tag descriptions...")
        
        descriptions = [tag.description for tag in self.tags]
        tag_ids = [tag.tag for tag in self.tags]
        
        # Generate embeddings in batch
        embeddings = []
        for i, description in enumerate(descriptions):
            logger.debug(f"Embedding tag {i+1}/{len(descriptions)}: {tag_ids[i]}")
            embedding = self._get_embedding(description)
            embeddings.append(embedding)
        
        # Store in Chroma with metadata
        metadatas = [
            {
                "tag": tag.tag,
                "description": tag.description,
                "unit": tag.unit
            }
            for tag in self.tags
        ]
        
        self.collection.add(
            embeddings=embeddings,
            documents=descriptions,
            metadatas=metadatas,
            ids=tag_ids
        )
        
        logger.info(f"Successfully embedded {len(descriptions)} tag descriptions")
    
    def search_tags(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Search for tags semantically similar to the query.
        
        Args:
            query: Natural language query describing desired tags
            top_k: Number of top similar tags to return
            
        Returns:
            List of dictionaries containing tag information and similarity scores
        """
        logger.debug(f"Searching for tags similar to: '{query}'")
        
        try:
            # Get embedding for query
            query_embedding = self._get_embedding(query)
            
            # Search Chroma collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['metadatas', 'documents', 'distances']
            )
            
            # Format results
            search_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'tag': results['metadatas'][0][i]['tag'],
                    'description': results['metadatas'][0][i]['description'],
                    'unit': results['metadatas'][0][i]['unit'],
                    'similarity_score': 1.0 - results['distances'][0][i]  # Convert distance to similarity
                }
                search_results.append(result)
            
            logger.debug(f"Found {len(search_results)} similar tags")
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
    
    def list_all_tags(self) -> List[str]:
        """
        Get list of all available tag names.
        
        Returns:
            List of tag names in the glossary
        """
        return [tag.tag for tag in self.tags]


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