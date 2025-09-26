#!/usr/bin/env python3
"""
CLI interface for the RAG Assistant
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List
import json

# Import from the main module
from rag_assistant import RAGAssistant, RAGConfig

def setup_cli_args():
    """Set up command line arguments."""
    parser = argparse.ArgumentParser(
        description="RAG-Powered Question Answering Assistant - CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize knowledge base from documents
  python cli_rag.py --docs document1.pdf document2.txt --groq-key YOUR_API_KEY
  
  # Load existing knowledge base and ask questions
  python cli_rag.py --load-kb ./vector_store --groq-key YOUR_API_KEY
  
  # Interactive mode with custom settings
  python cli_rag.py --docs docs/*.pdf --groq-key YOUR_KEY --interactive --chunk-size 800
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--groq-key",
        required=True,
        help="Groq API key (or set GROQ_API_KEY environment variable)"
    )
    
    # Knowledge base options (mutually exclusive)
    kb_group = parser.add_mutually_exclusive_group(required=True)
    kb_group.add_argument(
        "--docs",
        nargs="+",
        help="Path to documents to process (supports .txt, .pdf, .docx)"
    )
    kb_group.add_argument(
        "--load-kb",
        help="Path to existing vector store to load"
    )
    
    # Optional arguments
    parser.add_argument(
        "--save-kb",
        help="Path to save the vector store (default: ./vector_store)"
    )
    
    parser.add_argument(
        "--question",
        help="Single question to ask (non-interactive mode)"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode for multiple questions"
    )
    
    # RAG configuration
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Text chunk size (default: 1000)"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap size (default: 200)"
    )
    
    parser.add_argument(
        "--max-docs",
        type=int,
        default=5,
        help="Maximum documents to retrieve (default: 5)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="LLM temperature (default: 0.1)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser

def validate_files(file_paths: List[str]) -> List[str]:
    """Validate that files exist and are supported formats."""
    valid_files = []
    supported_extensions = {'.txt', '.pdf', '.docx', '.doc'}
    
    for file_path in file_paths:
        path = Path(file_path)
        
        if not path.exists():
            print(f"Warning: File not found: {file_path}")
            continue
        
        if path.suffix.lower() not in supported_extensions:
            print(f"Warning: Unsupported file type: {file_path}")
            continue
        
        valid_files.append(str(path))
    
    return valid_files

def print_welcome():
    """Print welcome message."""
    print("=" * 60)
    print("🤖 RAG-Powered Question Answering Assistant - CLI")
    print("=" * 60)
    print()

def print_sources(sources: List[dict]):
    """Print source information."""
    print("\n📚 Sources:")
    print("-" * 40)
    for i, source in enumerate(sources, 1):
        print(f"{i}. {source['filename']} (chunk {source['chunk_id']})")
        print(f"   Preview: {source['content_preview'][:100]}...")
        print()

def interactive_mode(rag_assistant: RAGAssistant):
    """Run the assistant in interactive mode."""
    print("\n🔍 Interactive Mode - Type 'quit' or 'exit' to stop")
    print("-" * 50)
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                print("Please enter a question or 'quit' to exit.")
                continue
            
            print("🔄 Processing your question...")
            result = rag_assistant.ask_question(question)
            
            print("\n" + "=" * 50)
            if result["error"]:
                print(f"❌ Error: {result['answer']}")
            else:
                print(f"💬 Answer: {result['answer']}")
                
                if result["sources"]:
                    print_sources(result["sources"])
            print("=" * 50)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")

def single_question_mode(rag_assistant: RAGAssistant, question: str):
    """Process a single question and exit."""
    print(f"❓ Question: {question}")
    print("🔄 Processing...")
    
    result = rag_assistant.ask_question(question)
    
    print("\n" + "=" * 60)
    if result["error"]:
        print(f"❌ Error: {result['answer']}")
        sys.exit(1)
    else:
        print(f"💬 Answer: {result['answer']}")
        
        if result["sources"]:
            print_sources(result["sources"])
    print("=" * 60)

def main():
    """Main CLI function."""
    parser = setup_cli_args()
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.INFO)
    
    print_welcome()
    
    # Get API key from args or environment
    groq_api_key = args.groq_key or os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ Error: Groq API key is required. Use --groq-key or set GROQ_API_KEY environment variable.")
        sys.exit(1)
    
    # Create RAG configuration
    config = RAGConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_retrieval_docs=args.max_docs,
        temperature=args.temperature
    )
    
    # Initialize RAG assistant
    print("🚀 Initializing RAG Assistant...")
    rag_assistant = RAGAssistant(groq_api_key, config)
    
    # Process knowledge base
    if args.docs:
        # Validate document files
        valid_files = validate_files(args.docs)
        if not valid_files:
            print("❌ Error: No valid documents found.")
            sys.exit(1)
        
        print(f"📄 Processing {len(valid_files)} documents...")
        success = rag_assistant.initialize_knowledge_base(
            valid_files, 
            args.save_kb or "./vector_store"
        )
        
        if not success:
            print("❌ Error: Failed to initialize knowledge base.")
            sys.exit(1)
        
        print("✅ Knowledge base initialized successfully!")
        
    elif args.load_kb:
        print(f"📂 Loading knowledge base from {args.load_kb}...")
        success = rag_assistant.load_knowledge_base(args.load_kb)
        
        if not success:
            print("❌ Error: Failed to load knowledge base.")
            sys.exit(1)
        
        print("✅ Knowledge base loaded successfully!")
    
    # Run question answering
    if args.question:
        single_question_mode(rag_assistant, args.question)
    elif args.interactive:
        interactive_mode(rag_assistant)
    else:
        # Default to interactive if no specific question
        interactive_mode(rag_assistant)

if __name__ == "__main__":
    main()