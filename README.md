# LangChain RAG Project

## Overview

This project leverages LangChain to implement Retrieval-Augmented Generation (RAG) for enhanced document processing and question answering. The application focuses on financial services, particularly in risk management and market analytics. The solution ingests data from various sources, processes it, and generates responses to queries by combining document retrieval and language generation.

## Features

- **Document Loading and Splitting:** Load documents from a directory and split them into manageable chunks for processing.
- **Vector Store Creation:** Generate and persist vector embeddings using Chroma.
- **Query Generation:** Create multiple versions of user queries to improve document retrieval accuracy.
- **Recursive Answer Generation:** Break down complex questions into sub-questions and recursively generate answers.
- **RAG Fusion:** Use reciprocal rank fusion to combine multiple search results for better accuracy.
- **Search and Respond:** Process user queries and generate comprehensive responses based on retrieved documents.

## Installation

1. **Clone the Repository and install dependencies:**
   ```bash
   git clone https://github.com/ishwar6/langchain-rag.git
   cd langchain-rag
   pip install -r requirements.txt
```


