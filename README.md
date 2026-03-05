# RAG-Pipeline-for-Document-Retrieval
A RAG pipeline that uses semantic chunking, vectore retreival, and a hugging face model for embedding. It also uses Gemini API for handling queries.  I will likely re-develop this in the future as I want to make modifications and gain more experience devleoping pipelines. 

# Design Choices

I used semantic chunking because I wanted to build a more general pipeline that I could reuse in the future. I chose this method because it is generally more reliable than fixed-length chunking, while also being less computationally expensive than using a full LLM to perform chunking (I also get to code more this way). It helps avoid splitting important pieces of information in the middle of a concept or sentence, which can hurt retrieval quality. It was also a design choice since I wanted to practice implementing a slightly more advanced chunking strategy that I would likely use professionally unless another strategy is more appropriate.
The retrieval technique I used was vector retrieval with ranking parameters.

This is a bit more complex than basic retrieval, but it builds on the strengths of semantic chunking. Instead of matching exact keywords, the system compares vector embeddings to measure semantic similarity between the query and the stored chunks. The ranking parameters then order the results by similarity score and return the top matching chunks from the vector database.
For embeddings, I used a Hugging Face embedding model because it is what I was most familiar with and it is generally fast and reliable for semantic similarity tasks. It also integrates well with LlamaIndex, which made it a practical choice for building the pipeline. I will likely refine this pipeline later, since I relied heavily on notes from previous implementations while setting it up.

# Example Responses

### Prompt 1: "What is the total estimated monthly payment?"

### Prompt 2: "How much does the borrower pay for lender's title insurance?"
