import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from langchain_neo4j import Neo4jGraph

llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

response = llm.embeddings.create(
        input="paper related to math",
        model="text-embedding-3-large",
        dimensions=768
    )

embedding = response.data[0].embedding

#print(f"Embedding: {embedding}")

# Connect to Neo4j
# graph = 
graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI','neo4j://127.0.0.1:7687'),
    username=os.getenv('NEO4J_USERNAME', 'neo4j'),
    password=os.getenv('NEO4J_PASSWORD','password'),
    database=os.getenv('NEO4J_DATABASE', 'neo4j')
)

# Run query
# result = 
result = graph.query("""
                        MATCH (node:Chunk)
                        SEARCH node IN (
                            VECTOR INDEX chunk_embedding_index
                            FOR $embedding
                            LIMIT 4
                        ) SCORE AS score

                        RETURN node.text, score
                        """, 
                        {"embedding": embedding})

# Display results
# for row ... 
for row in result:
    print(row['node.text'])
    print("____________________________________________________________")
    print(row['score'])