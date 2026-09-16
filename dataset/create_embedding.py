import os
import requests
from dotenv import load_dotenv
from neo4j import GraphDatabase
from llama_cpp import Llama

load_dotenv()
class EmbeddingConfig:
    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
    LLAMA_EMBEDDING_URL = os.getenv("LLAMA_EMBEDDING_URL", "http://localhost:8081/v1/embeddings")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "embeddinggemma-300m-GGUF:BF16")
    EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))

configs = EmbeddingConfig()
print(f"current batch size: {configs.BATCH_SIZE}")
driver = GraphDatabase.driver(configs.NEO4J_URI,auth=(configs.NEO4J_USERNAME, configs.NEO4J_PASSWORD),)

TOKENIZER_MODEL_PATH = "/Users/abhishekburagohain/.cache/huggingface/hub/models--unsloth--embeddinggemma-300m-GGUF/" \
"snapshots/6661a6504c30d8304af13455cb4a5d4f5bc6011f/embeddinggemma-300M-BF16.gguf"

tokenizer = Llama(model_path=TOKENIZER_MODEL_PATH, embedding=True, verbose=False)







def count_tokens(text: str) -> int:
    return len(tokenizer.tokenize(text.encode("utf-8")))

def get_embeddings(texts):

    response = requests.post(configs.LLAMA_EMBEDDING_URL,
        json={"input": texts,"model": configs.EMBEDDING_MODEL_NAME, "encoding_format": "float",},
        timeout=120)
    response.raise_for_status()
    data = response.json()
    return [item["embedding"] for item in data["data"]]


def create_constraints_and_index(tx):
    tx.run(
        """
        CREATE CONSTRAINT chunk_id IF NOT EXISTS
        FOR (c:Chunk)
        REQUIRE c.id IS UNIQUE
        """
    )

    tx.run(
        f"""
        CREATE VECTOR INDEX chunk_embedding_index IF NOT EXISTS
        FOR (c:Chunk)
        ON c.embedding
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: {configs.EMBEDDING_DIMENSIONS},
            `vector.similarity_function`: 'cosine'
          }}
        }}
        """
    )


def fetch_papers_without_chunks(tx, limit):
    result = tx.run(
        """
        MATCH (p:Paper)
        WHERE p.title IS NOT NULL
          AND p.abstract IS NOT NULL
          AND NOT (p)-[:HAS_CHUNK]->(:Chunk {chunk_type: 'title_abstract'})
        RETURN p.id AS paper_id,
               p.title AS title,
               p.abstract AS abstract
        LIMIT $limit
        """,
        limit=limit,
    )

    return [record.data() for record in result]


def write_chunks(tx, rows):
    tx.run(
        """
        UNWIND $rows AS row

        MATCH (p:Paper {id: row.paper_id})

        MERGE (c:Chunk {id: row.chunk_id})
        SET c.paper_id = row.paper_id,
            c.chunk_type = row.chunk_type,
            c.text = row.text,
            c.embedding = row.embedding,
            c.embedding_model = row.embedding_model,
            c.embedding_dimensions = row.embedding_dimensions

        MERGE (p)-[:HAS_CHUNK]->(c)
        """,
        rows=rows,
    )

def limit_text_by_tokens(text: str, max_tokens: int = 500) -> str:
    tokens = tokenizer.tokenize(text.encode("utf-8"))

    if len(tokens) <= max_tokens:
        return text

    limited_tokens = tokens[:max_tokens]
    return tokenizer.detokenize(limited_tokens).decode("utf-8", errors="ignore")


def main():
    total = 0

    with driver.session(database=configs.NEO4J_DATABASE) as session:
        session.execute_write(create_constraints_and_index)

        while True:
            papers = session.execute_read(fetch_papers_without_chunks, configs.BATCH_SIZE)

            if not papers:
                break

            texts = [
                f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
                for paper in papers 
            ]

            MAX_TOKENS = 400
            modified_texts = []
            for i, text in enumerate(texts):
                token_count = count_tokens(text)

                if token_count > MAX_TOKENS:
                    print(
                        f"Warning: Text at index {i} / paper {papers[i]['paper_id']} "
                        f"has {token_count} tokens. Truncating to {MAX_TOKENS} tokens."
                    )

                    text = limit_text_by_tokens(text, MAX_TOKENS)

                modified_texts.append(text)



                    

            vectors = get_embeddings(texts)

            rows = []

            for paper, text, vector in zip(papers, texts, vectors):
                rows.append(
                    {
                        "paper_id": paper["paper_id"],
                        "chunk_id": f"{paper['paper_id']}_title_abstract",
                        "chunk_type": "title_abstract",
                        "text": text,
                        "embedding": vector,
                        "embedding_model": configs.EMBEDDING_MODEL_NAME,
                        "embedding_dimensions": len(vector),
                    }
                )

            session.execute_write(write_chunks, rows)

            total += len(rows)
            print(f"Created and embedded {total} chunks...")

    driver.close()
    print("Done.")


if __name__ == "__main__":
    main()