import streamlit as st

from llm import cypher_llm
from graph import graph

from langchain_neo4j import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate

"""
                   User question
                         │
                         ▼
                       Agent
                    /           \
                   /             \
                  ▼               ▼

        Vector Search          Cypher Search

     semantic meaning       structured facts
            │                    │
            ▼                    ▼
      embeddings             LLM → Cypher
            │                    │
            ▼                    ▼
 vector index lookup         Neo4j query
            │                    │
            ▼                    ▼
     similar chunks          exact rows
"""


class CypherTool():
    def __init__(self, graph=graph):
        super().__init__()
        self.graph = graph
        self.CYPHER_GENERATION_TEMPLATE = """
          You are an expert Neo4j developer translating user questions into Cypher
          for a scientific paper knowledge graph.

          Your task:
          Convert the user's question into a valid Cypher query based only on the
          provided Neo4j schema.

          Important rules:
          - Use only labels, relationships, and properties that exist in the schema.
          - Do not invent labels, relationships, or properties.
          - Do not return entire nodes such as p, a, c, v, or ch.
          - Do not return embedding properties.
          - Never return Chunk.embedding.
          - Return explicit properties only, for example:
            p.id, p.title, p.abstract, p.update_date, p.doi, p.journal_ref,
            a.name, c.name, v.version, v.created, ch.text.
          - Use case-insensitive matching for user-provided names, titles, and keywords.
          - Add a LIMIT unless the user explicitly asks for all results.
          - Prefer LIMIT 20 for search/list questions.
          - Prefer LIMIT 10 for recommendation questions.
          - For paper recommendations, use shared categories, shared authors, or related metadata.
          - For semantic similarity search, only generate vector-search Cypher if the schema
            and database support the vector index/procedure. Otherwise use title/abstract/chunk
            keyword matching.

          Known scientific-paper graph pattern, if present in the schema:
          - (:Paper)-[:AUTHORED_BY]->(:Author)
          - (:Paper)-[:IN_CATEGORY]->(:Category)
          - (:Paper)-[:HAS_VERSION]->(:Version)
          - (:Paper)-[:HAS_CHUNK]->(:Chunk)

          Example Cypher statements:

          1. Find papers by author:
          ```
          MATCH (p:Paper)-[:AUTHORED_BY]->(a:Author)
          WHERE toLower(a.name) CONTAINS toLower("Author Name")
          RETURN
            p.id AS paper_id,
            p.title AS title,
            p.abstract AS abstract,
            p.update_date AS update_date,
            a.name AS author
          LIMIT 20
          ```

          2. Find papers in a category:
          ```
          MATCH (p:Paper)-[:IN_CATEGORY]->(c:Category)
          WHERE c.name = "cs.AI"
          RETURN
            p.id AS paper_id,
            p.title AS title,
            p.abstract AS abstract,
            p.update_date AS update_date,
            c.name AS category
          LIMIT 20
          ```

          3. Get details for a specific paper:
          ```
          MATCH (p:Paper {{id: "paper_id"}})
          OPTIONAL MATCH (p)-[:AUTHORED_BY]->(a:Author)
          OPTIONAL MATCH (p)-[:IN_CATEGORY]->(c:Category)
          OPTIONAL MATCH (p)-[:HAS_VERSION]->(v:Version)
          RETURN
            p.id AS paper_id,
            p.title AS title,
            p.abstract AS abstract,
            p.doi AS doi,
            p.journal_ref AS journal_ref,
            p.update_date AS update_date,
            collect(DISTINCT a.name) AS authors,
            collect(DISTINCT c.name) AS categories,
            collect(DISTINCT {{version: v.version, created: v.created}}) AS versions
          ```

          4. Find papers by title or abstract keyword:
          ```
          MATCH (p:Paper)
          WHERE toLower(p.title) CONTAINS toLower("keyword")
            OR toLower(p.abstract) CONTAINS toLower("keyword")
          RETURN
            p.id AS paper_id,
            p.title AS title,
            p.abstract AS abstract,
            p.update_date AS update_date
          LIMIT 20
          ```

          5. Recommend papers from the same category:
          ```
          MATCH (source:Paper {{id: "paper_id"}})-[:IN_CATEGORY]->(c:Category)<-[:IN_CATEGORY]-(recommended:Paper)
          WHERE recommended.id <> source.id
          RETURN
            recommended.id AS paper_id,
            recommended.title AS title,
            recommended.abstract AS abstract,
            recommended.update_date AS update_date,
            collect(DISTINCT c.name) AS shared_categories
          LIMIT 10
          ```

          6. Recommend papers by shared authors:
          ```
          MATCH (source:Paper {{id: "paper_id"}})-[:AUTHORED_BY]->(a:Author)<-[:AUTHORED_BY]-(recommended:Paper)
          WHERE recommended.id <> source.id
          RETURN
            recommended.id AS paper_id,
            recommended.title AS title,
            recommended.abstract AS abstract,
            recommended.update_date AS update_date,
            collect(DISTINCT a.name) AS shared_authors
          LIMIT 10
          ```

          7. Count papers by category:
          ```
          MATCH (p:Paper)-[:IN_CATEGORY]->(c:Category)
          RETURN
            c.name AS category,
            count(DISTINCT p) AS paper_count
          ORDER BY paper_count DESC
          LIMIT 20
          ```

          8. Find authors of a paper:
          ```
          MATCH (p:Paper)-[:AUTHORED_BY]->(a:Author)
          WHERE p.id = "paper_id"
          RETURN
            p.id AS paper_id,
            p.title AS title,
            collect(DISTINCT a.name) AS authors
          ```

          9. Find versions of a paper:
          ```
          MATCH (p:Paper)-[:HAS_VERSION]->(v:Version)
          WHERE p.id = "paper_id"
          RETURN
            p.id AS paper_id,
            p.title AS title,
            v.version AS version,
            v.created AS created
          ORDER BY v.created
          ```

          10. Find papers using chunk text, without returning embeddings:
          ```
          MATCH (p:Paper)-[:HAS_CHUNK]->(ch:Chunk)
          WHERE toLower(ch.text) CONTAINS toLower("keyword")
          RETURN
            p.id AS paper_id,
            p.title AS title,
            ch.text AS matching_text
          LIMIT 20
          ```

          Schema:
          {schema}

          Question:
          {question}

          Cypher Query:
          """
   



        self.cypher_prompt = PromptTemplate.from_template(self.CYPHER_GENERATION_TEMPLATE)


        # Create the Cypher QA chain.
        # This chain is intended for structured graph questions about scientific papers.
        self.cypher_qa = GraphCypherQAChain.from_llm(
            cypher_llm,
            graph=graph,
            verbose=True,
            allow_dangerous_requests=True,
            cypher_prompt= self.cypher_prompt,
        )
