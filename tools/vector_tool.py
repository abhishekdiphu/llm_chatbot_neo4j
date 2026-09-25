import streamlit as st

from llm import vector_llm, embeddings
from graph import graph

from langchain_neo4j import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain


#The vector database part answers:
#$$ \boxed{\text{"Which text is semantically similar?"}} $$

#while the knowledge graph part answers:

#"What is this text connected to?"


class VectorTool(): 
    def __init__(self, graph=graph, embeddings=embeddings):
        
        self.graph = graph
        self.embeddings = embeddings

        self.PAPER_RETRIEVAL_QUERY = """
                                    MATCH (p:Paper)-[:HAS_CHUNK]->(node)

                                    OPTIONAL MATCH (p)-[:AUTHORED_BY]->(a:Author)
                                    WITH node, score, p, collect(DISTINCT a.name) AS authors

                                    OPTIONAL MATCH (p)-[:IN_CATEGORY]->(c:Category)
                                    WITH node, score, p, authors, collect(DISTINCT c.name) AS categories

                                    RETURN
                                        node.text AS text,
                                        score,
                                        {
                                            paper_id: p.id,
                                            title: p.title,
                                            abstract: p.abstract,
                                            doi: p.doi,
                                            journal_ref: p.journal_ref,
                                            update_date: p.update_date,
                                            chunk_id: node.id,
                                            chunk_type: node.chunk_type,
                                            authors: authors,
                                            categories: categories,
                                            source: 'https://arxiv.org/abs/' + p.id
                                        } AS metadata
                                    """


        # Create the Neo4j vector retriever from the existing Chunk vector index.
        self.neo4jvector = Neo4jVector.from_existing_index(
            self.embeddings,
            graph=self.graph,
            index_name="chunk_embedding_index",
            node_label="Chunk",
            text_node_property="text",
            embedding_node_property="embedding",
            retrieval_query=self.PAPER_RETRIEVAL_QUERY,
        )


        # Crea  te the retriever.
        self.retriever = self.neo4jvector.as_retriever(
            search_kwargs={
                "k": 5,
            }
        )


        # Prompt for scientific-paper RAG.
        self.instructions = """
        You are a scientific paper assistant.

        Use only the given context to answer the user's question.
        The context comes from embedded title/abstract chunks from a Neo4j scientific paper knowledge graph.

        When answering:
        - Prefer papers with higher relevance to the question.
        - Mention paper titles when available.
        - Mention paper IDs when available.
        - Mention authors and categories when available.
        - Do not mention internal Chunk IDs unless useful for debugging.
        - Do not invent papers, authors, categories, DOI values, or URLs.
        - If the context does not contain enough information, say that you do not know from the available context.

        Context:
        {context}
        """

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.instructions),
                ("human", "{input}"),
            ]
        )


        # Create the RAG chain.
        self.question_answer_chain = create_stuff_documents_chain(vector_llm, self.prompt)

        self.paper_retriever = create_retrieval_chain(
            self.retriever,
            self.question_answer_chain,
        )


    def get_scientific_paper_answer(self, input_text: str):
            """
            Answer a semantic question about scientific papers using Neo4j vector search.

            Example questions:
            - "What papers are relevant to quantum computing and machine learning?"
            """
            return self.paper_retriever.invoke({"input": input_text})


        # Backward-compatible alias in case another file still imports the old function name.
    def get_movie_plot(self, input_text: str):
            return self.get_scientific_paper_answer(input_text)


