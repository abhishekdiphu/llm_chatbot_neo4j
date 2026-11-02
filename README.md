# Scientific Paper Knowledge Graph Chatbot

An experimental research assistant for discovering and discussing scientific papers stored in a Neo4j knowledge graph. The application combines a Streamlit chat interface, a LangChain tool-calling agent, an MCP tool server, local language models, vector retrieval, generated Cypher, and Neo4j-backed conversation memory.

![Project architecture](architecures/architecure.svg)

## Features

- Semantic paper search over embedded `Chunk` nodes in Neo4j.
- Structured searches for papers, authors, categories, versions, DOI values, and other metadata.
- Tool selection through a ReAct-style LangChain agent.
- External arXiv search as a fallback.
- Short-term chat history stored in Neo4j.
- Long-term user-preference storage.
- Reasoning-trace and tool-call recording.
- Streamlit interface with per-user sessions.
- Local Qwen and Gemma models with Apple Metal acceleration.
- MCP-based separation between the agent and database tools.


The planner currently exposes these main tools:

| Tool | Purpose |
|---|---|
| `paper_vector_search` | Semantic topic search and paper recommendations |
| `paper_cypher_search` | Exact metadata and graph-relationship queries |
| `schema_of_databse_lookup` | Read the scientific-paper graph schema |
| `graph_statistics` | Count graph nodes and relationships |
| `arvix_paper_chat` | Search arXiv when local data is insufficient |

## Knowledge graph

The expected graph model is:

```text
(:Paper)-[:AUTHORED_BY]->(:Author)
(:Paper)-[:IN_CATEGORY]->(:Category)
(:Paper)-[:HAS_VERSION]->(:Version)
(:Paper)-[:HAS_CHUNK]->(:Chunk)
```

`Chunk` nodes contain text and vector embeddings. Vector retrieval expects an existing Neo4j vector index named:

```text
chunk_embedding_index
```

## Project structure

```text
.
├── chatbot.py                 # Streamlit application
├── agents/
│   ├── planner_agent.py       # Agent, tools, history, and preferences
│   └── prompts.py             # Agent system prompts
├── custom_llms/               # LangChain adapters for local models
├── memory/                    # Chat history and reasoning traces
├── mcp_client/                # Synchronous client for the MCP server
├── mcp_server/                # FastMCP server and Neo4j memory lifecycle
├── tools/                     # Vector, Cypher, arXiv, and memory tools
├── dataset/                   # Local ingestion scripts and ignored data
├── llm_train/                 # Model fine-tuning and preprocessing scripts
├── graph.py                   # Neo4j connection
├── llm.py                     # LLM and embedding configuration
├── Dockerfile
└── docker-compose.yaml
```

## Requirements

- Python 3.12 is recommended. The MCP subproject declares Python 3.12 or later.
- [`uv`](https://docs.astral.sh/uv/) for running the MCP server environment.
- A running Neo4j database. (not provided)
- A populated scientific-paper graph matching the schema above.
- The `chunk_embedding_index` vector index.
- An OpenAI-compatible embedding endpoint, by default at `http://localhost:8081/v1`.
- Sufficient memory to load the local Qwen 2.5 3B and Gemma 3 4B models.
- Apple silicon with Metal/MPS for the current local-model configuration.

The local models configured in `custom_llms/custom_llm_qwen.py` are:

```text
Qwen/Qwen2.5-3B-Instruct
neo4j/text-to-cypher-Gemma-3-4B-Instruct-2025.04.0
```

The first run downloads model files from Hugging Face and may take some time.

## Installation

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/abhishekdiphu/llm_chatbot_neo4j.git
cd llm_chatbot_neo4j

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The application also imports Neo4j Agent Memory directly. If it is not already installed by the dependency set, install the version used during development:

```bash
pip install neo4j-agent-memory==0.5.0 sentence-transformers
```


## Configuration

### Streamlit secrets

Create `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "your-api-key"
OPENAI_MODEL = "your-chat-model"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "your-password"
NEO4J_DATABASE = "neo4j"

EMBEDDING_MODEL_NAME = "embeddinggemma-300m-GGUF:BF16"
LLAMA_EMBEDDING_BASE_URL = "http://localhost:8081/v1"
LLAMA_EMBEDDING_API_KEY = "not-needed"
```

Although the planner currently uses local models, `llm.py` still reads `OPENAI_API_KEY` and `OPENAI_MODEL` during import, so both values must be present.



### MCP server environment

Create a local `.env` file:

```dotenv
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=neo4j
MCP_SERVER_URL=http://localhost:8000/mcp
```

Both `.env` and `.streamlit/` are ignored by Git. Never commit credentials.

## Running the application

The Neo4j database must be available before starting the application. Start the embedding server, MCP server, and Streamlit interface in separate terminals.

### 1. Start the embedding server

The project uses `llama-server` to expose EmbeddingGemma through an OpenAI-compatible endpoint on port `8081`. Install `llama.cpp` with server support and download the GGUF model before running:

```bash
llama-server \
  -m ~/.cache/huggingface/hub/models--unsloth--embeddinggemma-300m-GGUF/snapshots/*/embeddinggemma-300M-BF16.gguf \
  --embedding \
  --pooling mean \
  -c 2048 \
  -b 2048 \
  -ub 2048 \
  -np 1 \
  --port 8081
```

The important options are:

| Option | Purpose |
|---|---|
| `-m` | Path to the EmbeddingGemma GGUF model |
| `--embedding` | Run the server in embedding mode |
| `--pooling mean` | Mean-pool token representations into one vector |
| `-c 2048` | Set the context length to 2,048 tokens |
| `-b 2048` | Set the logical batch size |
| `-ub 2048` | Set the physical micro-batch size |
| `-np 1` | Use one parallel request slot |
| `--port 8081` | Serve the OpenAI-compatible API on port 8081 |

The application connects to:

```text
http://localhost:8081/v1
```

Keep this terminal running while using the chatbot.

### 2. Start the MCP server

From the repository root:

```bash
source .venv/bin/activate
set -a
source .env
set +a
uv run python -m mcp_server.main
```

The client expects the server at:

```text
http://localhost:8000/mcp
```

### 3. Start Streamlit

In a second terminal:

```bash
source .venv/bin/activate
export MCP_SERVER_URL=http://localhost:8000/mcp
streamlit run chatbot.py
```

Open the displayed Streamlit URL, enter a username, and start asking questions.

Example questions:

```text
Find papers about graph neural networks for drug discovery.
Which papers were written by a specific author?
List papers in the math.AG category.
Recommend papers related to knowledge graphs.
Show the number of nodes and relationships in the database.
```

## How a request is processed

1. Streamlit sends the question to `Planner`.
2. The planner combines the recent Neo4j chat history with the new question.
3. The local Qwen agent chooses an MCP tool.
4. The MCP server runs vector retrieval, generated Cypher, graph statistics, memory operations, or arXiv search.
5. The tool result is returned to the agent and rendered in Streamlit.
6. User and assistant messages are saved to Neo4j.

## Data ingestion

The `dataset/` directory contains scripts for converting arXiv metadata and creating chunk embeddings. Large source and generated datasets are intentionally excluded from Git.

Before running these scripts, review their Neo4j credentials, local model paths, embedding dimensions, and input paths. Some values are currently development-machine specific.

Ignored dataset files include:

```text
dataset/archive.zip
dataset/arxiv-metadata-oai-snapshot.json
dataset/subset.json
```

## Docker

- not completely tested and integrated.

## Development notes


## Troubleshooting

### MCP connection failed

Confirm that the MCP server is running and that this URL is reachable:

```text
http://localhost:8000/mcp
```

### Neo4j initialization failed

Verify `NEO4J_URI`, username, password, database name, and database availability in both `.env` and `.streamlit/secrets.toml`.

### Vector index not found

Create or restore the Neo4j vector index named `chunk_embedding_index` and ensure `Chunk.embedding` uses the same model and dimensions as the query embeddings.

### Model loading fails

Check available memory, Hugging Face access, the installed PyTorch/Transformers versions, and the device settings in `custom_llms/custom_llm_qwen.py`.



## License

No license file is currently included. Add a license before distributing or accepting external contributions.
