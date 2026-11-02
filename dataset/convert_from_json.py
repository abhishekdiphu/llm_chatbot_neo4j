import json
from typing import Any, Dict, List

from neo4j import GraphDatabase

real_file = "subset.json"
example_file = "example.json"

NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"
NEO4J_DATABASE = "neo4j"

JSONL_FILE = example_file

BATCH_SIZE = 1000


def normalize_author_name(author_entry: Any) -> str:
    """
    authors_parsed example:
    ["Balázs", "C.", ""]
    ["Berger", "E. L.", ""]
    """
    if not isinstance(author_entry, list):
        return str(author_entry).strip()

    last_name = author_entry[0].strip() if len(author_entry) > 0 and author_entry[0] else ""
    first_names = author_entry[1].strip() if len(author_entry) > 1 and author_entry[1] else ""
    suffix = author_entry[2].strip() if len(author_entry) > 2 and author_entry[2] else ""

    parts = [first_names, last_name, suffix]
    return " ".join(part for part in parts if part).strip()


def transform_record(record: Dict[str, Any]) -> Dict[str, Any]:
    categories = record.get("categories") or ""
    category_list = [c.strip() for c in categories.split() if c.strip()]

    authors_parsed = record.get("authors_parsed") or []
    authors = [
        normalize_author_name(author)
        for author in authors_parsed
        if normalize_author_name(author)
    ]

    versions = record.get("versions") or []

    return {
        "id": record.get("id"),
        "submitter": record.get("submitter"),
        "authors_raw": record.get("authors"),
        "title": clean_text(record.get("title")),
        "comments": clean_text(record.get("comments")),
        "journal_ref": record.get("journal-ref"),
        "doi": record.get("doi"),
        "report_no": record.get("report-no"),
        "license": record.get("license"),
        "abstract": clean_text(record.get("abstract")),
        "update_date": record.get("update_date"),
        "categories": category_list,
        "authors": authors,
        "versions": versions,
    }


def clean_text(value: Any) -> Any:
    if value is None:
        return None

    if not isinstance(value, str):
        return value

    return " ".join(value.split())


def create_constraints(driver):
    constraints = [
        """
        CREATE CONSTRAINT paper_id IF NOT EXISTS
        FOR (p:Paper)
        REQUIRE p.id IS UNIQUE
        """,
        """
        CREATE CONSTRAINT author_name IF NOT EXISTS
        FOR (a:Author)
        REQUIRE a.name IS UNIQUE
        """,
        """
        CREATE CONSTRAINT category_name IF NOT EXISTS
        FOR (c:Category)
        REQUIRE c.name IS UNIQUE
        """,
        """
        CREATE CONSTRAINT version_id IF NOT EXISTS
        FOR (v:Version)
        REQUIRE v.id IS UNIQUE
        """
    ]

    with driver.session(database=NEO4J_DATABASE) as session:
        for query in constraints:
            session.run(query)


def insert_batch(tx, batch: List[Dict[str, Any]]):
    query = """
    UNWIND $batch AS row

    MERGE (p:Paper {id: row.id})
    SET p.submitter = row.submitter,
        p.authors_raw = row.authors_raw,
        p.title = row.title,
        p.comments = row.comments,
        p.journal_ref = row.journal_ref,
        p.doi = row.doi,
        p.report_no = row.report_no,
        p.license = row.license,
        p.abstract = row.abstract,
        p.update_date = row.update_date

    WITH p, row

    UNWIND row.authors AS author_name
    MERGE (a:Author {name: author_name})
    MERGE (p)-[:AUTHORED_BY]->(a)

    WITH DISTINCT p, row

    UNWIND row.categories AS category_name
    MERGE (c:Category {name: category_name})
    MERGE (p)-[:IN_CATEGORY]->(c)

    WITH DISTINCT p, row

    UNWIND row.versions AS version_data
    MERGE (v:Version {id: row.id + "_" + version_data.version})
    SET v.paper_id = row.id,
        v.version = version_data.version,
        v.created = version_data.created

    MERGE (p)-[:HAS_VERSION]->(v)
    """

    tx.run(query, batch=batch)


def read_jsonl_in_batches(path: str, batch_size: int):
    batch = []

    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                raw_record = json.loads(line)
                record = transform_record(raw_record)

                if record["id"]:
                    batch.append(record)

            except json.JSONDecodeError as error:
                print(f"Skipping invalid JSON at line {line_number}: {error}")

            if len(batch) >= batch_size:
                yield batch
                batch = []

    if batch:
        yield batch


def main():
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD)
    )

    try:
        driver.verify_connectivity()
        print("Connected to Neo4j.")

        print("Creating constraints...")
        create_constraints(driver)

        total = 0

        with driver.session(database=NEO4J_DATABASE) as session:
            for batch in read_jsonl_in_batches(JSONL_FILE, BATCH_SIZE):
                session.execute_write(insert_batch, batch)
                total += len(batch)
                print(f"Imported {total} papers...")

        print(f"Finished. Imported {total} papers.")

    finally:
        driver.close()


if __name__ == "__main__":
    main()