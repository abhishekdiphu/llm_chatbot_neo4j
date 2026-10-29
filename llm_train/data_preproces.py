import json
import re
import random
from pathlib import Path


class ArxivDatasetPreparer:
    def __init__(
        self,
        input_path: str,
        output_train_path: str = "data/train.jsonl",
        output_valid_path: str = "data/valid.jsonl",
        valid_ratio: float = 0.1,
        max_examples: int | None = None,
        seed: int = 42,
    ):
        self.input_path = Path(input_path)
        self.output_train_path = Path(output_train_path)
        self.output_valid_path = Path(output_valid_path)
        self.valid_ratio = valid_ratio
        self.max_examples = max_examples
        self.seed = seed

    def clean_text(self, text: str) -> str:
        if not text:
            return ""

        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def load_records(self):
        records = []

        with open(self.input_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError:
                    continue

        return records

    def record_to_training_example(self, record: dict):
        title = self.clean_text(record.get("title", ""))
        abstract = self.clean_text(record.get("abstract", ""))
        authors = self.clean_text(record.get("authors", ""))
        categories = self.clean_text(record.get("categories", ""))

        if not title or not abstract:
            return None

        instruction = f"""what is the author of this scientific paper?
                    Title: {title}"""

        response = f"""The author of this paper titled "{title}" is {authors}."""

        return {
            "instruction": instruction,
            "response": response,
        }

    def build_dataset(self):
        records = self.load_records()
        examples = []

        for record in records:
            example = self.record_to_training_example(record)

            if example:
                examples.append(example)

        random.seed(self.seed)
        random.shuffle(examples)

        if self.max_examples:
            examples = examples[: self.max_examples]

        return examples

    def split_dataset(self, examples):
        valid_size = int(len(examples) * self.valid_ratio)

        valid_examples = examples[:valid_size]
        train_examples = examples[valid_size:]

        return train_examples, valid_examples

    def save_jsonl(self, examples, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as file:
            for example in examples:
                file.write(
                    json.dumps(example, ensure_ascii=False) + "\n"
                )

    def run(self):
        examples = self.build_dataset()
        train_examples, valid_examples = self.split_dataset(examples)

        self.save_jsonl(train_examples, self.output_train_path)
        self.save_jsonl(valid_examples, self.output_valid_path)

        print("Dataset preparation finished.")
        print(f"Train examples: {len(train_examples)}")
        print(f"Valid examples: {len(valid_examples)}")
        print(f"Train file: {self.output_train_path}")
        print(f"Valid file: {self.output_valid_path}")


if __name__ == "__main__":
    preparer = ArxivDatasetPreparer(
        input_path="data/subset.json",
        output_train_path="data/train.jsonl",
        output_valid_path="data/valid.jsonl",
        valid_ratio=0.8,
        max_examples=None,
    )

    preparer.run()

    