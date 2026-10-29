import torch
from dataclasses import dataclass
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model

from huggingface_hub import login

login("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")

@dataclass
class TrainingConfig:
    model_name: str = "Qwen/Qwen3-4B"
    dataset_path: str = "react_classic_training_400.jsonl"
    validation_dataset_path: str = "react_classic_validation_100.jsonl"
    output_dir: str = "outputs/qwen3-4b-react-lor"
    adapter_dir: str = "outputs/qwen3-4b-react-adapter"
    max_seq_length: int = 2048
    num_train_epochs: int = 8
    learning_rate: float = 2e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    logging_steps: int = 5
    save_steps: int = 50


class ReActTrainer:

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.dataset = None
        self.trainer = None
        self.validation_dataset


    # --------------------------------------------------
    # 1. TOKENIZER
    # --------------------------------------------------

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer


    # --------------------------------------------------
    # 2. CONVERT ONE MESSAGE TO TEXT
    # --------------------------------------------------

    def message_to_text(self, message):

        role = message["role"]
        content = message.get("content") or ""

        if role == "system":

            return (
                "### System:\n"
                f"{content}\n\n"
            )

        elif role == "user":

            if content.startswith("Observation ("):

                return (
                    f"{content}\n\n"
                )

            return (
                "### User:\n"
                f"{content}\n\n"
            )

        elif role == "assistant":

            return (
                f"{content}\n\n"
            )

        raise ValueError(
            f"Unknown role: {role}"
        )


    # --------------------------------------------------
    # 3. CREATE INPUTS AND TARGETS
    # --------------------------------------------------

    def preprocess_example(self, example):

        input_ids = []
        labels = []

        for message in example["messages"]:

            role = message["role"]

            text = self.message_to_text(message)

            token_ids = self.tokenizer(
                text,
                add_special_tokens=False
            )["input_ids"]

            # ---------------------------------
            # All messages become model input
            # ---------------------------------
            input_ids.extend(token_ids)

            # ---------------------------------
            # Only assistant output is TARGET
            # ---------------------------------
            if role == "assistant":

                labels.extend(token_ids)

            else:

                labels.extend(
                    [-100] * len(token_ids)
                )

        # ---------------------------------
        # EOS
        # ---------------------------------

        eos_id = self.tokenizer.eos_token_id

        input_ids.append(eos_id)

        if example["messages"][-1]["role"] == "assistant":
            labels.append(eos_id)
        else:
            labels.append(-100)

        # ---------------------------------
        # Truncation
        # ---------------------------------

        input_ids = input_ids[
            :self.config.max_seq_length
        ]

        labels = labels[
            :self.config.max_seq_length
        ]

        attention_mask = [
            1
        ] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


    # --------------------------------------------------
    # 4. LOAD DATASET
    # --------------------------------------------------

    def load_dataset(self):

        dataset = load_dataset(
            "json",
            data_files=self.config.dataset_path
        )["train"]

        validation_dataset = load_dataset(
            "json",
            data_files=self.config.validation_dataset_path
        )["validation"]


        self.dataset = dataset.map(
            self.preprocess_example,
            remove_columns=
            dataset.column_names
        )

        self.validation_dataset = validation_dataset.map(
            self.preprocess_example,
            remove_columns=
            validation_dataset.column_names
        )


        return self.dataset , self.validation_dataset


    # --------------------------------------------------
    # 5. LOAD MODEL
    # --------------------------------------------------

    def load_model(self):

        if torch.cuda.is_available():
            dtype = torch.float16

        else:

            dtype = torch.float32


        self.model = (
            AutoModelForCausalLM
            .from_pretrained(
                self.config.model_name,
                torch_dtype=dtype
            )
        )


        # Important with gradient checkpointing / training
        self.model.config.use_cache = False


        return self.model


    # --------------------------------------------------
    # 6. LoRA
    # --------------------------------------------------

    def add_lora(self):

        lora_config = LoraConfig(

            r=8,

            lora_alpha=16,

            lora_dropout=0.05,

            bias="none",

            task_type="CAUSAL_LM",

            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ],
        )


        self.model = get_peft_model(
            self.model,
            lora_config
        )


        self.model.print_trainable_parameters()


    # --------------------------------------------------
    # 7. TRAINING ARGUMENTS
    # --------------------------------------------------

    def build_training_arguments(self):

        return TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size= self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=2,
            report_to="none",
            fp16=torch.cuda.is_available(),
            remove_unused_columns=False)


    # --------------------------------------------------
    # 8. BUILD TRAINER
    # --------------------------------------------------

    def build_trainer(self):

        training_args = (
            self.build_training_arguments()
        )


        # This pads input_ids, attention_mask
        # and labels correctly.
        #
        # label_pad_token_id = -100 means
        # padded target tokens are ignored.

        data_collator = DataCollatorForSeq2Seq(

            tokenizer=self.tokenizer,

            model=self.model,

            padding=True,

            label_pad_token_id=-100,

            return_tensors="pt",
        )


        self.trainer = Trainer(

            model=self.model,

            args=training_args,

            train_dataset=self.dataset,
            eval_dataset =self.validation_dataset,

            data_collator=data_collator,
        )


        return self.trainer


    # --------------------------------------------------
    # 9. TRAIN
    # --------------------------------------------------

    def train(self):

        self.load_tokenizer()

        self.load_dataset()

        self.load_model()

        self.add_lora()

        self.build_trainer()


        # Inspect one example before training
        self.inspect_example(0)


        self.trainer.train()


        self.model.save_pretrained(
            self.config.adapter_dir
        )


        self.tokenizer.save_pretrained(
            self.config.adapter_dir
        )


        print(
            "Training finished.\n"
            f"Adapter saved to: "
            f"{self.config.adapter_dir}"
        )


    # --------------------------------------------------
    # 10. INSPECT INPUT / TARGET
    # --------------------------------------------------

    def inspect_example(self, index=0):

        example = self.dataset[index]


        input_ids = example["input_ids"]

        labels = example["labels"]


        input_text = self.tokenizer.decode(
            input_ids,
            skip_special_tokens=False
        )


        target_ids = [

            token_id

            for token_id, label
            in zip(input_ids, labels)

            if label != -100
        ]


        target_text = self.tokenizer.decode(
            target_ids,
            skip_special_tokens=False
        )


        print("\n======================")
        print("INPUT TO MODEL")
        print("======================")

        print(input_text)


        print("\n======================")
        print("TARGET")
        print("======================")

        print(target_text)



if __name__ == "__main__":
    config = TrainingConfig()
    trainer = ReActTrainer(config)
    trainer.train()