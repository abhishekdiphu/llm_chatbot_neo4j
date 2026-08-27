
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

import torch
from langchain_core.runnables import RunnableLambda



chat_gpt= True

class LocalLLAMA:
    def __init__(
        self,
        base_model="meta-llama/Llama-3.2-1B-Instruct",
        adapter_path="../llm_train/outputs/llama-3.2-1b-lora-adapter",
        max_new_tokens=128,
    ):
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self.tokenizer = None
        self.model = None
    
    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True,
            truncation=True,
            max_length=2048,
        )

        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

        self.model = PeftModel.from_pretrained(
            base,
            self.adapter_path,
        )

        if torch.backends.mps.is_available():
            self.model.to("mps")
        elif torch.cuda.is_available():
            self.model.to("cuda")

        self.model.eval()
        return self

    def invoke(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )
        
        final_response = text.replace(prompt, "").strip()
        
        return final_response



local_llama = LocalLLAMA(
    base_model="meta-llama/Llama-3.2-1B-Instruct",
    adapter_path="./llm_train/outputs/llama-3.2-1b-lora-adapter",
).load()

def local_llm_invoke(prompt_value, **kwargs):
    if hasattr(prompt_value, "to_string"):
        prompt = prompt_value.to_string()
    else:
        prompt = str(prompt_value)

    stop = kwargs.get("stop")

    response = local_llama.invoke(prompt)

    if stop:
        for stop_token in stop:
            if stop_token in response:
                response = response.split(stop_token)[0]

    return response


llm = RunnableLambda(local_llm_invoke)

if __name__ == "__main__":

    print(local_llama.invoke("The cat sat on the"))