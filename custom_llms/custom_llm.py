
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

import torch
from langchain_core.runnables import RunnableLambda

from transformers import MetalConfig



llamma = "meta-llama/Llama-3.2-1B-Instruct"
llamma_adpt = "../llm_train/outputs/02/Llama-3.2-1B-Instruct-lora-adapter"
qwen= "Qwen/Qwen3-4B"
qwen_adpt = "./llm_train/outputs/03/qwen3-4b-react-adapter"


class LocalLLAMA:
    def __init__(self, base_model=None, adapter_path= None, max_new_tokens=128):
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
            max_length=128,
        )

        # ---------------------------------------------------------
        # 3. 4-bit Metal quantization configuration
        # ---------------------------------------------------------

        quantization_config = MetalConfig(
            bits=4,
            group_size=64,
        )
        # ---------------------------------------------------------



        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            #torch_dtype=torch.float32,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map="mps",)
        self.model = PeftModel.from_pretrained(
            base,
            self.adapter_path)
        if torch.backends.mps.is_available():
            self.model.to("mps")

        elif torch.cuda.is_available():
            self.model.to("cuda")
        else: 
            self.model.to("cpu")
        self.model.eval()

        print(f"device: {self.model.device}")
        return self

    def invoke(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt").to(self.model.device)
        print(f"Text1: 1")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=False)
            print(f"Text2: 1")
        text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True)
        print(f"Text: {text}")
        final_response = text.replace(prompt, "").strip()
        return final_response



local_llama = LocalLLAMA(base_model=qwen, adapter_path=qwen_adpt).load()
print("loading completed for local llm")






def local_llm_invoke(prompt_value, model=local_llama, **kwargs ):
    if hasattr(prompt_value, "to_string"):
        prompt = prompt_value.to_string()
    else:
        prompt = str(prompt_value)
    stop = kwargs.get("stop")
    response = model.invoke(prompt)
    if stop:
        for stop_token in stop:
            if stop_token in response:
                response = response.split(stop_token)[0]
    return response


cus_llm = RunnableLambda(local_llm_invoke)
if __name__ == "__main__":
    print("starting local llm test")
    print(local_llama.invoke("where is the paper of pytorch published?" ))
    print("ending local llm test")