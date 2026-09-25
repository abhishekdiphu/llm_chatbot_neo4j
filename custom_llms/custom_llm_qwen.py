
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from transformers import MetalConfig



from custom_llms.custom_rag_llm import QwenRagChatModel
from custom_llms.custom_memory_llm import QwenMemoryProvider
from custom_llms.custom_react_llm import QwenReActChatModel



class ModelConfig:
    qwen_react_base = "Qwen/Qwen2.5-3B-Instruct"
    qwen_react_adapter="Cheng-1/agenttune-qwen2.5-3b-react-lora"




class QwenModel:
    def __init__(self, base_model=None, adapter_path= None, max_new_tokens=512):
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
                max_length=4096,
            )
            self.template = self.tokenizer.chat_template
    
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
                #trust_remote_code=True,
                quantization_config=quantization_config,
                device_map="mps",)
    
    
    
            if self.adapter_path:
                print("adapter loaded ")
                self.model = PeftModel.from_pretrained(
                    base,
                    self.adapter_path)
            else:
                print("base model used  ")
                self.model = base
    
            if torch.backends.mps.is_available():
                self.model.to("mps")
    
            elif torch.cuda.is_available():
                self.model.to("cuda")
            else: 
                self.model.to("cpu")
            self.model.eval()
    
            print(f"device: {self.model.device}")
            return self

    def invoke(self, messages:list[dict[str, str]], max_new_tokens: int | None = None)-> str:

        generation_limit = max_new_tokens or self.max_new_tokens
    
        inputs = self.tokenizer.apply_chat_template(
                                                    messages,
                                                    tokenize=True,
                                                    add_generation_prompt=True,
                                                    return_tensors="pt",
                                                    return_dict=True).to(self.model.device)
        input_length = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = self.model.generate(**inputs,
                                        max_new_tokens=generation_limit,
                                        do_sample=False,
                                        pad_token_id=self.tokenizer.eos_token_id,
                                        eos_token_id=self.tokenizer.eos_token_id)

        decoded_output = self.tokenizer.decode(outputs[0][input_length:],
                                               skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)

        
        decoded_output_post_proc = decoded_output.strip()
        return decoded_output_post_proc


model_config = ModelConfig()
local_qwen   = QwenModel(base_model=model_config.qwen_react_base, 
                         adapter_path=None).load()

print("loading completed for local llm", model_config.qwen_react_base)

cus_llm    = QwenReActChatModel(qwen=local_qwen)
vector_llm = QwenRagChatModel(qwen=local_qwen)
cypher_llm = QwenRagChatModel(qwen=local_qwen)
memory_llm = QwenMemoryProvider(qwen=local_qwen)