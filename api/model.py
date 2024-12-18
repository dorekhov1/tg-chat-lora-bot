import os

from comet_ml import API
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

comet_workspace = os.getenv("COMET_WORKSPACE")
hg_token = os.getenv("HG_TOKEN")


class ModelService:
    def __init__(self):
        self.api = API()
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        lora_path = self.api.get_model(
            workspace=comet_workspace,
            model_name="final_model",
            project_name="telegram-bot",
        ).download()

        base_model_name = "mistralai/Mistral-Nemo-Base-2407"

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=hg_token)

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False,
            ),
            device_map="auto",
            token=hg_token,
        )

        self.model = PeftModel.from_pretrained(
            base_model,
            lora_path,
            device_map="auto",
        )

    def generate(self, prompt: str, max_length: int = 512) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
