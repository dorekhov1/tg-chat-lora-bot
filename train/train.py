import os

from comet_ml import Experiment
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    MistralForCausalLM,
    TrainingArguments,
)
from trl import SFTTrainer

from finetune_utils import (
    DataCollatorForLanguageModelingChatML,
    prepare_dataset,
    print_trainable_parameters,
)

hg_token = os.getenv("HG_TOKEN")


def train(
    model_name_or_path: str = "mistralai/Mistral-Nemo-Base-2407",
    data_path: str = "../.data/result_sessions.json",
    output_dir: str = "../weights/LoRA/",
    batch_size: int = 16,
    micro_batch_size: int = 8,
    num_epochs: int = 3,
    lora_r: int = 32,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    learning_rate: float = 2e-4,
    lr_scheduler_type: str = "cosine",
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.001,
    max_seq_length: int = 1024,
    comet_project: str = "telegram-bot",
):
    experiment = Experiment(project_name=comet_project, auto_param_logging=True)

    gradient_accumulation_steps = batch_size // micro_batch_size

    experiment.log_parameters(
        {
            "model_name": model_name_or_path,
            "batch_size": batch_size,
            "micro_batch_size": micro_batch_size,
            "num_epochs": num_epochs,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "learning_rate": learning_rate,
            "lr_scheduler_type": lr_scheduler_type,
            "warmup_ratio": warmup_ratio,
            "weight_decay": weight_decay,
            "max_seq_length": max_seq_length,
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=hg_token)
    data_collator = DataCollatorForLanguageModelingChatML(tokenizer=tokenizer)

    dataset = Dataset.from_dict({"session": prepare_dataset(data_path)})

    model: MistralForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        ),
        device_map={"": 0},
        token=hg_token,
    )
    model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim="paged_adamw_8bit",
        save_steps=500,
        logging_steps=10,
        logging_first_step=True,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=lr_scheduler_type,
        report_to=["comet_ml"],
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        dataset_text_field="session",
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
        data_collator=data_collator,
    )

    trainer.train()

    experiment.log_model("final_model", output_dir)
    trainer.model.save_pretrained(output_dir)

    experiment.end()


if __name__ == "__main__":
    train()
