import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

from src.core.config import MODEL_CONFIG, LORA_CONFIG
from src.utils import logger


class ModelManager:
    @staticmethod
    def _get_compute_dtype():
        dtype_name = MODEL_CONFIG.compute_dtype.lower()

        if dtype_name == "bfloat16":
            return torch.bfloat16
        if dtype_name == "float16":
            return torch.float16
        if dtype_name == "float32":
            return torch.float32

        raise ValueError(f"Неподдерживаемый compute_dtype: {MODEL_CONFIG.compute_dtype}")

    @staticmethod
    def load_base_model():
        logger.info(f"Загрузка модели {MODEL_CONFIG.model_name}...")

        compute_dtype = ModelManager._get_compute_dtype()

        quantization_config = None
        if MODEL_CONFIG.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=MODEL_CONFIG.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=MODEL_CONFIG.use_double_quant,
            )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_CONFIG.model_name,
            device_map=MODEL_CONFIG.device_map,
            torch_dtype=compute_dtype,
            quantization_config=quantization_config,
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_CONFIG.model_name,
            trust_remote_code=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "right"

        # Нужно при gradient checkpointing, иначе могут быть предупреждения/ошибки.
        model.config.use_cache = False

        logger.info("✓ Базовая модель загружена")
        return model, tokenizer

    @staticmethod
    def setup_lora_model():
        model, tokenizer = ModelManager.load_base_model()

        if MODEL_CONFIG.load_in_4bit:
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=LORA_CONFIG.r,
            lora_alpha=LORA_CONFIG.lora_alpha,
            target_modules=LORA_CONFIG.target_modules,
            lora_dropout=LORA_CONFIG.lora_dropout,
            bias=LORA_CONFIG.bias,
            task_type=LORA_CONFIG.task_type,
        )

        model = get_peft_model(model, lora_config)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        logger.info(
            f"✓ LoRA: {trainable:,} / {total:,} "
            f"({100 * trainable / total:.2f}%) параметров"
        )

        return model, tokenizer, lora_config