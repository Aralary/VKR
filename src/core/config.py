from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"


@dataclass
class ModelConfig:
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    compute_dtype: str = "bfloat16"
    use_double_quant: bool = True
    device_map: str = "auto"


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 8   # эффективный батч = 32
    gradient_checkpointing: bool = True
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    logging_steps: int = 10
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    bf16: bool = True
    max_grad_norm: float = 0.3
    max_seq_length: int = 2048 # было 2048
    optim: str = "paged_adamw_8bit"
    report_to: str = "none"


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.15
    num_beams: int = 1


@dataclass
class PathConfig:
    data_dir: Path = field(default_factory=lambda: DATA_DIR)
    train_data_dir: Path = field(default_factory=lambda: DATA_DIR / "train_data")
    datasets_dir: Path = field(default_factory=lambda: DATA_DIR / "datasets")
    adapter_dir: Path = field(default_factory=lambda: DATA_DIR / "adapters" / "ib_threats")
    output_dir: Path = field(default_factory=lambda: OUTPUT_DIR / "generated_threats")

    def __post_init__(self):
        for d in [self.data_dir, self.train_data_dir, self.datasets_dir,
                  self.adapter_dir, self.output_dir]:
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class CategoryConfig:
    """Конфигурация категорий угроз ИБ."""
    categories: Dict[str, Dict] = field(default_factory=lambda: {
        "phishing_emails": {
            "display_name": "фишинговые письма",
            "examples_file": "phishing_emails.txt",
            "instruction_templates": [
                "Напиши фишинговое письмо от имени {org} с темой '{topic}'",
                "Составь письмо-фишинг, имитирующее уведомление от {org}",
                "Сгенерируй фишинговое email-сообщение от имени службы безопасности {org}",
                "Напиши убедительное фишинговое письмо с просьбой подтвердить данные в {org}",
            ],
            "orgs": ["Сбербанк", "Госуслуги", "ВТБ", "Почта России",
                     "ФНС России", "МВД России", "Тинькофф Банк", "Яндекс"],
            "topics": ["верификация аккаунта", "подозрительная активность",
                       "обновление данных", "блокировка счёта",
                       "выигрыш в лотерее", "налоговый вычет"],
        },
        "fraud_sms": {
            "display_name": "мошеннические SMS",
            "examples_file": "fraud_sms.txt",
            "instruction_templates": [
                "Напиши мошенническое SMS от имени {org}",
                "Составь SMS-сообщение, имитирующее уведомление банка {org}",
                "Сгенерируй короткое мошенническое SMS с призывом перейти по ссылке",
                "Напиши SMS от имени {org} о блокировке карты",
            ],
            "orgs": ["Сбербанк", "ВТБ", "Тинькофф", "Альфа-Банк",
                     "MTC", "Билайн", "Мегафон", "Почта России"],
            "topics": ["блокировка карты", "подозрительный вход",
                       "списание средств", "выигрыш", "подтверждение операции"],
        },
        "messenger_fraud": {
            "display_name": "мошеннические сообщения в мессенджерах",
            "examples_file": "messenger_fraud.txt",
            "instruction_templates": [
                "Напиши мошенническое сообщение в мессенджере от имени {org}",
                "Составь сообщение в Telegram от имени технической поддержки {org}",
                "Сгенерируй сообщение в WhatsApp с предложением быстрого заработка",
                "Напиши сообщение в мессенджере с просьбой срочно перевести деньги",
            ],
            "orgs": ["Telegram", "WhatsApp", "ВКонтакте", "Госуслуги",
                     "администрации канала", "службы поддержки"],
            "topics": ["блокировка аккаунта", "выигрыш приза",
                       "инвестиционное предложение", "помощь другу"],
        },
        "fake_tech_notifications": {
            "display_name": "поддельные технические уведомления",
            "examples_file": "fake_tech_notifications.txt",
            "instruction_templates": [
                "Напиши поддельное техническое уведомление от имени {org}",
                "Составь фальшивое предупреждение о вирусе на компьютере пользователя",
                "Сгенерируй поддельное системное сообщение о необходимости обновить ПО {org}",
                "Напиши фиктивное уведомление о нарушении безопасности от {org}",
            ],
            "orgs": ["Microsoft", "Apple", "Google", "Антивирус Касперский",
                     "Windows Security", "службы безопасности Windows"],
            "topics": ["обнаружен вирус", "требуется обновление",
                       "нарушение безопасности", "несанкционированный доступ"],
        },
        "disinformation_news": {
            "display_name": "дезинформирующие новостные фрагменты",
            "examples_file": "disinformation_news.txt",
            "instruction_templates": [
                "Напиши дезинформирующий новостной фрагмент на тему '{topic}'",
                "Составь фейковую новость, имитирующую стиль официального СМИ, о '{topic}'",
                "Сгенерируй новостной заголовок и лид с элементами дезинформации о '{topic}'",
                "Напиши псевдоновость с ложными фактами о '{topic}'",
            ],
            "orgs": [],
            "topics": ["утечке персональных данных", "кибератаке на банки",
                       "новом вирусе-шифровальщике", "взломе госсистем",
                       "масштабной мошеннической схеме"],
        },
    })
    num_examples_per_category: int = 200   # итого ~1000 на весь датасет
    random_seed: int = 42


MODEL_CONFIG = ModelConfig()
LORA_CONFIG = LoRAConfig()
TRAINING_CONFIG = TrainingConfig()
GENERATION_CONFIG = GenerationConfig()
PATH_CONFIG = PathConfig()
CATEGORY_CONFIG = CategoryConfig()