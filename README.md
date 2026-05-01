Конечно! Вот готовый `README.md`, который можно сразу скопировать в проект:

# Questions Generator Core

Проект реализует генератор текстов, имитирующих распространённые информационные угрозы: фишинговые письма, мошеннические SMS, сообщения в мессенджерах, фейковые технические уведомления и дезинформационные новости.

Генератор основан на модели `mistralai/Mistral-7B-Instruct-v0.3`, дообученной методом QLoRA на специализированном датасете.

---

## Возможности

- Подготовка обучающего датасета из текстовых файлов.
- Валидация JSONL-датасета.
- Дообучение LoRA-адаптера.
- Генерация текстов по категориям угроз.
- Сохранение результатов генерации в `output/generated_threats`.

---

## Структура проекта

```

Questions_Generator_Core
├── data
│   ├── adapters
│   │   └── ib_threats
│   ├── datasets
│   │   ├── ib_threats_train.jsonl
│   │   ├── ib_threats_val.jsonl
│   │   └── ib_threats_test.jsonl
│   └── train_data
│       └── ib_threats
├── output
│   └── generated_threats
├── src
│   ├── core
│   │   └── config.py
│   ├── data
│   │   ├── data_preparation.py
│   │   └── validate_dataset.py
│   ├── generation
│   │   └── generator.py
│   ├── model
│   │   ├── model_setup.py
│   │   └── train_model.py
│   ├── utils
│   │   └── helpers.py
│   └── main.py
├── requirements.txt
└── README.md

````

---

## Требования

- Python 3.11
- NVIDIA GPU с поддержкой CUDA
- CUDA Toolkit, совместимый с PyTorch
- Доступ к модели `mistralai/Mistral-7B-Instruct-v0.3` на Hugging Face
- Hugging Face CLI для авторизации (`huggingface-cli login`)

---

## Установка окружения

Создание виртуального окружения и установка зависимостей:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\activate
python.exe -m pip install --upgrade pip setuptools wheel
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install -r requirements.txt
````

Проверка CUDA:

```powershell
python -c "import torch; print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA not available')"
```

---

## Подготовка данных

Исходные тексты располагаются в:

```
data/train_data/ib_threats
```

Ожидаемые категории:

* `phishing_emails.txt`
* `fraud_sms.txt`
* `messenger_fraud.txt`
* `fake_tech_notifications.txt`
* `disinformation_news.txt`

Запуск подготовки датасета:

```powershell
python -m src.main --prepare
```

После этого будут созданы файлы:

```
data/datasets/ib_threats_train.jsonl
data/datasets/ib_threats_val.jsonl
data/datasets/ib_threats_test.jsonl
```

---

## Валидация датасета

```powershell
python -m src.main --validate
```

---

## Обучение адаптера

```powershell
python -m src.main --train
```

После обучения LoRA-адаптер сохраняется в:

```
data/adapters/ib_threats
```

Основные параметры обучения:

| Параметр            | Значение                             |
| ------------------- | ------------------------------------ |
| Базовая модель      | `mistralai/Mistral-7B-Instruct-v0.3` |
| Метод обучения      | QLoRA                                |
| LoRA rank           | 16                                   |
| LoRA alpha          | 32                                   |
| LoRA dropout        | 0.05                                 |
| Target modules      | `q_proj`, `k_proj`                   |
| Epochs              | 5                                    |
| Max sequence length | 2048                                 |
| Learning rate       | 2e-4                                 |
| Optimizer           | `paged_adamw_8bit`                   |

---

## Генерация текстов

```powershell
python -m src.main --generate
```

Результаты сохраняются в:

```
output/generated_threats
```

---

## Результаты обучения

Финальный запуск обучения был выполнен на 800 обучающих и 100 валидационных примерах.

| Эпоха | Eval loss |
| ----- | --------- |
| 1     | 0.901     |
| 2     | 0.390     |
| 3     | 0.251     |
| 4     | 0.218     |
| 5     | 0.211     |

Финальное значение `eval_loss` составило `0.211`.

---

## Основные команды

```powershell
python -m src.main --prepare
python -m src.main --validate
python -m src.main --train
python -m src.main --generate
```

---

## Примечание

Проект предназначен для исследовательских и образовательных целей. Сгенерированные тексты используются для моделирования информационных угроз и дальнейшего анализа методов их обнаружения.
