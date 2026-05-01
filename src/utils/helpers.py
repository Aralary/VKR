import json
import logging
from pathlib import Path
from typing import List, Dict


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ib_threats.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def save_jsonl(data: List[Dict], filepath: Path):
    """Сохраняет данные в JSONL формате."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"Сохранено {len(data)} записей в {filepath}")


def print_section(title: str, width: int = 80, char: str = '='):
    """Выводит форматированный заголовок раздела."""
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}\n")


def print_progress(current: int, total: int, prefix: str = "Прогресс"):
    """Выводит прогресс-бар."""
    percent = 100 * (current / float(total))
    filled = int(50 * current // total)
    bar = '█' * filled + '-' * (50 - filled)
    print(f'\r{prefix}: |{bar}| {percent:.1f}% ({current}/{total})', end='')
    if current == total:
        print()