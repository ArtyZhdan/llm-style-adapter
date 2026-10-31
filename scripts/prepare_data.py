import json
import os

# 1. Конфигурация: из каких файлов собираем ChatML и куда складываем.
# ChatML собирается ПЕРЕД обучением/оценкой — перезапускайте скрипт после
# любого изменения исходных jsonl (short/long/test).
SOURCES = {
    "train_type1": (
        ["data/train_type1_short.jsonl", "data/train_type1_long.jsonl"],
        "data/train_type1_chatml.jsonl",
    ),
    "train_type2": (
        ["data/train_type2_short.jsonl", "data/train_type2_long.jsonl"],
        "data/train_type2_chatml.jsonl",
    ),
    "test_all": (
        ["data/test_all.jsonl"],
        "data/test_all_chatml.jsonl",
    ),
}

# Системная заглушка для тестового набора (на инференсе подменяется
# целевым промптом стиля в scripts/generate_outputs.py)
TEST_SYSTEM_STUB = "Ты — профессиональный редактор. Перепиши текст ниже, сохраняя факты, но меняя стиль подачи."


def load_jsonl(file_path: str):
    """Загружает JSONL файл построчно."""
    data = []
    if not os.path.exists(file_path):
        print(f"❌ Файл не найден: {file_path}")
        return data
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if line.strip():  # Пропускаем пустые строки
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"⚠️ Ошибка парсинга строки {line_num + 1} в {file_path}: {e}")
    return data


def format_to_chatml(item: dict) -> dict:
    """
    Конвертирует запись в стандарт ChatML.
    Два формата источников:
    - train: {"instruction", "input", "output"} → system + user + assistant
    - test:  {"id", "text"}                    → system-заглушка + user
    """
    if "instruction" in item:
        messages = [
            {"role": "system", "content": item["instruction"]},
            {"role": "user", "content": item["input"]},
            {"role": "assistant", "content": item["output"]},
        ]
    else:
        messages = [
            {"role": "system", "content": TEST_SYSTEM_STUB},
            {"role": "user", "content": item["text"]},
        ]
    return {"messages": messages}


if __name__ == "__main__":
    for name, (paths, out_path) in SOURCES.items():
        print(f"\n🔄 Собираем {name}...")
        raw_data = []
        for path in paths:
            part = load_jsonl(path)
            print(f"   ➕ {path}: {len(part)} примеров")
            raw_data.extend(part)
        print(f"✅ Всего загружено {len(raw_data)} примеров.")

        if raw_data:
            chatml_data = [format_to_chatml(item) for item in raw_data]
            with open(out_path, 'w', encoding='utf-8') as f:
                for item in chatml_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"💾 Сохранено в {out_path}")
            print(f"👉 Первый пример:\n{json.dumps(chatml_data[0], ensure_ascii=False, indent=2)}\n")
