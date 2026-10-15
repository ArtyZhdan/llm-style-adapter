import json
import os

def load_jsonl(file_path: str):
    """Загружает JSONL файл построчно."""
    data = []
    if not os.path.exists(file_path):
        print(f"❌ Файл не найден: {file_path}")
        return data
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if line.strip(): # Пропускаем пустые строки
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"⚠️ Ошибка парсинга строки {line_num + 1}: {e}")
    return data

def format_to_chatml(data: list) -> list:
    """
    Конвертирует данные в стандарт ChatML.
    Автоматически адаптируется под разные ключи (input/text, instruction).
    """
    formatted = []
    for item in data:
        # 1. Определяем системный промпт (инструкцию)
        # Если её нет (как в test_inputs), ставим универсальную заглушку
        system_content = item.get("instruction", "Ты — профессиональный редактор. Перепиши текст ниже, сохраняя факты, но меняя стиль подачи.")
        
        # 2. Определяем текст пользователя
        # Ищем 'input', если его нет, ищем 'text'. Если нет ничего, берем пустую строку.
        user_content = item.get("input") or item.get("text", "")
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        # 3. Если есть целевой ответ (для train-файлов), добавляем его
        if "output" in item and item["output"]:
            messages.append({"role": "assistant", "content": item["output"]})
            
        formatted.append({"messages": messages})
    return formatted

if __name__ == "__main__":
    files = {
        "train_type1": "data/train_type1.jsonl",
        "train_type2": "data/train_type2.jsonl",
        "test_inputs": "data/test_inputs.jsonl"
    }
    
    for name, path in files.items():
        print(f"\n🔄 Обрабатываем {name}...")
        raw_data = load_jsonl(path)
        print(f"✅ Загружено {len(raw_data)} примеров.")
        
        if raw_data:
            chatml_data = format_to_chatml(raw_data)
            
            # Сохраняем в новый jsonl файл
            out_path = f"data/{name}_chatml.jsonl"
            with open(out_path, 'w', encoding='utf-8') as f:
                for item in chatml_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"💾 Сохранено в {out_path}")
            
            # Выводим первый пример для проверки
            print(f"👉 Пример (ChatML):\n{json.dumps(chatml_data[0], ensure_ascii=False, indent=2)}\n")