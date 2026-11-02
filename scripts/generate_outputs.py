import json
import os
import torch
import gc
from unsloth import FastLanguageModel

# 1. Конфигурация
# Через переменные окружения можно подменить адаптеры/файлы вывода и
# ограничить набор стилей (например, STYLES=type1 — только один прогон):
#   STYLES=type1 LORA_TYPE1=models/lora_type1_7b_colab \
#   OUTPUT_TYPE1=outputs_lora_type1_7b_t02.txt python scripts/generate_outputs.py
MODEL_ID = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"  # Не используется: base берётся из adapter_config
LORA_TYPE1 = os.environ.get("LORA_TYPE1", "models/lora_type1_v3_3b")                # Финальный адаптер стиля 1 (banki_oil)
LORA_TYPE2 = os.environ.get("LORA_TYPE2", "models/lora_type2_v5_3b/checkpoint-60")  # Финальный адаптер стиля 2 (moscowach, эпоха 3)
TEST_DATA_PATH = "data/test_all_chatml.jsonl"  # Расширенный тест: 25 новостей
OUTPUT_TYPE1 = os.environ.get("OUTPUT_TYPE1", "outputs_lora_type1_final_t02.txt")
OUTPUT_TYPE2 = os.environ.get("OUTPUT_TYPE2", "outputs_lora_type2_final_t02.txt")

# Параметры генерации: температура 0.2 — лучший баланс точности и стиля.
# Проверено: 0.7 даёт мусор и иностранные токены, 0.3–0.45 возвращают
# уверенные числовые выдумки. Более горячий сэмплинг нежелателен.
GENERATION_TEMPERATURE = 0.2

# Фиксируем seed, чтобы результаты генерации можно было воспроизвести
torch.manual_seed(42)

# Стандартный ChatML-шаблон Qwen2.5 (с ролями) — тот же, что используется
# при обучении. В папках адаптеров сохранён шаблон БЕЗ ролей
# (chat_template.jinja) — если его использовать, модель на инференсе
# получает формат, отличный от обучения, и ответы ломаются.
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)

# Системные промпты: те же короткие инструкции, что в обучающих данных
# (единый формат train/inference). Шаблон ChatML с ролями принудительно
# переопределяем ниже: в папках адаптеров сохранён шаблон без ролей.
SYSTEM_PROMPT_TYPE1 = "Перепиши новостной текст в стиле Telegram-канала @banki_oil."
SYSTEM_PROMPT_TYPE2 = "Перепиши новостной текст в стиле Telegram-канала @moscowach: эмодзи, **жирный заголовок**, короткие абзацы."

def load_model_with_adapter(adapter_path):
    """Загружает базовую модель и применяет к ней указанный LoRA-адаптер."""
    print(f"🔄 Загрузка модели с адаптером: {adapter_path}...")
    # Unsloth автоматически прочитает adapter_config.json, найдет базовую модель в кэше и загрузит их вместе
    # offload_layers="auto" — страховка для моделей >3B на 6 ГБ VRAM:
    # планировщик при нехватке памяти выгрузит часть слоёв на CPU (для 3B не срабатывает)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
        offload_layers="auto"
    )
    FastLanguageModel.for_inference(model) # Включаем режим быстрого инференса
    # Переопределяем шаблон на стандартный ChatML с ролями (см. CHAT_TEMPLATE выше)
    tokenizer.chat_template = CHAT_TEMPLATE
    print("✅ Модель и адаптер загружены.")
    return model, tokenizer

def generate_for_style(model, tokenizer, data, system_prompt, output_path):
    print(f"\n🚀 Генерация текстов...")
    results = []
    
    for i, item in enumerate(data):
        # Подменяем системный промпт на целевой для этого стиля
        item["messages"][0]["content"] = system_prompt
        
        # Применяем chat template
        prompt_text = tokenizer.apply_chat_template(
            item["messages"], 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        
        # Генерация
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=GENERATION_TEMPERATURE,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        # Декодируем только новый сгенерированный текст
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        results.append(generated_text.strip())
        print(f"  ✅ Обработан пример {i+1}/{len(data)}")

    # Сохраняем результаты
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, text in enumerate(results):
            f.write(f"--- Пример {i+1} ---\n{text}\n\n")
            
    print(f"💾 Результаты сохранены в {output_path}")

if __name__ == "__main__":
    # Какие стили генерируем (по умолчанию оба; для экспериментов можно STYLES=type1)
    STYLES = os.environ.get("STYLES", "type1,type2").split(",")

    # Загрузка тестовых данных
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]
    print(f"✅ Загружено {len(test_data)} тестовых примеров.")

    if "type1" in STYLES:
        # 1. Генерация для Стиля 1
        model1, tokenizer1 = load_model_with_adapter(LORA_TYPE1)
        generate_for_style(model1, tokenizer1, test_data, SYSTEM_PROMPT_TYPE1, OUTPUT_TYPE1)

    if "type2" in STYLES:
        # Освобождаем видеопамять перед загрузкой второго адаптера (критично для 6 ГБ VRAM!)
        print("\n🧹 Очистка видеопамяти...")
        try:
            del model1, tokenizer1
        except NameError:
            pass
        gc.collect()
        torch.cuda.empty_cache()

        # 2. Генерация для Стиля 2
        model2, tokenizer2 = load_model_with_adapter(LORA_TYPE2)
        generate_for_style(model2, tokenizer2, test_data, SYSTEM_PROMPT_TYPE2, OUTPUT_TYPE2)

    print("\n🎉 Генерация результатов завершена!")
    print(f"Стиль 1: {OUTPUT_TYPE1}")
    print(f"Стиль 2: {OUTPUT_TYPE2}")