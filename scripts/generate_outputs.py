import json
import torch
import gc
from unsloth import FastLanguageModel

# 1. Конфигурация
MODEL_ID = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
LORA_TYPE1 = "models/lora_type1_v3_3b"
LORA_TYPE2 = "models/lora_type2_v3_3b"
TEST_DATA_PATH = "data/test_inputs_chatml.jsonl"
OUTPUT_TYPE1 = "outputs_lora_type1_v3_3b_t02.txt"
OUTPUT_TYPE2 = "outputs_lora_type2_v3_3b_t02.txt"

# Параметры генерации: температура 0.2 — лучший баланс для этой модели.
# Проверено экспериментально: 0.7 даёт мусор и галлюцинации, 0.45 —
# возвращает выдуманные числа (0,6%→0,9%, «30 банкоматов»), 0.2 — точно.
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
# (единый формат train/inference — решение шага 0). Базовый few-shot и
# baseline-скрипты не трогаем: это фиксированные точки сравнения.
SYSTEM_PROMPT_TYPE1 = "Перепиши новостной текст в стиле Telegram-канала @banki_oil."
SYSTEM_PROMPT_TYPE2 = "Перепиши новостной текст в стиле Telegram-канала @moscowach: эмодзи, **жирный заголовок**, короткие абзацы."

def load_model_with_adapter(adapter_path):
    """Загружает базовую модель и применяет к ней указанный LoRA-адаптер."""
    print(f"🔄 Загрузка модели с адаптером: {adapter_path}...")
    # Unsloth автоматически прочитает adapter_config.json, найдет базовую модель в кэше и загрузит их вместе
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path, 
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True
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
    # Загрузка тестовых данных
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]
    print(f"✅ Загружено {len(test_data)} тестовых примеров.")
    
    # 1. Генерация для Стиля 1
    model1, tokenizer1 = load_model_with_adapter(LORA_TYPE1)
    generate_for_style(model1, tokenizer1, test_data, SYSTEM_PROMPT_TYPE1, OUTPUT_TYPE1)
    
    # Освобождаем видеопамять перед загрузкой второго адаптера (критично для 6 ГБ VRAM!)
    print("\n🧹 Очистка видеопамяти...")
    del model1, tokenizer1
    gc.collect()
    torch.cuda.empty_cache()
    
    # 2. Генерация для Стиля 2
    model2, tokenizer2 = load_model_with_adapter(LORA_TYPE2)
    generate_for_style(model2, tokenizer2, test_data, SYSTEM_PROMPT_TYPE2, OUTPUT_TYPE2)
    
    print("\n🎉 Генерация всех результатов завершена!")
    print("Проверь файлы outputs_lora_type1.txt и outputs_lora_type2.txt в корне проекта.")