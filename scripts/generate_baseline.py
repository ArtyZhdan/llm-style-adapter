import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Конфигурация
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
TEST_DATA_PATH = "data/test_inputs_chatml.jsonl"
OUTPUT_BASELINE_1 = "data/baseline_type1.txt"
OUTPUT_BASELINE_2 = "data/baseline_type2.txt"

# Системные промпты для каждого стиля (наивный, но четкий промптинг)
SYSTEM_PROMPT_TYPE1 = "Ты — аналитик финансового Telegram-канала @banki_oil. Твой стиль: сухой, ироничный, с использованием профессионального сленга, без воды. Делай акцент на цифрах и фактах. В конце добавляй тег канала."
SYSTEM_PROMPT_TYPE2 = "Ты — редактор городского новостного Telegram-канала @moscowach. Твой стиль: начинай с цепляющего заголовка с эмодзи, пиши короткими абзацами, используй простой и живой язык, выделяй главное жирным (если возможно в формате текста)."

def load_model_and_tokenizer():
    print("🔄 Загрузка модели и токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Используем float16 для экономии видеопамяти. 
    # Qwen2.5-1.5B в fp16 занимает ~3 ГБ VRAM, что комфортно даже для карт с 6-8 ГБ.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto" # Автоматически распределяет слои по доступным GPU/CPU
    )
    print("✅ Модель загружена.")
    return model, tokenizer

def generate_for_style(model, tokenizer, data, system_prompt, output_path):
    print(f"\n🚀 Генерация для стиля: {system_prompt[:50]}...")
    results = []
    
    for item in data:
        # Подменяем системный промпт на целевой для этого стиля
        item["messages"][0]["content"] = system_prompt
        
        # Применяем chat template модели Qwen
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
                max_new_tokens=256,      # Ограничиваем длину ответа
                temperature=0.7,         # Добавляем немного креативности
                top_p=0.9,               # Ядерный сэмплинг для качества
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        # Декодируем только новый сгенерированный текст
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        results.append(generated_text.strip())
        print(f"  ✅ Обработан пример {len(results)}")

    # Сохраняем результаты
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, text in enumerate(results):
            f.write(f"--- Пример {i+1} ---\n{text}\n\n")
            
    print(f"💾 Результаты сохранены в {output_path}")

if __name__ == "__main__":
    # Загрузка данных
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]
        
    print(f"✅ Загружено {len(test_data)} тестовых примеров.")
    
    # Загрузка модели
    model, tokenizer = load_model_and_tokenizer()
    
    # Генерация для обоих стилей
    generate_for_style(model, tokenizer, test_data, SYSTEM_PROMPT_TYPE1, OUTPUT_BASELINE_1)
    generate_for_style(model, tokenizer, test_data, SYSTEM_PROMPT_TYPE2, OUTPUT_BASELINE_2)
    
    print("\n🎉 Бейзлайн-генерация завершена!")