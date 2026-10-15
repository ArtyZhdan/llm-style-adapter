import json
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Конфигурация
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct" # Берем обычную, не 4-битную, для лучшего качества Few-Shot (она влезет в 6 ГБ, так как мы не обучаем)
TRAIN_DATA_PATH = "data/train_type1_chatml.jsonl" # Шпаргалки
TEST_DATA_PATH = "data/test_inputs_chatml.jsonl"
OUTPUT_FEWSHOT = "outputs_fewshot_type1.txt"
NUM_SHOTS = 3 # Количество примеров-шпаргалок, которые мы сунем в промпт

SYSTEM_PROMPT = "Ты — аналитик финансового Telegram-канала @banki_oil. Твой стиль: сухой, ироничный, с использованием профессионального сленга, без воды. Делай акцент на цифрах и фактах. В конце добавляй тег канала."

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def build_few_shot_prompt(train_examples, test_item, tokenizer):
    """Склеивает системный промпт, N примеров из тренировки и целевой запрос."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # 1. Добавляем случайные примеры из train (шпаргалки)
    shots = random.sample(train_examples, NUM_SHOTS)
    for shot in shots:
        # В train файлах у нас уже есть user и assistant
        messages.append({"role": "user", "content": shot["messages"][1]["content"]})
        messages.append({"role": "assistant", "content": shot["messages"][2]["content"]})
        
    # 2. Добавляем целевой тестовый запрос
    messages.append({"role": "user", "content": test_item["messages"][1]["content"]})
    
    # Применяем шаблон Qwen
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def main():
    print("🔄 Загрузка базовой модели для Few-Shot...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    train_data = load_jsonl(TRAIN_DATA_PATH)
    test_data = load_jsonl(TEST_DATA_PATH)
    
    results = []
    print(f"🚀 Запуск Few-Shot генерации ({NUM_SHOTS} примеров в контексте)...")
    
    for i, test_item in enumerate(test_data):
        prompt = build_few_shot_prompt(train_data, test_item, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=256, temperature=0.7, top_p=0.9, do_sample=True
            )
            
        text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        results.append(text.strip())
        print(f"  ✅ Пример {i+1} готов")
        
    with open(OUTPUT_FEWSHOT, 'w', encoding='utf-8') as f:
        for i, text in enumerate(results):
            f.write(f"--- Пример {i+1} ---\n{text}\n\n")
            
    print(f"💾 Сохранено в {OUTPUT_FEWSHOT}")

if __name__ == "__main__":
    main()