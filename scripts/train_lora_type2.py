import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# 1. Конфигурация
MODEL_ID = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"  # 4-битная версия для экономии памяти
MAX_SEQ_LENGTH = 1024  # Максимальная длина контекста (с запасом)
LORA_RANK = 16  # Ранг LoRA (чем выше, тем больше параметров обучаем, но и больше памяти)
LORA_ALPHA = 32  # Коэффициент масштабирования (обычно 2 * rank)
LORA_DROPOUT = 0.0  # Dropout для регуляризации
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

def load_model_and_tokenizer():
    """Загружает модель и токенизатор через Unsloth."""
    print("🔄 Загрузка модели через Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Автоматически определит float16/bfloat16
        load_in_4bit=True  # 4-битное квантование для экономии VRAM
    )
    
    # Применяем LoRA к модели
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Экономия памяти через Unsloth
        random_state=42
    )
    
    model.print_trainable_parameters()
    print("✅ Модель успешно загружена и подготовлена к обучению!")
    return model, tokenizer

def prepare_dataset(tokenizer, data_path):
    """Загружает и форматирует датасет для SFTTrainer."""
    print(f"🔄 Загрузка данных из {data_path}...")
    
    # Загружаем наш подготовленный JSONL
    dataset = load_dataset("json", data_files=data_path, split="train")
    
    # Применяем chat template к каждому примеру
    def apply_chat_template(examples):
        texts = tokenizer.apply_chat_template(
            examples["messages"],
            tokenize=False,
            add_generation_prompt=False  # Мы обучаем, поэтому добавляем и ответ ассистента
        )
        return {"text": texts}
    
    dataset = dataset.map(apply_chat_template, batched=True)
    print(f"✅ Загружено {len(dataset)} примеров для обучения.")
    return dataset

def train_lora(model, tokenizer, dataset, output_dir):
    """Запускает процесс обучения LoRA."""
    print(f"🚀 Запуск обучения. Результаты будут сохранены в {output_dir}")
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=TrainingArguments(
            per_device_train_batch_size=1,  
            gradient_accumulation_steps=8,  
            warmup_steps=5,  # Прогрев learning rate
            num_train_epochs=3,  # 3 эпохи на 25 примерах — достаточно для стилизации
            learning_rate=2e-4,  # Стандартный LR для LoRA
            fp16=not torch.cuda.is_bf16_supported(),  # Используем bf16, если GPU поддерживает
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",  # 8-битный оптимизатор для экономии памяти
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir=output_dir,
            save_strategy="epoch",  # Сохраняем после каждой эпохи
            save_total_limit=1  # Храним только последний чекпоинт
        ),
    )
    
    # Запускаем обучение
    stats = trainer.train()
    print(f"✅ Обучение завершено. Потеря (loss): {stats.training_loss:.4f}")
    
    # Сохраняем адаптер
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"💾 LoRA-адаптер сохранен в {output_dir}")

if __name__ == "__main__":
    # Загружаем модель (она загрузится быстро, так как веса уже в кэше)
    model, tokenizer = load_model_and_tokenizer()
    
    # Готовим данные для СТИЛЯ 2
    dataset_type2 = prepare_dataset(tokenizer, "data/train_type2_chatml.jsonl")
    
    # Обучаем LoRA для Стиля 2 и сохраняем в другую папку!
    train_lora(model, tokenizer, dataset_type2, "models/lora_type2")
    
    print("\n🎉 Обучение для Стиля 2 завершено!")
    print("Теперь у нас есть два адаптера: models/lora_type1 и models/lora_type2")