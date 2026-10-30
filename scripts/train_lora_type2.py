import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# 1. Конфигурация
MODEL_ID = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"  # 4-битная версия для экономии памяти
MAX_SEQ_LENGTH = 1024  # Максимальная длина контекста (с запасом)
LORA_RANK = 16  # Ранг LoRA (чем выше, тем больше параметров обучаем, но и больше памяти)
LORA_ALPHA = 32  # Коэффициент масштабирования (обычно 2 * rank)
LORA_DROPOUT = 0.0  # Dropout выключен: на малом датасете мешает, а fused-ядра Unsloth требуют 0.0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DATA_PATH = "data/train_type2_final_chatml.jsonl"  # Итоговый датасет v2 (25 fixed + 75 new)
OUTPUT_DIR = "models/lora_type2_v2"  # Не перезаписываем старый адаптер v1!

# Стандартный ChatML-шаблон Qwen2.5 (с ролями) — тот же, что на инференсе.
# Явно фиксируем, чтобы обучение и генерация гарантированно видели один формат.
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)

def load_model_and_tokenizer():
    """Загружает модель и токенизатор через Unsloth."""
    print("🔄 Загрузка модели через Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Автоматически определит float16/bfloat16
        load_in_4bit=True  # 4-битное квантование для экономии VRAM
    )
    tokenizer.chat_template = CHAT_TEMPLATE  # Единый формат train/inference

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
        processing_class=tokenizer,  # В trl 1.x вместо tokenizer
        train_dataset=dataset,
        args=SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,  # Эффективный batch size = 1 × 8 = 8
            warmup_steps=5,  # Прогрев learning rate
            num_train_epochs=4,
            learning_rate=2e-4,  # Стандартный LR для LoRA
            fp16=not torch.cuda.is_bf16_supported(),  # RTX 2060 не поддерживает bf16
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",  # 8-битный оптимизатор для экономии памяти
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir=output_dir,
            save_strategy="epoch",  # Сохраняем после каждой эпохи
            save_total_limit=2,  # Храним два последних чекпоинта
            dataset_text_field="text",  # В trl 1.x поле датасета задаётся в конфиге
            max_length=MAX_SEQ_LENGTH,  # В trl 1.x вместо max_seq_length
            report_to=[],  # Не отправлять логи ни в один трекер
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
    dataset_type2 = prepare_dataset(tokenizer, DATA_PATH)

    # Обучаем LoRA для Стиля 2 (v2) и сохраняем в другую папку!
    train_lora(model, tokenizer, dataset_type2, OUTPUT_DIR)

    print("\n🎉 Обучение для Стиля 2 (v2) завершено!")
    print("Теперь у нас два новых адаптера: models/lora_type1_v2 и models/lora_type2_v2")
