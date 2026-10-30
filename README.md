# LLM Style Adapter

Адаптация русскоязычной LLM под стилистику Telegram-каналов с использованием LoRA Fine-Tuning и Few-Shot In-Context Learning.

## Описание проекта

Проект реализует пайплайн адаптации модели Qwen2.5-1.5B-Instruct под два стиля:
- Тип 1: Аналитика/Финансы (по мотивам @banki_oil)
- Тип 2: Городской новостник (по мотивам @moscowach)

Реализованы и сравнены два подхода:
1. LoRA Fine-Tuning через Unsloth/PEFT (приоритетный путь)
2. Few-Shot In-Context Learning (альтернативный путь для малых датасетов)

## Структура проекта

llm_style_adapter/
├── README.md                    # Этот файл
├── RESULTS.md                   # Отчёт с результатами (включая итерацию v2)
├── AGENTS.md                    # Правила работы с проектом
├── requirements.txt             # Зафиксированные зависимости
├── .gitignore
├── data/                        # Данные
│   ├── train_type1.jsonl        # Исходные данные заказчика (стиль 1)
│   ├── train_type2.jsonl        # Исходные данные заказчика (стиль 2)
│   ├── train_type1_final.jsonl  # Итоговый датасет v2 (100 примеров)
│   ├── train_type2_final.jsonl  # Итоговый датасет v2 (100 примеров)
│   ├── *_final_chatml.jsonl     # То же в ChatML — вход для обучения
│   ├── test_inputs.jsonl        # Тестовый набор (5 новостей)
│   ├── *_chatml.jsonl           # Подготовленные исходные данные
│   ├── generation_instruction.md# Инструкция для генерации датасета
│   └── baseline_type*.txt       # Бейзлайн (Zero-Shot) для сравнения
├── models/                      # Обученные LoRA-адаптеры (в .gitignore)
│   ├── lora_type1_v2/           # Актуальные адаптеры v2
│   └── lora_type2_v2/
├── scripts/                     # Скрипты пайплайна
│   ├── prepare_data.py          # Подготовка данных
│   ├── generate_baseline.py     # Бейзлайн (Zero-Shot)
│   ├── train_lora.py            # Обучение LoRA (Стиль 1) → models/lora_type1_v2
│   ├── train_lora_type2.py      # Обучение LoRA (Стиль 2) → models/lora_type2_v2
│   ├── generate_outputs.py      # Инференс LoRA (temp 0.2, seed 42)
│   └── generate_fewshot.py      # Few-Shot инференс
├── outputs_lora_type1_v2_t02.txt    # Результаты LoRA v2, стиль 1 (текущие)
├── outputs_lora_type2_v2_t02.txt    # Результаты LoRA v2, стиль 2 (текущие)
├── outputs_lora_type1.txt       # Результаты LoRA v1, стиль 1 (для сравнения)
└── outputs_lora_type2.txt       # Результаты LoRA v1, стиль 2 (для сравнения)

## Установка окружения

### Требования
- Python 3.10+
- GPU с CUDA (минимум 6 GB VRAM для обучения, 24 GB для продакшена)
- Conda (рекомендуется)

### Шаги установки

1. Создать и активировать окружение:
   conda create -n llm_clean python=3.10 -y
   conda activate llm_clean

2. Установить зависимости:
   pip install -r requirements.txt

3. Установить Unsloth с GitHub (для свежих фиксов):
   pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

## Запуск пайплайна

1. Подготовка данных:
   python scripts/prepare_data.py

2. Генерация бейзлайна (Zero-Shot):
   python scripts/generate_baseline.py

3. Обучение LoRA-адаптеров:
   python scripts/train_lora.py
   python scripts/train_lora_type2.py

4. Инференс:
   python scripts/generate_outputs.py
   python scripts/generate_fewshot.py

## Результаты

Подробный отчёт с примерами сравнения "до/после", расчётом памяти для vLLM и анализом сложностей — в файле RESULTS.md.

## Ключевые выводы

- Для малых датасетов (<100 примеров) Few-Shot ICL надёжнее, чем LoRA
- Настройка окружения на Windows требует жёсткой фиксации версий зависимостей
- Для GPU с 6 GB VRAM обязательно использование 4-битного квантования и gradient accumulation
- Сравнение двух подходов показало важность инженерного анализа и выбора оптимального метода