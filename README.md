# LLM Style Adapter

Адаптация русскоязычной LLM под стилистику Telegram-каналов с помощью QLoRA (Unsloth/PEFT/TRL).

## Описание проекта

Базовая модель — **Qwen2.5-3B-Instruct** (4-битная сборка Unsloth). На одной базе
обучены два независимых LoRA-адаптера, каждый переписывает нейтральную новость
в свой стиль:

- **Стиль 1** — аналитика/финансы по мотивам @banki_oil: короткий ироничный
  разбор с колкой добивкой и тегом `@banki_oil` в конце.
- **Стиль 2** — городской новостник по мотивам @moscowach: эмодзи,
  `**жирный заголовок**`, короткие абзацы.

Стиль выбирается системным промптом; адаптеры и датасеты у стилей разные
(подробности и результаты — в RESULTS.md).

## Структура проекта

```
llm_style_adapter/
├── README.md                     # Этот файл
├── RESULTS.md                    # Отчёт: данные, конфиг, замеры на 25 тестовых новостях
├── AGENTS.md                     # Правила работы с проектом
├── requirements.txt              # Зафиксированные зависимости окружения
├── .gitignore
├── data/
│   ├── train_type1.jsonl         # Исходные данные заказчика (стиль 1, 25 примеров)
│   ├── train_type2.jsonl         # Исходные данные заказчика (стиль 2, 25 примеров)
│   ├── test_inputs.jsonl         # Исходный тест заказчика (5 новостей)
│   ├── train_type1_short.jsonl   # Датасет стиля 1: короткие темы, 150 примеров
│   ├── train_type2_short.jsonl   # Датасет стиля 2: короткие темы, 150 примеров
│   ├── train_type1_long.jsonl    # Датасет стиля 1: длинные новости, 10 примеров
│   ├── train_type2_long.jsonl    # Датасет стиля 2: длинные новости, 10 примеров
│   ├── train_type1_chatml.jsonl  # То же в ChatML — собирается prepare_data.py
│   ├── train_type2_chatml.jsonl  # То же в ChatML — собирается prepare_data.py
│   ├── test_all.jsonl            # Расширенный тест: 25 новостей (5 + 20 новых)
│   ├── test_all_chatml.jsonl     # Тест в ChatML — вход для generate_outputs.py
│   ├── generation_instruction*.md# Инструкции, по которым сгенерированы датасеты
├── models/                       # Обученные адаптеры (в .gitignore, не коммитятся)
│   ├── lora_type1_v3_3b/         # Финальный адаптер стиля 1
│   ├── lora_type2_v5_3b/         # Финальный адаптер стиля 2 (checkpoint-60 = эпоха 3)
│   └── lora_type2_v3_3b/         # Предыдущий адаптер стиля 2 (архив сравнения)
├── scripts/
│   ├── prepare_data.py           # Сборка ChatML из short/long/test перед обучением
│   ├── train_lora_3b.py          # Обучение адаптера стиля 1 (3B, QLoRA)
│   ├── train_lora_type2_3b.py    # Обучение адаптера стиля 2 (3B, QLoRA)
│   └── generate_outputs.py       # Инференс финальной связки (temp 0.2, seed 42)
├── outputs_lora_type1_final_t02.txt  # Финальные ответы стиля 1 (25 тестовых новостей)
└── outputs_lora_type2_final_t02.txt  # Финальные ответы стиля 2 (25 тестовых новостей)
```

## Установка окружения

### Требования

- Python 3.10+
- GPU с CUDA: минимум ~6 GB VRAM для обучения (4-bit), 24 GB для vLLM-продакшена
- Windows + Git Bash (проект разрабатывался на этой связке)
- Conda (рекомендуется)

### Шаги

1. Создать и активировать окружение:
   ```bash
   conda create -n llm_clean python=3.10 -y
   conda activate llm_clean
   ```

2. Установить зависимости (версии зафиксированы, связка хрупкая — не обновлять):
   ```bash
   pip install -r requirements.txt
   ```

3. Установить Unsloth с GitHub (в requirements.txt его нет намеренно):
   ```bash
   pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
   ```

## Запуск пайплайна

Все скрипты запускаются из корня проекта. Python — из окружения `llm_clean`
(в Windows/Git Bash: `/c/Users/<user>/miniconda3/envs/llm_clean/python.exe`).

1. **Сборка ChatML** (после любого изменения jsonl-данных):
   ```bash
   python scripts/prepare_data.py
   ```

2. **Обучение адаптеров.** Внимание: скрипты пишут в папки финальных адаптеров —
   для экспериментов предварительно поменяйте `OUTPUT_DIR` в скрипте!
   На 6 GB VRAM обучение требует чанкование loss:
   ```bash
   UNSLOTH_CE_LOSS_TARGET_GB=0.3 python scripts/train_lora_3b.py
   UNSLOTH_CE_LOSS_TARGET_GB=0.3 python scripts/train_lora_type2_3b.py
   ```

3. **Инференс** на расширенном тесте (25 новостей, оба стиля):
   ```bash
   python scripts/generate_outputs.py
   ```

## Результаты

Полный отчёт — в RESULTS.md: эволюция датасета, конфигурация QLoRA, расчёт
памяти для vLLM и итоговые замеры на 25 тестовых новостях (финальная связка:
стиль 1 — адаптер `lora_type1_v3_3b`, стиль 2 — `lora_type2_v5_3b/checkpoint-60`).
