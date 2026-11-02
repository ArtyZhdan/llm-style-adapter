# LLM Style Adapter

Адаптация русскоязычной LLM под стилистику Telegram-каналов с помощью QLoRA (Unsloth/PEFT/TRL).

## Описание проекта

Базовая модель — **Qwen2.5-7B-Instruct** (4-битная сборка Unsloth). На одной базе
обучены два независимых LoRA-адаптера, каждый переписывает нейтральную новость
в свой стиль:

- **Стиль 1** — аналитика/финансы по мотивам @banki_oil: короткий ироничный
  разбор с колкой добивкой и тегом `@banki_oil` в конце.
- **Стиль 2** — городской новостник по мотивам @moscowach: эмодзи,
  `**жирный заголовок**`, короткие абзацы.

Стиль выбирается системным промптом; адаптеры и данные у стилей общие по составу
(160 примеров каждый), лучшие чекпоинты разные: стиль 1 — эпоха 4, стиль 2 —
эпоха 3 (подробности и метрики — в RESULTS.md).

Финальные адаптеры обучены на бесплатном GPU **Google Colab (T4)** —
ноутбук `colab/colab_train_lora.ipynb`. На локальной RTX 2060 6 ГБ 7B обучать
нельзя: там работает только инференс (с offload части слоёв на CPU).

## Структура проекта

```
llm_style_adapter/
├── README.md                     # Этот файл
├── RESULTS.md                    # Отчёт: данные, конфиг, расчёт vLLM, метрики на 25 тестовых новостях
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
│   └── generation_instruction*.md# Инструкции, по которым сгенерированы датасеты
├── models/                       # Обученные адаптеры (в .gitignore, не коммитятся)
│   ├── lora_type1_7b/            # Финальный адаптер стиля 1 (эпоха 4, 160 примеров)
│   └── lora_type2_7b/            # Финальный адаптер стиля 2 (эпоха 3, 160 примеров)
├── colab/
│   ├── colab_train_lora.ipynb    # Ноутбук обучения на T4 (загрузить на Colab и Run all)
│   └── README.md                 # Инструкция по Colab-прогонам
├── scripts/
│   ├── prepare_data.py           # Сборка ChatML из short/long/test перед обучением
│   ├── train_lora_3b.py          # Локальный 3B-пайплайн (исторический, для 6 ГБ VRAM)
│   ├── train_lora_type2_3b.py    # Локальный 3B-пайплайн (исторический)
│   └── generate_outputs.py       # Инференс финальной связки (temp 0.2, seed 42)
├── outputs_lora_type1_final_t02.txt  # Финальные ответы стиля 1 (25 тестовых новостей)
└── outputs_lora_type2_final_t02.txt  # Финальные ответы стиля 2 (25 тестовых новостей)
```

## Установка окружения

### Требования

- Python 3.10+
- GPU с CUDA: минимум ~6 GB VRAM для инференса (с offload) и локального 3B-обучения;
  обучение финальных 7B-адаптеров идёт на Google Colab (T4, 16 ГБ)
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

1. **Обучение финальных адаптеров (7B)** — на Colab:
   загрузить `colab/colab_train_lora.ipynb` на https://colab.research.google.com,
   выбрать среду T4 GPU, Run all (подробности — `colab/README.md`).
   ChatML внутри ноутбука собирается сам; локальный `prepare_data.py` нужен
   для исторических 3B-скриптов.

2. **Инференс** на расширенном тесте (25 новостей, оба стиля, финальные 7B-адаптеры):
   ```bash
   PYTHONIOENCODING=utf-8 python scripts/generate_outputs.py
   ```
   На 6 ГБ 7B грузится с `offload_layers="auto"` — прогон занимает ~20 минут.
   Через переменные окружения можно подменить адаптеры/выходы и выбрать один
   стиль: `STYLES=type1 LORA_TYPE1=<путь> OUTPUT_TYPE1=<файл>`.

3. **Локальные 3B-эксперименты** (исторический пайплайн, ~9 минут на стиль):
   ```bash
   UNSLOTH_CE_LOSS_TARGET_GB=0.3 python scripts/train_lora_3b.py
   UNSLOTH_CE_LOSS_TARGET_GB=0.3 python scripts/train_lora_type2_3b.py
   ```

## Результаты

Полный отчёт — в RESULTS.md: эволюция данных, конфигурация QLoRA, расчёт
памяти для vLLM и итоговые замеры на 25 тестовых новостях. Финальная связка:
стиль 1 — `models/lora_type1_7b` (**24/25** чистых ответов), стиль 2 —
`models/lora_type2_7b` (**~25/25**); предыдущая 3B-связка давала 16/25 и 18/25.
