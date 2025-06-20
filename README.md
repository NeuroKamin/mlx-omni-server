# MLX Omni Server [Kamin]

MLX Omni Server — это локальный сервер для инференса, работающий на библиотеке MLX от Apple. Сервер ориентирован на чипы Apple Silicon (M-серия) и реализует OpenAI-совместимые API, позволяя им интегрироваться с клиентами OpenAI и запускать модели локально.

## Возможности

- 🚀 **Оптимизация под Apple Silicon**: работает на MLX, оптимизирован для M1/M2/M3/M4
- 🔌 **Совместимость с API OpenAI**: полная замена оригинальных эндпойнтов
- 🎯 **Разнообразные возможности ИИ**:
    - Обработка аудио (TTS и STT)
    - Генерация текста
    - Создание эмбедингов
    - Генерация изображений
- ⚡️ **Высокая производительность**: локальная инференция с аппаратным ускорением
- 🔐 **Приватность**: все вычисления выполняются на вашем компьютере
- 🛠 **Поддержка SDK**: работает с официальным SDK OpenAI и другими клиентами
- ♻️ **Автоматическая очистка кеша**: ненужные загрузки удаляются для экономии места

## Поддерживаемые эндпойнты API

Сервер реализует OpenAI-совместимые эндпойнты:

- [Chat completions](https://platform.openai.com/docs/api-reference/chat): `/v1/chat/completions`
    - ✅ Чаты
    - ✅ Инструменты, вызов функций
    - ✅ Структурированный вывод
    - ✅ LogProbs
    - 🚧 Vision
- [Audio](https://platform.openai.com/docs/api-reference/audio)
    - ✅ `/v1/audio/speech` - текст в речь
    - ✅ `/v1/audio/transcriptions` - речь в текст
- [Models](https://platform.openai.com/docs/api-reference/models/list)
    - ✅ `/v1/models` - список моделей
    - ✅ `/v1/models/{model}` - получить или удалить модель
    - ✅ `/v1/models/load` - загрузка модели в фоне
    - ✅ `/v1/models/load/{id}` - статус загрузки
- [Images](https://platform.openai.com/docs/api-reference/images)
    - ✅ `/v1/images/generations` - генерация изображений
- [Embeddings](https://platform.openai.com/docs/api-reference/embeddings)
    - ✅ `/v1/embeddings` - создание эмбеддингов


### Параметры сервера

```bash
# Запуск с настройками по умолчанию (port 10240)
mlx-omni-server

# Или укажите свой порт
mlx-omni-server --port 8000

# Посмотреть все доступные опции
mlx-omni-server --help
```

### Переменные окружения

`WHISPER_CPP_MAX_WORKERS` задает число одновременных копий whisper.cpp для распознавания речи. Столько же моделей может находиться одновременно в памяти. Увеличьте значение, если ожидаете много параллельных запросов.

```bash
export WHISPER_CPP_MAX_WORKERS=2  # разрешить две копии whisper.cpp
```

### Базовая настройка клиента

```python
from openai import OpenAI

# Подключение к локальному серверу
client = OpenAI(
    base_url="http://localhost:10240/v1",  # адрес локального сервера
    api_key="not-needed"                   # ключ не требуется
)

# Простой запрос к чату
response = client.chat.completions.create(
    model="mlx-community/gemma-3-1b-it-4bit-DWQ",
    messages=[{"role": "user", "content": "Hello, how are you?"}]
)
print(response.choices[0].message.content)
```

## Расширенное использование

MLX Omni Server поддерживает несколько способов взаимодействия с моделями.

### Варианты обращения к API

#### REST API

Можно обращаться напрямую через HTTP:

```bash
# Эндпойнт для чата
curl http://localhost:10240/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/gemma-3-1b-it-4bit-DWQ",
    "messages": [{"role": "user", "content": "Hello"}]
  }'

# Список моделей
curl http://localhost:10240/v1/models
```

#### OpenAI SDK

Используйте официальный Python SDK:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:10240/v1",  # адрес локального сервера
    api_key="not-needed"                   # для локальной работы не нужен
)
```

В разделе FAQ есть информация о работе через TestClient.

### Примеры API

#### Чат

```python
response = client.chat.completions.create(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
    ],
    temperature=0,
    stream=True
)

for chunk in response:
    print(chunk)
    print(chunk.choices[0].delta.content)
    print("****************")
```

<details>
<summary>Curl Example</summary>

```shell
curl http://localhost:10240/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "stream": true,
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
    ]
  }'
```

</details>

#### Текст в речь

```python
speech_file_path = "mlx_example.wav"
response = client.audio.speech.create(
  model="lucasnewman/f5-tts-mlx",
  voice="alloy",
  input="MLX project is awsome.",
)
response.stream_to_file(speech_file_path)
```

<details>
<summary>Curl Example</summary>

```shell
curl -X POST "http://localhost:10240/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lucasnewman/f5-tts-mlx",
    "input": "MLX project is awsome",
    "voice": "alloy"
  }' \
  --output ~/Desktop/mlx.wav
```

</details>

#### Речь в текст

```python
audio_file = open("speech.mp3", "rb")
transcript = client.audio.transcriptions.create(
    model="mlx-community/whisper-large-v3-turbo",
    file=audio_file
)

print(transcript.text)
```

<details>
<summary>Curl Example</summary>

```shell
curl -X POST "http://localhost:10240/v1/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@mlx_example.wav" \
  -F "model=mlx-community/whisper-large-v3-turbo"
```

Response:

```json
{
  "text": " MLX Project is awesome!"
}
```

</details>

##### Использование whisper.cpp

Для более быстрого распознавания речи сервер может использовать [whisper.cpp](https://github.com/ggerganov/whisper.cpp).

Сборите проект из исходников:

```bash
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
make
```

Бинарник `whisper-cli` нужно поместить в `whisper.cpp/build/bin` в корне этого репозитория. Модели `ggml-large-v3.bin` и `ggml-silero-v5.1.2.bin` следует разместить в `whisper.cpp/models/`.

Скачивание модели:

```bash
./models/download-ggml-model.sh large-v3
```

При запуске сервер автоматически ищет этот бинарник. Пути можно переопределить переменными окружения:

- `WHISPER_CPP_CLI` — путь к `whisper-cli`
- `WHISPER_CPP_MODEL` — модель для распознавания
- `WHISPER_CPP_VAD_MODEL` — модель VAD
- `WHISPER_CPP_THREADS` — число потоков
- `WHISPER_CPP_MAX_WORKERS` — количество предварительно запущенных копий

`WHISPER_CPP_MAX_WORKERS` управляет числом рабочих процессов whisper.cpp и позволяет увеличить пропускную способность при необходимости.

#### Генерация изображений

```python
image_response = client.images.generate(
    model="argmaxinc/mlx-FLUX.1-schnell",
    prompt="A serene landscape with mountains and a lake",
    n=1,
    size="512x512"
)
```

<details>
<summary>Curl Example</summary>

```shell
curl http://localhost:10240/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "argmaxinc/mlx-FLUX.1-schnell",
    "prompt": "A cute baby sea otter",
    "n": 1,
    "size": "1024x1024"
  }'
```

</details>

#### Эмбеддинги

```python
# Generate embedding for a single text
response = client.embeddings.create(
    model="mlx-community/all-MiniLM-L6-v2-4bit", input="I like reading"
)

# Examine the response structure
print(f"Response type: {type(response)}")
print(f"Model used: {response.model}")
print(f"Embedding dimension: {len(response.data[0].embedding)}")
```

<details>
<summary>Curl Example</summary>

```shell
curl http://localhost:10240/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/all-MiniLM-L6-v2-4bit",
    "input": ["Hello world!", "Embeddings are useful for semantic search."]
  }'
```

</details>

Для дополнительных примеров обратитесь к папке [examples](examples).

## FAQ

### Как управляются модели?

MLX Omni Server использует Hugging Face для загрузки и хранения моделей. Если указанная модель еще не скачана, она будет загружена при первом запросе. Однако это может занять значительное время.

- Рекомендуется заранее скачивать модели через Hugging Face
- Для использования модели из локальной папки передайте путь в `model`

Заранее скачать модель можно через `/v1/models/load`. Ответ содержит идентификатор, который можно отследить через `/v1/models/load/{id}`.

```python
# Модель с Hugging Face
response = client.chat.completions.create(
    model="mlx-community/gemma-3-1b-it-4bit-DWQ",
    messages=[{"role": "user", "content": "Hello"}]
)

# Локальная модель
response = client.chat.completions.create(
    model="/path/to/your/local/model",
    messages=[{"role": "user", "content": "Hello"}]
)
```

Доступные модели на системе можно просмотреть запросом:

```bash
curl http://localhost:10240/v1/models
```

### Как указать модель для запроса?

Передайте параметр `model`:

```python
response = client.chat.completions.create(
    model="mlx-community/gemma-3-1b-it-4bit-DWQ",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Можно ли использовать TestClient для разработки?

Да. TestClient позволяет обращаться к API без запуска HTTP-сервера, что удобно для тестов.

```python
from openai import OpenAI
from fastapi.testclient import TestClient
from mlx_omni_server.main import app

client = OpenAI(
    http_client=TestClient(app)
)

response = client.chat.completions.create(
    model="mlx-community/gemma-3-1b-it-4bit-DWQ",
    messages=[{"role": "user", "content": "Hello"}]
)
```

Этот метод полезен для модульных тестов и быстрых итераций.

### Что делать, если при запуске появляются ошибки?

- Убедитесь, что это Mac на Apple Silicon (M1/M2/M3/M4)
- Проверьте, что версия Python не ниже 3.9
- Убедитесь, что установлена актуальная версия mlx-omni-server
- Ознакомьтесь с логами для деталей


## Лицензия

Проект распространяется по лицензии MIT — см. [LICENSE](LICENSE).

## Благодарности

- Создано с [MLX](https://github.com/ml-explore/mlx) от Apple
- Дизайн API вдохновлен [OpenAI](https://openai.com)
- Сервер базируется на [FastAPI](https://fastapi.tiangolo.com/)
- Чат от [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm)
- Генерация изображений через [mflux](https://github.com/filipstrand/mflux)
- TTS проекты [lucasnewman/f5-tts-mlx](https://github.com/lucasnewman/f5-tts-mlx) и [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio)
- STT на базе [mlx-whisper](https://github.com/ml-explore/mlx-examples/blob/main/whisper/README.md)
- Эмбеддинги от [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings)

## Отказ от ответственности

Этот проект не является официальным продуктом OpenAI или Apple. Это независимая реализация API OpenAI на базе MLX.
