# AI-Test-Task

Демо-агент: планирование исследования, сбор фактов из Wikipedia и финальный ответ через LLM. Сборка через Docker Compose (frontend + backend).

## Запуск

```bash
docker compose up -d --build
```

- Backend: [link](http://localhost:8000/docs)
- Frontend: [link](http://localhost:3000)

Healthcheck backend: `GET http://localhost:8000/health`.

## Frontend (кратко)

- **Стек:** React, Vite, TypeScript.
- **Сборка:** multi-stage Dockerfile → статика в nginx.
- **API:** браузер ходит на `POST /api/v1/answer` через прокси nginx (`/api/` → backend). Чат отправляет историю сообщений; для запроса к агенту используется последнее пользовательское сообщение как вопрос.
- **Таймауты:** у прокси в `frontend/nginx.conf` заданы `proxy_read_timeout` / `proxy_send_timeout`, чтобы длинные ответы агента не обрывались HTML-страницей ошибки вместо JSON.

## Backend (подробнее)

### Слои приложения

- **`backend/main.py`** — точка входа FastAPI: CORS, подключение роутера.
- **`backend/api/routes.py`** — HTTP: `POST /api/v1/answer`, `GET /health`. Создаёт `WikipediaKnowledgeProvider`, `OpenAICompatibleLLMProvider`, вызывает `AgentService`.
- **`backend/services/agent.py`** — оркестрация: собирает `AgentState`, вызывает скомпилированный LangGraph (`ainvoke`), маппит результат в `AnswerResponse`.
- **`backend/schemas/`** — Pydantic-модели запросов/ответов.
- **`backend/core/`** — настройки (`settings`), логирование.
- **`backend/clients/`** — внешние интеграции (см. ниже «Инструменты»).
- **`backend/agent/`** — состояние графа и узлы агента.

### Состояние агента (`backend/agent/state.py`)

`AgentState` — TypedDict с полями вроде `question`, `topic`, `intent`, `plan`, `assumptions`, `pending_wikipedia_queries`, `wikipedia_query_history`, `findings`, `warnings`, `trace`, `enough_information`, `research_iterations`, `max_research_iterations`, итоговые `answer`, `confidence`, `self_check_*`. Узлы графа читают и дополняют это состояние.

### Граф агента (LangGraph)

Граф собирается в **`backend/agent/graph.py`** функцией `build_agent_graph(provider, llm_provider)`:

```
build_plan → research_loop ⇄ (условие) → prepare_answer ⇄ (условие) → finalize_answer → END
```

**Узлы:**

1. **`build_plan`** (`backend/agent/nodes/build_plan.py`)  
   Один вызов LLM с `response_format: json_object`: план, допущения, список `wikipedia_queries` (английские сущности для поиска).

2. **`research_loop`** (`backend/agent/nodes/research_loop.py`)  
   Для каждого pending-запроса к Wikipedia (без дубликатов в истории):
   - **Инструмент Wikipedia:** `KnowledgeProvider.get_entity_page` — один лучший матч поиска + текст extract (при отсутствии страницы подставляется заглушка).
   - **LLM (извлечение):** на каждую пару «запрос + документ» — отдельный вызов (`stage: extract_document_findings`): только `findings` и `missing_information` из переданного отрывка extract (с ограничением длины текста).
   - **LLM (решение):** один вызов (`stage: research_decision`) по уже собранным findings итерации + контексту плана: `enough_information`, при необходимости доработка findings, `missing_information`, `next_wikipedia_queries`.  
   История запросов и итерации обновляются в state; при лимите итераций выход из цикла решается роутингом.

3. **`prepare_answer`** (`backend/agent/nodes/prepare_answer.py`)  
   Лёгкая локальная проверка / self-check перед финалом (без внешних API, по сути правила над state).

4. **`finalize_answer`** (`backend/agent/nodes/finalize_answer.py`)  
   Финальный вызов LLM: пользовательский ответ + `confidence` в JSON.

**Условные переходы:**

- После **`research_loop`**: если `enough_information` или достигнут `max_research_iterations` → `prepare_answer`, иначе снова `research_loop`.
- После **`prepare_answer`**: если self-check прошёл → `finalize_answer`; если итерации исчерпаны — тоже `finalize_answer`; иначе снова `build_plan` (повторное планирование).

Граф компилируется через `workflow.compile()` из **LangGraph** (`langgraph.graph.StateGraph`).

### Инструменты (внешние возможности агента)

| Инструмент | Реализация | Роль |
|------------|------------|------|
| **Wikipedia** | `backend/clients/wikipedia.py` — `WikipediaKnowledgeProvider` | HTTP к MediaWiki API (`action=query`, `list=search` + `prop=extracts`), `httpx`, повторы через `tenacity`. Протокол `KnowledgeProvider`: `get_entity_page`, также есть `get_entity_pages` для расширенных сценариев. |
| **LLM** | `backend/clients/llm.py` — `OpenAICompatibleLLMProvider` | `AsyncOpenAI` с OpenAI-совместимым API (в т.ч. Yandex Cloud AI по `llm_base_url`). Метод `generate_json`: `chat.completions` с `response_format: json_object`, парсинг JSON из ответа. Таймауты и лимиты токенов задаются в `backend/core/settings.py`. |

Отдельных «tool calls» в стиле OpenAI tools нет: Wikipedia и LLM вызываются явно из кода узлов графа.

### Конфигурация

Переменные окружения и дефолты — в **`backend/core/settings.py`** и **`backend/.env.example`**. Для Docker используется `backend/.env` (см. `docker-compose.yml`).
