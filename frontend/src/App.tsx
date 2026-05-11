import { useEffect, useMemo, useState } from 'react'
import type { FormEvent, KeyboardEvent } from 'react'
import './App.css'

type Role = 'user' | 'assistant'

type ChatMessage = {
  role: Role
  content: string
}

type AnswerResponse = {
  answer: string
  message: ChatMessage
  confidence: 'low' | 'medium' | 'high'
  findings: string[]
  warnings: string[]
  assumptions: string[]
}

const initialMessages: ChatMessage[] = [
  {
    role: 'assistant',
    content:
      'Спроси, сколько времени понадобится гепарду, чтобы пересечь Москву-реку по Большому Каменному мосту.',
  },
]

function App() {
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages)
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [loadingDots, setLoadingDots] = useState('.')
  const [error, setError] = useState<string | null>(null)

  const canSend = useMemo(() => input.trim().length > 0 && !isLoading, [input, isLoading])

  const submitMessage = async () => {
    const content = input.trim()
    if (!content || isLoading) {
      return
    }

    const nextUserMessage: ChatMessage = { role: 'user', content }
    const nextMessages = [...messages, nextUserMessage]

    setMessages(nextMessages)
    setInput('')
    setError(null)
    setIsLoading(true)

    try {
      const response = await fetch('/api/v1/answer', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ messages: nextMessages }),
      })
      const contentType = response.headers.get('content-type') || ''
      const responseText = await response.text()
      const isJson = contentType.includes('application/json')
      const payload = isJson
        ? (JSON.parse(responseText) as AnswerResponse | { detail?: string })
        : null

      if (!response.ok) {
        throw new Error(
          typeof payload === 'object' && payload && 'detail' in payload
            ? payload.detail || 'Не удалось получить ответ'
            : isJson
              ? 'Не удалось получить ответ'
              : `Сервер вернул не JSON (${response.status}). Проверь backend/nginx таймауты.`,
        )
      }
      if (!payload) {
        throw new Error('Сервер вернул не JSON при успешном статусе')
      }

      const answerPayload = payload as AnswerResponse
      setMessages((currentMessages) => [...currentMessages, answerPayload.message])
    } catch (submissionError) {
      setError(
        submissionError instanceof Error
          ? submissionError.message
          : 'Что-то пошло не так во время запроса',
      )
    } finally {
      setIsLoading(false)
    }
  }

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    await submitMessage()
  }

  const handleInputKeyDown = async (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== 'Enter') {
      return
    }
    if (event.ctrlKey) {
      return
    }
    event.preventDefault()
    await submitMessage()
  }

  const handleClear = () => {
    setMessages(initialMessages)
    setInput('')
    setError(null)
  }

  useEffect(() => {
    if (!isLoading) {
      setLoadingDots('.')
      return
    }

    const intervalId = window.setInterval(() => {
      setLoadingDots((currentDots) => (currentDots.length === 3 ? '.' : `${currentDots}.`))
    }, 400)

    return () => {
      window.clearInterval(intervalId)
    }
  }, [isLoading])

  return (
    <main className="app-shell">
      <section className="chat-panel">
        <header className="chat-header">
          <div>
            <h1>Bridge Time Agent</h1>
            <p>Простой чат поверх backend-а с агентным расчетом и self-check.</p>
          </div>
          <button type="button" className="secondary-button" onClick={handleClear}>
            Очистить чат
          </button>
        </header>

        <div className="messages">
          {messages.map((message, index) => (
            <article
              key={`${message.role}-${index}`}
              className={`message message-${message.role}`}
            >
              <span className="message-role">
                {message.role === 'user' ? 'Вы' : 'Ассистент'}
              </span>
              <p>{message.content}</p>
            </article>
          ))}

          {isLoading ? (
            <article className="message message-assistant">
              <span className="message-role">Ассистент</span>
              <p>Думаю над ответом{loadingDots}</p>
            </article>
          ) : null}
        </div>

        <form className="chat-form" onSubmit={handleSubmit}>
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={handleInputKeyDown}
            placeholder="Введите сообщение"
            rows={4}
          />
          <div className="chat-actions">
            <button type="submit" className="primary-button" disabled={!canSend}>
              Отправить
            </button>
          </div>
        </form>

        {error ? <p className="error-banner">{error}</p> : null}
      </section>
    </main>
  )
}

export default App
