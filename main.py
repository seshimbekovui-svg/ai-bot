import os
import httpx
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional, List
from openai import AsyncOpenAI

app = FastAPI()

# Клиент OpenAI
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Настройки edna
EDNA_CALLBACK_URL = os.getenv("EDNA_CALLBACK_URL", "https://kompanion.edna.kz/api/v1/chatbot")
EDNA_AUTH_TOKEN = os.getenv("EDNA_AUTH_TOKEN")

# Системный промпт — настрой под себя
SYSTEM_PROMPT = """Ты — AI-помощник Банка Компаньон. 
Отвечай кратко и по делу на вопросы клиентов о банковских продуктах и услугах.
Если вопрос требует участия оператора — скажи об этом.
Отвечай на том языке, на котором пишет клиент (русский или кыргызский)."""


class ChannelInfo(BaseModel):
    id: int
    channelType: str
    authorized: bool


class EdnaMessage(BaseModel):
    action: str
    clientId: Optional[str] = None
    threadsClientId: int
    sessionId: str
    questionId: Optional[int] = None
    questionIndex: int
    receivedAt: str
    text: Optional[str] = None
    channelInfo: ChannelInfo
    attachments: Optional[List] = []
    clientData: Optional[dict] = None
    sender: Optional[str] = None


async def send_to_edna(session_id: str, question_index: int, channel_type: str, text: str):
    """Отправляет ответ обратно в edna"""
    payload = {
        "action": "MESSAGE",
        "sessionId": session_id,
        "questionIndex": question_index + 1,
        "receivedAt": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "text": text,
        "formattedText": None,
        "code": "SUCCESS",
        "channelType": channel_type,
        "quickReplies": []
    }

    async with httpx.AsyncClient(verify=False) as client:
        response = await client.post(
            EDNA_CALLBACK_URL,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {EDNA_AUTH_TOKEN}"
            },
            timeout=10.0
        )
        print(f"edna response: {response.status_code} — {response.text}")
        return response


async def get_ai_response(user_text: str, client_name: Optional[str] = None) -> str:
    """Получает ответ от ChatGPT"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    if client_name:
        messages[0]["content"] += f"\nИмя клиента: {client_name}"

    messages.append({"role": "user", "content": user_text})

    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=500,
        temperature=0.7
    )

    return response.choices[0].message.content


@app.post("/webhook")
async def webhook(message: EdnaMessage):
    """Принимает сообщения от edna"""
    print(f"Входящее от edna: action={message.action}, text={message.text}, channel={message.channelInfo.channelType}")

    # Обрабатываем только текстовые сообщения
    if message.action != "MESSAGE" or not message.text:
        return {"status": "ignored"}

    try:
        # Получаем имя клиента если есть
        client_name = None
        if message.clientData:
            client_name = message.clientData.get("name")

        # Отправляем в ChatGPT
        ai_text = await get_ai_response(message.text, client_name)

        # Отправляем ответ обратно в edna
        await send_to_edna(
            session_id=message.sessionId,
            question_index=message.questionIndex,
            channel_type=message.channelInfo.channelType,
            text=ai_text
        )

        return {"status": "ok"}

    except Exception as e:
        print(f"Ошибка: {e}")
        # В случае ошибки — отправляем дефолтное сообщение
        await send_to_edna(
            session_id=message.sessionId,
            question_index=message.questionIndex,
            channel_type=message.channelInfo.channelType,
            text="Извините, произошла ошибка. Попробуйте позже или обратитесь к оператору."
        )
        return {"status": "error", "detail": str(e)}


@app.get("/health")
async def health():
    return {"status": "ok"}
