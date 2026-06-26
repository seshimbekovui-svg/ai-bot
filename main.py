import os
import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
from openai import AsyncOpenAI

app = FastAPI()

client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

EDNA_CALLBACK_URL = os.getenv("EDNA_CALLBACK_URL", "https://kompanion.edna.kz/api/v1/chatbot")
EDNA_AUTH_TOKEN = os.getenv("EDNA_AUTH_TOKEN")

SYSTEM_PROMPT = """Ты — AI-помощник Банка Компаньон по имени Нурай 🤖

Правила:
- Отвечай ТОЛЬКО на основе информации с сайта https://www.kompanion.kg/
- Если информации нет на сайте — скажи что не знаешь и предложи обратиться к оператору
- Отвечай КОРОТКО — максимум 3-4 предложения
- НЕ задавай вопросы клиенту
- Используй эмоджи в ответах 😊
- Отвечай ТОЛЬКО на русском или кыргызском языке
- Если клиент пишет на кыргызском — отвечай на кыргызском 🇰🇬
- Если клиент пишет на русском — отвечай на русском 🇷🇺
- Если клиент пишет на другом языке — отвечай на русском
- Представляйся как Нурай только если клиент спрашивает кто ты"""

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


async def send_to_edna(session_id: str, question_index: int, received_at: str, text: str):
    payload = {
        "action": "MESSAGE",
        "sessionId": session_id,
        "questionIndex": question_index,
        "receivedAt": received_at,
        "text": text,
        "code": "SUCCESS"
    }
    print(f"Отправляем в edna: {payload}")
    async with httpx.AsyncClient(verify=False) as http_client:
        response = await http_client.post(
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
    system = SYSTEM_PROMPT
    if client_name:
        system += f"\nИмя клиента: {client_name}"

    response = await client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_text}
        ],
        max_tokens=500,
        temperature=0.7
    )
    return response.choices[0].message.content


@app.post("/webhook")
async def webhook(message: EdnaMessage):
    print(f"Входящее: action={message.action}, text={message.text}, channel={message.channelInfo.channelType}")

    if message.action != "MESSAGE" or not message.text:
        return {"status": "ignored"}

    try:
        client_name = None
        if message.clientData:
            client_name = message.clientData.get("name")

        ai_text = await get_ai_response(message.text, client_name)

        await send_to_edna(
            session_id=message.sessionId,
            question_index=message.questionIndex,
            received_at=message.receivedAt,
            text=ai_text
        )

        return {"status": "ok"}

    except Exception as e:
        print(f"Ошибка: {e}")
        await send_to_edna(
            session_id=message.sessionId,
            question_index=message.questionIndex,
            received_at=message.receivedAt,
            text="Извините, произошла ошибка. Попробуйте позже или обратитесь к оператору."
        )
        return {"status": "error", "detail": str(e)}


@app.get("/health")
async def health():
    return {"status": "ok"}
