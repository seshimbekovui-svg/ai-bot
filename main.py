import os
import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
from openai import AsyncOpenAI
from bs4 import BeautifulSoup

app = FastAPI()

client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

EDNA_CALLBACK_URL = os.getenv("EDNA_CALLBACK_URL", "https://kompanion.edna.kz/api/v1/chatbot")
EDNA_AUTH_TOKEN = os.getenv("EDNA_AUTH_TOKEN")

SYSTEM_PROMPT = """Ты — AI-помощник Банка Компаньон по имени Нурай 🤖

ВАЖНЫЕ ПРАВИЛА:
- НЕ здоровайся и НЕ приветствуй клиента — сразу отвечай по делу
- Отвечай на вопросы используя ТОЛЬКО предоставленную информацию с сайта банка
- Отвечай КОРОТКО и ПОЛЕЗНО — максимум 4-5 предложений
- НЕ задавай вопросы клиенту
- Используй эмоджи 😊
- Отвечай ТОЛЬКО на русском или кыргызском — в зависимости от языка клиента
- Если информации нет — направь на сайт kompanion.kg или к оператору 👨‍💼
- Представляйся как Нурай только если клиент спрашивает кто ты"""


async def fetch_site_content(query: str) -> str:
    urls = ["https://www.kompanion.kg/"]

    query_lower = query.lower()
    if any(w in query_lower for w in ["кредит", "займ", "кредиттер", "насыя"]):
        urls = ["https://www.kompanion.kg/credits/", "https://www.kompanion.kg/"]
    elif any(w in query_lower for w in ["депозит", "вклад", "депозиттер"]):
        urls = ["https://www.kompanion.kg/deposits/", "https://www.kompanion.kg/"]
    elif any(w in query_lower for w in ["карт", "карта"]):
        urls = ["https://www.kompanion.kg/cards/", "https://www.kompanion.kg/"]
    elif any(w in query_lower for w in ["перевод", "которуу"]):
        urls = ["https://www.kompanion.kg/transfers/", "https://www.kompanion.kg/"]
    elif any(w in query_lower for w in ["мобильн", "приложен", "тиркеме"]):
        urls = ["https://www.kompanion.kg/mobile-bank/", "https://www.kompanion.kg/"]

    content = ""
    async with httpx.AsyncClient(verify=False, timeout=10.0) as http_client:
        for url in urls[:2]:
            try:
                response = await http_client.get(url, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                })
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")
                    for tag in soup(["script", "style", "nav", "footer", "header"]):
                        tag.decompose()
                    text = soup.get_text(separator=" ", strip=True)
                    content += f"\n[{url}]:\n{text[:3000]}\n"
            except Exception as e:
                print(f"Ошибка загрузки {url}: {e}")

    return content if content else "Информация с сайта недоступна."


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
    site_content = await fetch_site_content(user_text)

    system = SYSTEM_PROMPT
    if client_name:
        system += f"\nИмя клиента: {client_name}"
    system += f"\n\nИнформация с сайта Банка Компаньон:\n{site_content}"

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
            text="Извините, произошла ошибка. Попробуйте позже или обратитесь к оператору 👨‍💼"
        )
        return {"status": "error", "detail": str(e)}


@app.get("/health")
async def health():
    return {"status": "ok"}
