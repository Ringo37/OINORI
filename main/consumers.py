import json
import numpy as np
import asyncio
import os
import base64
import edge_tts
from channels.generic.websocket import AsyncWebsocketConsumer
from faster_whisper import WhisperModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pypdf import PdfReader
from io import BytesIO

load_dotenv()

model = WhisperModel("small", device="cpu", compute_type="int8")


class TranscriptConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_ENDPOINT")

        self.openai_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        # 初期 system（後で config で上書きされる）
        self.conversation_history = [
            {
                "role": "system",
                "content": (
                    "あなたは日本語で話す面接官です。"
                    "ユーザーの発言を自然な日本語に補正しつつ、"
                    "面接として適切な質問を返してください。"
                ),
            }
        ]

        # 音声処理
        self.audio_queue = asyncio.Queue()
        self.process_task = asyncio.create_task(self.process_audio_loop())
        self.SILENCE_THRESHOLD = 0.1
        self.SILENCE_LIMIT = 4
        self.audio_buffer = np.array([], dtype=np.float32)
        self.silence_chunks = 0
        self.is_speaking = False

    async def disconnect(self, code):
        if hasattr(self, "process_task"):
            self.process_task.cancel()

    async def send_initial_greeting(self):
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    *self.conversation_history,
                    {
                        "role": "user",
                        "content": (
                            "これから面接を開始してください。"
                            "最初に丁寧な挨拶をして、"
                            "自己紹介をお願いする質問をしてください。"
                            "必ず平文にして下さい(マークダウン記号は不要です)"
                        ),
                    },
                ],
                temperature=0.6,
            )

            greeting = response.choices[0].message.content

            # 【デバッグ用】生成されたテキストを確認
            print(
                f"=== OpenAI Greeting ===\n{greeting}\n======================="
            )

            if not greeting or not greeting.strip():
                print("Error: Empty greeting received.")
                return

            # 会話履歴に追加
            self.conversation_history.append(
                {"role": "assistant", "content": greeting}
            )

            # テキスト送信
            await self.send(json.dumps({"type": "ai", "message": greeting}))

            # 音声化
            voice = "ja-JP-NanamiNeural"
            tts = edge_tts.Communicate(greeting, voice, rate="+20%")

            audio_bytes = b""
            try:
                # TTS処理をtryブロックで囲む
                async for chunk in tts.stream():
                    if chunk["type"] == "audio":
                        audio_bytes += chunk["data"]

                if not audio_bytes:
                    print("Warning: No audio data generated.")
                    return

                audio_base64 = base64.b64encode(audio_bytes).decode()
                await self.send(
                    json.dumps({"type": "audio", "audio": audio_base64})
                )

            except edge_tts.exceptions.NoAudioReceived:
                print(
                    "EdgeTTS Error: NoAudioReceived. Text may be invalid or connection failed."
                )
            except Exception as e:
                print(f"EdgeTTS Unexpected Error: {e}")

        except Exception as e:
            print(f"Error in send_initial_greeting: {e}")

    async def receive(self, text_data=None, bytes_data=None):
        # ---------- 設定メッセージ ----------
        if text_data:
            data = json.loads(text_data)

            if data.get("type") == "config":
                await self.handle_config(data)
                return

        # ---------- 音声 ----------
        if bytes_data:
            chunk = np.frombuffer(bytes_data, dtype=np.float32)
            self.audio_queue.put_nowait(chunk)

    async def handle_config(self, data):
        system_prompt = data.get("prompt", "")
        resume_base64 = data.get("resume", "")

        resume_text = ""
        if resume_base64:
            resume_text = self.extract_text_from_pdf(resume_base64)

        system_message = (
            "あなたはAI面接官です。\n\n"
            f"【面接方針】\n{system_prompt}\n\n"
            f"【履歴書】\n{resume_text}\n\n"
            "この情報を元に、応募者に対して自然で実践的な面接を行ってください。"
        )

        self.conversation_history = [
            {"role": "system", "content": system_message}
        ]

        print("=== System Prompt Set ===")
        await self.send_initial_greeting()

    def extract_text_from_pdf(self, base64_pdf: str) -> str:
        try:
            pdf_bytes = base64.b64decode(base64_pdf)
            reader = PdfReader(BytesIO(pdf_bytes))

            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

            return text[:4000]  # トークン爆発防止
        except Exception as e:
            print("PDF parse error:", e)
            return ""

    async def process_audio_loop(self):
        while True:
            chunk = await self.audio_queue.get()
            amplitude = np.max(np.abs(chunk))

            if amplitude > self.SILENCE_THRESHOLD:
                self.silence_chunks = 0
                self.is_speaking = True
                self.audio_buffer = np.concatenate((self.audio_buffer, chunk))
            else:
                if self.is_speaking:
                    self.silence_chunks += 1
                    self.audio_buffer = np.concatenate(
                        (self.audio_buffer, chunk)
                    )

            if self.is_speaking and self.silence_chunks >= self.SILENCE_LIMIT:
                await self.transcribe_and_chat()
                self.is_speaking = False
            elif len(self.audio_buffer) > 16000 * 30:
                await self.transcribe_and_chat()
                self.is_speaking = False

    async def transcribe_and_chat(self):
        if len(self.audio_buffer) < 1000:
            self.reset_buffer()
            return

        audio = self.audio_buffer.copy()
        self.reset_buffer()
        loop = asyncio.get_event_loop()
        segments, _ = await loop.run_in_executor(
            None,
            lambda: model.transcribe(
                audio,
                beam_size=1,
                language="ja",
                vad_filter=True,
                condition_on_previous_text=False,
            ),
        )

        text = "".join(s.text.strip() for s in segments)
        if not text:
            return

        await self.send(json.dumps({"type": "user", "message": text}))
        await self.generate_ai_response(text)

    async def generate_ai_response(self, user_text):
        self.conversation_history.append(
            {"role": "user", "content": user_text}
        )

        response = await self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.conversation_history,
            temperature=0.7,
        )

        ai_message = response.choices[0].message.content
        self.conversation_history.append(
            {"role": "assistant", "content": ai_message}
        )

        await self.send(json.dumps({"type": "ai", "message": ai_message}))

        # --- TTS ---
        voice = "ja-JP-NanamiNeural"
        tts = edge_tts.Communicate(ai_message, voice, rate="+20%")

        audio_bytes = b""
        async for chunk in tts.stream():
            if chunk["type"] == "audio":
                audio_bytes += chunk["data"]

        audio_base64 = base64.b64encode(audio_bytes).decode()
        await self.send(json.dumps({"type": "audio", "audio": audio_base64}))

    def reset_buffer(self):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.silence_chunks = 0
