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

load_dotenv()

model = WhisperModel("small", device="cpu", compute_type="int8")


class TranscriptConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

        # --- API設定 ---
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_ENDPOINT")
        self.openai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        # --- 会話履歴 ---
        self.conversation_history = [
            {
                "role": "system",
                "content": "あなたは面接官です。ユーザーの言葉に対して日本語で返答してください。"
                "また、誤字については指摘せずに自然と修正してください。",
            }
        ]

        # --- 音声処理用 ---
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
            try:
                await self.process_task
            except asyncio.CancelledError:
                pass

    async def receive(self, text_data=None, bytes_data=None):
        if bytes_data:
            chunk = np.frombuffer(bytes_data, dtype=np.float32)
            self.audio_queue.put_nowait(chunk)

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

        audio_to_process = self.audio_buffer.copy()
        self.reset_buffer()
        loop = asyncio.get_event_loop()

        try:
            segments, info = await loop.run_in_executor(
                None,
                lambda: model.transcribe(
                    audio_to_process,
                    beam_size=1,
                    best_of=1,
                    language="ja",
                    vad_filter=True,
                    condition_on_previous_text=False,
                ),
            )

            full_text = ""
            hallucinations = [
                "ご視聴ありがとうございました。",
                "ご視聴ありがとうございました",
            ]
            for segment in segments:
                text = segment.text.strip()
                if text not in hallucinations and len(text) > 0:
                    full_text += text

            if full_text:
                print(f"User: {full_text}")
                await self.send(
                    text_data=json.dumps(
                        {"type": "user", "message": full_text}
                    )
                )
                await self.generate_ai_response(full_text)

        except Exception as e:
            print(f"Transcription Error: {e}")

    async def generate_ai_response(self, user_text):
        try:
            # 1. テキスト生成
            self.conversation_history.append(
                {"role": "user", "content": user_text}
            )
            if len(self.conversation_history) > 10:
                del self.conversation_history[1:3]

            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.conversation_history,
                temperature=0.7,
            )

            ai_message = response.choices[0].message.content
            print(f"AI: {ai_message}")

            self.conversation_history.append(
                {"role": "assistant", "content": ai_message}
            )

            # まずテキストだけ先に送る（表示のレスポンス向上）
            await self.send(
                text_data=json.dumps({"type": "ai", "message": ai_message})
            )

            # -------------------------------------------------
            # 2. 音声合成 (TTS)
            # -------------------------------------------------
            # 声の種類: "ja-JP-NanamiNeural" (女性) または "ja-JP-KeitaNeural" (男性)
            voice = "ja-JP-NanamiNeural"
            communicate = edge_tts.Communicate(ai_message, voice)

            # メモリ上に音声データをストリーミング生成
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]

            # Base64エンコードして送信
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")

            await self.send(
                text_data=json.dumps({"type": "audio", "audio": audio_base64})
            )

        except Exception as e:
            print(f"Error: {e}")
            await self.send(
                text_data=json.dumps(
                    {"type": "error", "message": "エラーが発生しました"}
                )
            )

    def reset_buffer(self):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.silence_chunks = 0
