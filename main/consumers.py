import json
import numpy as np
import asyncio
import os
from channels.generic.websocket import AsyncWebsocketConsumer
from faster_whisper import WhisperModel
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()


model = WhisperModel("small", device="cpu", compute_type="int8")


class TranscriptConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

        # -------------------------------------------------
        # 1. OpenAI クライアント設定 (INIAD対応)
        # -------------------------------------------------
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_ENDPOINT")

        if not api_key:
            print("Warning: OPENAI_API_KEY is not set.")

        self.openai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        # -------------------------------------------------
        # 2. 会話履歴の初期化
        # -------------------------------------------------
        self.conversation_history = [
            {
                "role": "system",
                "content": "あなたは面接官です。ユーザーの言葉に対して日本語で返答してください。"
                "また、誤字については指摘せずに自然と修正してください。",
            }
        ]

        # -------------------------------------------------
        # 3. 音声処理用変数の初期化
        # -------------------------------------------------
        self.audio_queue = asyncio.Queue()
        self.process_task = asyncio.create_task(self.process_audio_loop())

        # VAD（無音検知）の設定
        self.SILENCE_THRESHOLD = 0.1
        self.SILENCE_LIMIT = 4  # 無音が何回続いたら「話し終わり」とみなすか

        # バッファ関連
        self.audio_buffer = np.array([], dtype=np.float32)
        self.silence_chunks = 0
        self.is_speaking = False

    async def disconnect(self, code):
        """切断時のクリーンアップ"""
        if hasattr(self, "process_task"):
            self.process_task.cancel()
            try:
                await self.process_task
            except asyncio.CancelledError:
                pass

    async def receive(self, text_data=None, bytes_data=None):
        """音声データの受信（ノンブロッキング）"""
        if bytes_data:
            # 受信したバイナリデータをfloat32に変換してキューに入れる
            chunk = np.frombuffer(bytes_data, dtype=np.float32)
            self.audio_queue.put_nowait(chunk)

    async def process_audio_loop(self):
        """バックグラウンドで音声データを監視・結合・判定するループ"""
        while True:
            # キューからデータを取得（データが来るまでここで待機）
            chunk = await self.audio_queue.get()
            amplitude = np.max(np.abs(chunk))

            # --- VADロジック ---
            if amplitude > self.SILENCE_THRESHOLD:
                # 声が出ている
                self.silence_chunks = 0
                self.is_speaking = True
                self.audio_buffer = np.concatenate((self.audio_buffer, chunk))
            else:
                # 無音
                if self.is_speaking:
                    self.silence_chunks += 1
                    # 語尾が切れないように少しだけ無音も含めておく
                    self.audio_buffer = np.concatenate(
                        (self.audio_buffer, chunk)
                    )

            # --- 判定と実行 ---

            # A. 話し終わり判定 (発話中 かつ 一定時間無音が続いた)
            if self.is_speaking and self.silence_chunks >= self.SILENCE_LIMIT:
                await self.transcribe_and_chat()
                self.is_speaking = False

            # B. バッファ溢れ防止 (長すぎる発話は強制的に処理)
            elif len(self.audio_buffer) > 16000 * 30:  # 30秒以上
                await self.transcribe_and_chat()
                self.is_speaking = False

    async def transcribe_and_chat(self):
        """Whisperで文字起こし -> 結果があればOpenAIへ送信"""
        # 極端に短いノイズは無視
        if len(self.audio_buffer) < 1000:
            self.reset_buffer()
            return

        # 処理用にバッファをコピーしてリセット
        audio_to_process = self.audio_buffer.copy()
        self.reset_buffer()

        loop = asyncio.get_event_loop()

        try:
            # -------------------------------------------------
            # 1. 音声認識 (Faster Whisper) - CPUバウンドなのでExecutorで実行
            # -------------------------------------------------
            segments, info = await loop.run_in_executor(
                None,
                lambda: model.transcribe(
                    audio_to_process,
                    beam_size=1,  # 高速化
                    best_of=1,  # 高速化
                    language="ja",
                    vad_filter=True,  # Whisper内蔵VADも有効化
                    condition_on_previous_text=False,
                ),
            )

            full_text = ""
            # Whisperの幻覚(Hallucination)フィルタ
            hallucinations = [
                "ご視聴ありがとうございました。",
                "ご視聴ありがとうございました",
            ]

            for segment in segments:
                text = segment.text.strip()
                if text not in hallucinations and len(text) > 0:
                    full_text += text

            # テキストが得られた場合のみ会話処理へ
            if full_text:
                print(f"User: {full_text}")

                # フロントエンドに「ユーザーの発言」として送信
                await self.send(
                    text_data=json.dumps(
                        {"type": "user", "message": full_text}
                    )
                )

                # OpenAIへ投げる
                await self.generate_ai_response(full_text)

        except Exception as e:
            print(f"Transcription Error: {e}")

    async def generate_ai_response(self, user_text):
        """OpenAI APIで応答を生成"""
        try:
            # 履歴に追加
            self.conversation_history.append(
                {"role": "user", "content": user_text}
            )

            # 履歴が長くなりすぎたら古いものを削除 (Systemプロンプトは維持)
            if len(self.conversation_history) > 10:
                del self.conversation_history[1:3]

            # -------------------------------------------------
            # 2. OpenAI API リクエスト (非同期)
            # -------------------------------------------------
            # INIAD環境などでは使用可能なモデル名を確認してください (例: gpt-3.5-turbo, gpt-4o など)
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.conversation_history,
                temperature=0.7,
            )

            ai_message = response.choices[0].message.content
            print(f"AI: {ai_message}")

            # 履歴に追加
            self.conversation_history.append(
                {"role": "assistant", "content": ai_message}
            )

            # フロントエンドに「AIの回答」として送信
            await self.send(
                text_data=json.dumps({"type": "ai", "message": ai_message})
            )

        except Exception as e:
            print(f"OpenAI Error: {e}")
            await self.send(
                text_data=json.dumps(
                    {
                        "type": "error",
                        "message": "AIとの通信でエラーが発生しました。",
                    }
                )
            )

    def reset_buffer(self):
        """音声バッファのリセット"""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.silence_chunks = 0
