import json
import numpy as np
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer
from faster_whisper import WhisperModel


model = WhisperModel("small", device="cpu", compute_type="int8")


class TranscriptConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

        # 音声データを溜めるキュー（受信と処理を分離するため）
        self.audio_queue = asyncio.Queue()

        # 処理ループをバックグラウンドで開始
        self.process_task = asyncio.create_task(self.process_audio_loop())

        # 設定
        self.SILENCE_THRESHOLD = 0.1
        self.SILENCE_LIMIT = 2  # 少し余裕を持たせる（調整可）

        # 状態管理
        self.audio_buffer = np.array([], dtype=np.float32)
        self.silence_chunks = 0
        self.is_speaking = False

    async def disconnect(self, code):
        # タスクのキャンセル処理
        if hasattr(self, "process_task"):
            self.process_task.cancel()
            try:
                await self.process_task
            except asyncio.CancelledError:
                pass

    async def receive(self, text_data=None, bytes_data=None):
        if bytes_data:
            # --- ポイント3: 受信処理はキューに入れるだけ（ここでawaitしない） ---
            # これにより、推論中でも次の音声パケットを取りこぼさない
            chunk = np.frombuffer(bytes_data, dtype=np.float32)
            self.audio_queue.put_nowait(chunk)

    async def process_audio_loop(self):
        """音声データを絶え間なく処理するループ"""
        while True:
            # キューからデータを取り出す（データが来るまで待機）
            chunk = await self.audio_queue.get()

            amplitude = np.max(np.abs(chunk))

            # 1. 音声がある場合
            if amplitude > self.SILENCE_THRESHOLD:
                self.silence_chunks = 0
                self.is_speaking = True
                self.audio_buffer = np.concatenate((self.audio_buffer, chunk))

            # 2. 無音の場合
            else:
                if self.is_speaking:  # 話している最中の無音のみカウント
                    self.silence_chunks += 1
                    # 語尾切れ防止のため少しだけ無音も含める
                    self.audio_buffer = np.concatenate(
                        (self.audio_buffer, chunk)
                    )

            # --- 判定ロジック ---

            # A. 話し終わり判定
            # ポイント2: 「バッファサイズ > 8000」の制約を削除し、短い返事も即座に反応させる
            if self.is_speaking and self.silence_chunks >= self.SILENCE_LIMIT:
                await self.transcribe_audio()
                self.is_speaking = False

            # B. バッファ溢れ防止（強制変換）
            elif len(self.audio_buffer) > 16000 * 10:
                await self.transcribe_audio()
                self.is_speaking = False

    async def transcribe_audio(self):
        if len(self.audio_buffer) < 1000:  # あまりに短すぎるノイズは無視
            self.audio_buffer = np.array([], dtype=np.float32)
            self.silence_chunks = 0
            return

        audio_to_process = self.audio_buffer.copy()

        # バッファリセット（次を受け入れ可能にする）
        self.audio_buffer = np.array([], dtype=np.float32)
        self.silence_chunks = 0

        loop = asyncio.get_event_loop()

        # 推論実行（ブロッキングを防ぐためExecutorで実行）
        # vad_filter=True は内部でもVADを行い、精度を上げますが、速度重視ならFalseも検討
        segments, info = await loop.run_in_executor(
            None,
            lambda: model.transcribe(
                audio_to_process,
                beam_size=1,  # 高速化のためbeam_sizeを下げる（デフォルト5→1）
                best_of=1,  # 高速化
                language="ja",
                vad_filter=True,  # 精度のためにTrue推奨だが、ここが重ければFalseへ
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
            await self.send(text_data=json.dumps({"message": full_text}))
