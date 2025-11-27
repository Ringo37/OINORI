import json
import numpy as np
from channels.generic.websocket import AsyncWebsocketConsumer
from faster_whisper import WhisperModel
import asyncio


model = WhisperModel("medium", device="cpu", compute_type="int8")


class TranscriptConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        # 音声バッファ
        self.audio_buffer = np.array([], dtype=np.float32)

        # 無音検知用のカウンター
        self.silence_chunks = 0

        # 設定: 無音とみなす閾値（環境に合わせて調整が必要）
        self.SILENCE_THRESHOLD = 0.05

        # 設定: 何回連続で無音なら確定するか（1チャンク約0.25秒なので、2回で約0.5秒）
        self.SILENCE_LIMIT = 1

    async def disconnect(self, code):
        pass

    async def receive(self, text_data=None, bytes_data=None):
        if bytes_data:
            chunk = np.frombuffer(bytes_data, dtype=np.float32)

            # 1. 音量（振幅の最大値）を計算
            amplitude = np.max(np.abs(chunk))

            # 2. 音声がある場合
            if amplitude > self.SILENCE_THRESHOLD:
                self.silence_chunks = 0  # 無音カウンタリセット
                self.audio_buffer = np.concatenate((self.audio_buffer, chunk))

            # 3. 無音の場合
            else:
                # バッファにデータがある時だけ無音カウントを進める（話し始め待ちの無音は無視）
                if len(self.audio_buffer) > 0:
                    self.silence_chunks += 1
                    # 無音部分も少し含めないと語尾が切れるので追加しておく
                    self.audio_buffer = np.concatenate(
                        (self.audio_buffer, chunk)
                    )

            # --- 判定ロジック ---

            # A. 無音が一定続き、かつバッファがある程度溜まっている -> 話し終わりとみなして変換
            if (
                self.silence_chunks >= self.SILENCE_LIMIT
                and len(self.audio_buffer) > 8000
            ):  # 0.5秒以上
                await self.transcribe_audio()

            # B. ずっと喋り続けてバッファが長くなりすぎた場合（10秒以上など） -> 強制変換
            # メモリ溢れとレスポンス遅延を防ぐため
            elif len(self.audio_buffer) > 16000 * 10:
                await self.transcribe_audio()

    async def transcribe_audio(self):
        # 処理対象をコピー
        audio_to_process = self.audio_buffer.copy()

        # バッファとカウンタをリセット
        self.audio_buffer = np.array([], dtype=np.float32)
        self.silence_chunks = 0

        # 推論実行
        loop = asyncio.get_event_loop()

        # VADフィルタはここでも有効にしておくと安心です
        segments, info = await loop.run_in_executor(
            None,
            lambda: model.transcribe(
                audio_to_process,
                beam_size=5,
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
            await self.send(text_data=json.dumps({"message": full_text}))
