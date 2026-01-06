import json
import numpy as np
import asyncio
import os
import base64
import edge_tts
import io
from pypdf import PdfReader
import docx
from channels.generic.websocket import AsyncWebsocketConsumer
from faster_whisper import WhisperModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
from .models import InterviewSession

load_dotenv()

model = WhisperModel("small", device="cpu", compute_type="int8")


class TranscriptConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.difficulty = "normal" # デフォルト
        self.resume_text = ""

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

    async def send_initial_greeting(self):
        try:
            # ▼▼▼ 修正: 難易度に応じたキャラ設定を読み込む ▼▼▼
            if self.difficulty == "hard":
                system_instruction = "あなたは冷徹で威圧的な圧迫面接官です。敬語ですが冷たく、短い言葉で話します。"
            elif self.difficulty == "easy":
                system_instruction = "あなたは非常に優しく、親身な新人教育担当です。明るく元気よく話します。"
            else:
                system_instruction = "あなたは一般的な企業の採用担当者です。丁寧かつ事務的に話します。"

            # ▼▼▼ 履歴書の内容があれば、それも考慮させる（任意） ▼▼▼
            resume_context = ""
            if self.resume_text:
                resume_context = f"応募者の履歴書: {self.resume_text[:500]}..." 

            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_instruction}, # ★ここが重要
                    *self.conversation_history,
                    {
                        "role": "user", 
                        "content": (
                            f"{resume_context}\n"
                            "面接を開始してください。"
                            "最初に役割に応じた挨拶をして、自己紹介をお願いする質問をしてください。"
                            "必ず平文にして下さい(マークダウン記号は不要です)"
                        ),
                    },
                ],
                temperature=0.7, # 少し創造的にする
            )
            # ▲▲▲ 修正ここまで ▲▲▲

            greeting = response.choices[0].message.content

            print(f"=== OpenAI Greeting ===\n{greeting}\n=======================")

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

            except Exception as e:
                print(f"EdgeTTS Error: {e}")

        except Exception as e:
            print(f"Error in send_initial_greeting: {e}")

    async def receive(self, text_data=None, bytes_data=None):
        if bytes_data:
            if hasattr(self, 'process_audio'):
                await self.process_audio(bytes_data)
            print("音声データ受信 (bytes)")
            return

        if text_data:
            try:
                data = json.loads(text_data)
                msg_type = data.get("type")

                if msg_type == "config":
                    self.difficulty = data.get("difficulty", "normal")
                    print(f"難易度設定: {self.difficulty}")

                    resume_data = data.get("resume")
                    if resume_data:
                        try:
                            file_name = resume_data.get("fileName", "").lower()
                            file_content = resume_data.get("content", "")

                            if "," in file_content:
                                _, encoded = file_content.split(",", 1)
                            else:
                                encoded = file_content

                            # デコード
                            decoded_bytes = base64.b64decode(encoded)
                            file_stream = io.BytesIO(decoded_bytes)
                            
                            # 変数の初期化（これを忘れるとエラーになります）
                            extracted_text = ""

                            # ファイル形式による分岐
                            if file_name.endswith(".pdf"):
                                reader = PdfReader(file_stream)
                                for page in reader.pages:
                                    extracted_text += page.extract_text() + "\n"
                                print("PDFとして処理しました")

                            elif file_name.endswith(".docx"):
                                doc = docx.Document(file_stream)
                                for para in doc.paragraphs:
                                    extracted_text += para.text + "\n"
                                print("Wordとして処理しました")
                            
                            else:
                                try:
                                    extracted_text = decoded_bytes.decode("utf-8")
                                except:
                                    extracted_text = "【読込不可のファイル形式】"

                            if len(extracted_text) > 2000:
                                extracted_text = extracted_text[:2000] + "\n...(省略)..."

                            self.resume_text = extracted_text
                            print(f"抽出文字数: {len(self.resume_text)}")

                        except Exception as e:
                            print(f"ファイル読み込みエラー: {e}")
                            self.resume_text = ""
                    
                    await self.send_initial_greeting()
                    return

                elif msg_type == "resume":
                    if hasattr(self, 'handle_resume_upload'):
                        await self.handle_resume_upload(data)
                    return

                elif msg_type == "finish":
                    await self.generate_evaluation()
                    return

            except json.JSONDecodeError:
                print("JSONデコードエラー: データ形式が不正です")
            except Exception as e:
                print(f"予期せぬエラー: {e}")

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

    async def generate_evaluation(self):
        # 会話ログを取得
        formatted_history = []
        for item in self.conversation_history:
            if isinstance(item, dict):
                role = item.get("role", "Unknown")
                content = item.get("content") or item.get("message") or ""
                formatted_history.append(f"{role}: {content}")
            else:
                formatted_history.append(str(item))
        conversation_text = "\n".join(formatted_history)

        # ▼▼▼ 2. ここで履歴書の有無を明確に分岐させる（ここが重要！） ▼▼▼
        if self.resume_text and len(self.resume_text) > 50:
            # ちゃんと文字が入っている場合
            resume_prompt = f"【応募者の履歴書情報】\n{self.resume_text}"
        else:
            # 中身が空、または短すぎる場合
            resume_prompt = "【重要：応募者は履歴書を提出しましたが、白紙（または読み取り不可）でした。基本的にeasy5点,normal10点,hard,15点程度の点数減少を行ってください。】"

        # 3. ユーザープロンプトの作成
        conversation_text = "\n".join(formatted_history)
        user_content = f"{resume_prompt}\n\n【面接のやり取り】\n{conversation_text}"

        # ▼ 難易度に応じたプロンプト分岐
        if self.difficulty == "easy":
            base_prompt = """
            あなたは非常に優しく、応援してくれる新人研修担当のメンターです。
            良いところを最大限に見つけ、ユーザーの自信をつけさせてください。
            採点基準：甘め（基本80点スタート）。
            アドバイス：具体的かつ、次につながるポジティブな言葉のみ。
            """
            temperature = 0.8
            
        elif self.difficulty == "hard":
            base_prompt = """
            あなたは世界トップ企業の冷徹な最終面接官です。（通称：お祈りモード）
            論理の飛躍や具体性の欠如を一切許しません。
            採点基準：激辛（基本30点スタート）。70点以上は即採用レベルのみ。
            フィードバック：なぜダメなのかを痛烈に指摘し、厳しい現実を突きつけてください。
            """
            temperature = 0.3 
            
        else: 
            base_prompt = """
            あなたは一般的な企業の採用担当者です。
            ビジネスマンとしての適性を公平に判断してください。
            採点基準：標準（基本50点スタート）。
            良い点と悪い点をバランスよく指摘してください。
            """
            temperature = 0.5

        system_instruction = f"""
        {base_prompt}

        出力フォーマット（必ずJSONで返してください）:
        {{
            "score": 0〜100の整数,
            "good_points": "文字列",
            "bad_points": "文字列",
            "advice": "文字列"
        }}
        【重要ルール】
        - 矛盾した評価を絶対にしないでください。（例：「履歴書を出したのは良い」と言いつつ「履歴書がない」と批判するなど）
        - 履歴書の情報が無い、または極端に少ない場合は、easyでは軽く指摘、normalでは必要性を述べて、hardでは「悪い点」で徹底的に批判してください。
        - 評価スコアとコメントの整合性を取ってください。
        """

        # APIコール (temperatureも動的に変える)
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_content}
                ],
                response_format={"type": "json_object"},
                temperature=temperature, 
            )
            
            result_json_str = response.choices[0].message.content
            result_data = json.loads(result_json_str)

            # 1. データベースに保存 (Sync_to_asyncでラップして保存)
            from channels.db import database_sync_to_async
            
            @database_sync_to_async
            def save_record():
                return InterviewSession.objects.create(
                    score=result_data.get("score", 0),
                    feedback_text=result_json_str, # JSONのまま保存しておくと便利
                    conversation_log=self.conversation_history
                )
            
            await save_record()

            # 2. フロントエンドに結果を送信
            await self.send(text_data=json.dumps({
                "type": "evaluation_result",
                "data": result_data
            }))
            
            print("評価完了・保存済み")

        except Exception as e:
            print(f"評価エラー: {e}")
            error_message = "申し訳ありません。AIの利用制限に達したため、評価を作成できませんでした。"
            if "429" in str(e):
                error_message = "⚠️ 本日のAI利用上限に達しました。明日以降に再度お試しください。"

            await self.send(text_data=json.dumps({
                "type": "system", 
                "message": error_message
            }))

    def reset_buffer(self):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.silence_chunks = 0
