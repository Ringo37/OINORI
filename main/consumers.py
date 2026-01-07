import json
import numpy as np
import asyncio
import os
import base64
import edge_tts
import io
import docx
from pypdf import PdfReader
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from faster_whisper import WhisperModel
import google.generativeai as genai
from dotenv import load_dotenv
from .models import InterviewSession
from .prompts import get_system_prompt, BASE_INSTRUCTION

load_dotenv()

model = WhisperModel("small", device="cpu", compute_type="int8")

class TranscriptConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resume_text = ""
        self.system_prompt = None 
        self.conversation_history = [] 
        self.tts_voice = "ja-JP-NanamiNeural"
        self.difficulty = "normal"
        self.gender = "female"

    async def connect(self):
        await self.accept()

        # --- Gemini API設定 ---
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("Error: GEMINI_API_KEY not found in .env")
            await self.send(json.dumps({"type": "error", "message": "APIキー設定エラー"}))
            await self.close()
            return
            
        genai.configure(api_key=api_key)

        # --- 音声処理用変数 ---
        self.audio_queue = asyncio.Queue()
        self.process_task = asyncio.create_task(self.process_audio_loop())
        self.SILENCE_THRESHOLD = 0.1
        self.SILENCE_LIMIT = 4 # 無音検知チャンク数
        self.audio_buffer = np.array([], dtype=np.float32)
        self.silence_chunks = 0
        self.is_speaking = False

        print("WebSocket Connected (Waiting for config)")

    async def disconnect(self, code):
        if hasattr(self, "process_task"):
            self.process_task.cancel()
            try:
                await self.process_task
            except asyncio.CancelledError:
                pass

    def _get_gemini_model(self, response_mime_type="text/plain"):
        """Geminiモデルの初期化 (現在のsystem_promptを使用)"""
        
        # まだ設定が来ていない場合のフォールバックとして、prompts.pyのBASEを使用
        instruction = self.system_prompt if self.system_prompt else BASE_INSTRUCTION

        generation_config = {
            "temperature": 0.7,
            "max_output_tokens": 1024,
            "response_mime_type": response_mime_type,
        }
        
        return genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=generation_config,
            system_instruction=instruction
        )

    async def generate_audio_and_send(self, text):
        """テキストを音声化してフロントエンドに送信"""
        voice = self.tts_voice
        tts = edge_tts.Communicate(text, voice=voice, rate="+20%")
        audio_bytes = b""
        try:
            async for chunk in tts.stream():
                if chunk["type"] == "audio":
                    audio_bytes += chunk["data"]

            if audio_bytes:
                audio_base64 = base64.b64encode(audio_bytes).decode()
                await self.send(json.dumps({"type": "audio", "audio": audio_base64}))
        except Exception as e:
            print(f"EdgeTTS Error: {e}")

    async def send_initial_greeting(self):
        """設定されたペルソナに基づいて最初の挨拶を行う"""
        try:
            prompt = (
                "面接を開始します。"
                "あなたの設定（役割・口調）に基づき、応募者に対して最初の挨拶と、自己紹介を促す発言を行ってください。"
                "回答は必ず平文で、Markdown記号を含めないでください。"
                "長さは短めに、20文字〜40文字程度でお願いします。"
            )

            model = self._get_gemini_model()
            response = await model.generate_content_async(prompt)
            greeting = response.text

            print(f"=== Initial Greeting ===\n{greeting}\n=======================")

            if not greeting or not greeting.strip():
                greeting = "面接を始めます。自己紹介をお願いします。"

            self.conversation_history.append({"role": "model", "parts": [greeting]})

            await self.send(json.dumps({"type": "ai", "message": greeting}))
            await self.generate_audio_and_send(greeting)

        except Exception as e:
            print(f"Error in send_initial_greeting: {e}")

    async def generate_ai_response(self, user_text):
        try:
            # ユーザー発言を履歴に追加
            self.conversation_history.append({"role": "user", "parts": [user_text]})

            model = self._get_gemini_model()
            
            # チャット履歴を使って応答生成
            # (直前の発言以外をhistoryとして渡す仕様に対応)
            chat = model.start_chat(history=self.conversation_history[:-1])
            response = await chat.send_message_async(user_text)
            
            ai_text = response.text
            print(f"=== Gemini Response ===\n{ai_text}\n=======================")

            # AI発言を履歴に追加
            self.conversation_history.append({"role": "model", "parts": [ai_text]})

            # テキストと音声を送信
            await self.send(json.dumps({"type": "ai", "message": ai_text}))
            await self.generate_audio_and_send(ai_text)

        except Exception as e:
            print(f"Response Error: {e}")

    async def generate_evaluation(self):
        """面接終了後の評価生成"""
        # 会話ログの整形
        formatted_history = []
        for item in self.conversation_history:
            role = item.get("role")
            text = item.get("parts", [""])[0]
            role_label = "面接官" if role == "model" else "応募者"
            formatted_history.append(f"{role_label}: {text}")
        
        conversation_text = "\n".join(formatted_history)

        # 評価用プロンプト
        # 現在のペルソナ設定(system_prompt)を引き継ぎつつ、評価モードに切り替える
        base_prompt = self.system_prompt if self.system_prompt else BASE_INSTRUCTION
        
        evaluation_system_instruction = f"""
        {base_prompt}
        
        === 面接終了 ===
        これまでの会話ログと履歴書を元に、応募者の評価を行ってください。
        あなたは今の役割（難易度設定）の視点から厳正に評価する必要があります。

        【重要：ユーザーの発言が極端に少ない、または無い場合の対応】
        会話ログを確認し、ユーザー（応募者）の発言がほとんどない、あるいは全くない場合は、
        「情報不足」として処理するのではなく、**「意欲欠如」「コミュニケーションの拒否」とみなして、以下のトーンで厳しく批判してください。
        また、例のまま出力することは禁じます。

        1.総評アドバイス:
        「評価できません」と逃げるのではなく、「面接の場に来ておきながら沈黙を貫くのは、時間を無駄にする行為であり、社会人として失礼である」という旨を強く指摘してください。
        「準備不足以前に、対話をする姿勢がなっていません」と厳しく叱責してください。
        お祈りメールの定型文（今後のご活躍を〜）の前に、厳しい苦言を呈してください。

        2.改善点:
        辛辣な内容にしてください。
        例 (このまま使用することを禁じます)
        「一言も発しないため、改善以前の問題である」
        「コミュニケーションを取る意思が見られない」
        「マイクの不調でないなら、ただの職務放棄とみなされる」

        3.良かった点:
        皮肉を込めて「（該当なし）」あるいは「時間通りに席についたことくらいしか褒める点がない」としてください。

        出力フォーマット（必ずJSON）:
        {{
            "score": 0〜100の整数,
            "good_points": "良かった点（Markdown文字列）",
            "bad_points": "改善点（Markdown文字列）",
            "advice": "総評アドバイス（Markdown文字列）"
        }}
        """
        
        user_content = f"【会話ログ】\n{conversation_text}\n\n評価を出力してください。"

        try:
            # JSONモードで生成
            generation_config = {
                "temperature": 0.5,
                "response_mime_type": "application/json",
            }
            
            eval_model = genai.GenerativeModel(
                model_name="gemini-2.0-flash-exp",
                generation_config=generation_config,
                system_instruction=evaluation_system_instruction
            )

            response = await eval_model.generate_content_async(user_content)
            result_json_str = response.text
            result_data = json.loads(result_json_str)

            @database_sync_to_async
            def save_record():
                return InterviewSession.objects.create(
                    score=result_data.get("score", 0),
                    feedback_text=result_data,
                    conversation_log=self.conversation_history,
                    difficulty=self.difficulty,
                    gender=self.gender
                )
            await save_record()

            # 結果送信
            await self.send(text_data=json.dumps({
                "type": "evaluation_result",
                "data": result_data
            }))
            print("評価完了(Gemini)")

        except Exception as e:
            print(f"評価エラー: {e}")
            await self.send(text_data=json.dumps({
                "type": "error", 
                "message": "評価の生成に失敗しました。"
            }))

    async def receive(self, text_data=None, bytes_data=None):
        if bytes_data:
            audio_array = np.frombuffer(bytes_data, dtype=np.float32)
            await self.audio_queue.put(audio_array)
            return

        if text_data:
            try:
                data = json.loads(text_data)
                msg_type = data.get("type")

                if msg_type == "config":
                    self.difficulty = data.get("difficulty", "normal")
                    self.gender = data.get("gender", "female")

                    if self.gender == "male":
                        self.tts_voice = "ja-JP-KeitaNeural"
                        print("設定: 男性面接官 (Keita)")
                    else:
                        self.tts_voice = "ja-JP-NanamiNeural"
                        print("設定: 女性面接官 (Nanami)")

                    resume_data = data.get("resume")
                    if resume_data:
                        await self.process_resume_file(resume_data)
                    else:
                        self.resume_text = ""

                    self.system_prompt = get_system_prompt(self.difficulty, self.resume_text, self.gender)
                    
                    self.conversation_history = [] 

                    await self.send_initial_greeting()
                    return

                elif msg_type == "finish":
                    await self.generate_evaluation()
                    return

            except json.JSONDecodeError:
                print("JSON Error")
            except Exception as e:
                print(f"Error in receive: {e}")

    async def process_resume_file(self, resume_data):
        """履歴書ファイルを解析してテキスト化する"""
        try:
            file_name = resume_data.get("fileName", "").lower()
            file_content = resume_data.get("content", "")
            
            if "," in file_content:
                _, encoded = file_content.split(",", 1)
            else:
                encoded = file_content
            
            decoded_bytes = base64.b64decode(encoded)
            file_stream = io.BytesIO(decoded_bytes)
            extracted_text = ""

            if file_name.endswith(".pdf"):
                reader = PdfReader(file_stream)
                for page in reader.pages:
                    txt = page.extract_text()
                    if txt: extracted_text += txt + "\n"
                print("PDF読み込み完了")

            elif file_name.endswith(".docx"):
                doc = docx.Document(file_stream)
                for para in doc.paragraphs:
                    extracted_text += para.text + "\n"
                print("Word読み込み完了")
            
            else:
                try:
                    extracted_text = decoded_bytes.decode("utf-8")
                except:
                    extracted_text = "【読込不可ファイル】"
            
            if len(extracted_text) > 3000:
                 extracted_text = extracted_text[:3000] + "\n...(省略)..."
            
            self.resume_text = extracted_text
            print(f"履歴書抽出文字数: {len(self.resume_text)}")

        except Exception as e:
            print(f"Resume Error: {e}")
            self.resume_text = ""

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
                    self.audio_buffer = np.concatenate((self.audio_buffer, chunk))

            if self.is_speaking and self.silence_chunks >= self.SILENCE_LIMIT:
                await self.transcribe_and_chat()
                self.is_speaking = False
            elif len(self.audio_buffer) > 16000 * 30: # 30秒以上
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
                    language="ja",
                    vad_filter=True,
                ),
            )

            full_text = ""
            hallucinations = ["ご視聴ありがとうございました。", "ご視聴ありがとうございました"]
            
            for segment in segments:
                text = segment.text.strip()
                if text not in hallucinations and len(text) > 0:
                    full_text += text

            if full_text:
                print(f"User: {full_text}")
                await self.send(text_data=json.dumps({"type": "user", "message": full_text}))
                
                await self.generate_ai_response(full_text)

        except Exception as e:
            print(f"Transcription Error: {e}")

    def reset_buffer(self):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.silence_chunks = 0