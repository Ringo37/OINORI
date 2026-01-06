from django.db import models

class InterviewSession(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    score = models.IntegerField(default=0)  # AIによる採点 (0-100)
    feedback_text = models.TextField()      # AIからのフィードバック
    conversation_log = models.JSONField()   # 会話履歴全体
    
    def __str__(self):
        return f"面接履歴: {self.created_at.strftime('%Y/%m/%d %H:%M')}"