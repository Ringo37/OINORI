from django.db import models

class InterviewSession(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    score = models.IntegerField(default=0)
    
    feedback_text = models.JSONField(verbose_name="AIフィードバック", default=dict)
    
    conversation_log = models.JSONField(verbose_name="会話履歴", default=list)
    difficulty = models.CharField(max_length=20, default="normal", verbose_name="難易度")
    gender = models.CharField(max_length=20, default="female", verbose_name="面接官性別")   
    def __str__(self):
        return f"面接履歴: {self.created_at.strftime('%Y/%m/%d %H:%M')}"

    class Meta:
        ordering = ['-created_at']