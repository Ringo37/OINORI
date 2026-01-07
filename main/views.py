from django.shortcuts import render, get_object_or_404
from .models import InterviewSession

def index(request):
    return render(request, "main/index.html")

def history_list(request):
    sessions = InterviewSession.objects.all()
    return render(request, "main/history_list.html", {"sessions": sessions})

def history_detail(request, pk):
    session = get_object_or_404(InterviewSession, pk=pk)
    return render(request, "main/history_detail.html", {"session": session})

def interview(request):
    return render(request, 'main/interview.html')

def result(request):
    return render(request, 'main/result.html')