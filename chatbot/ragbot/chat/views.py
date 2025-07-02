from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import json
from .embedding import ask_question

@csrf_exempt
def chat_ui(request):
    answer = ""
    question = ""
    if request.method == "POST":
        question = request.POST.get("question", "")
        if question:
            answer = ask_question(question)

    return render(request, "chat.html", {"answer": answer, "question": question})
