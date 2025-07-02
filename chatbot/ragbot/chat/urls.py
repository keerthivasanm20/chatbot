from chat.views import chat_ui
from django.contrib import admin
from django.urls import path

urlpatterns = [
    path("", chat_ui, name="chat_ui"),
    path("chat/", chat_ui, name="chat_view"),
]