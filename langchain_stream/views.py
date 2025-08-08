import json
import os
import asyncio
from dotenv import load_dotenv
from channels.generic.websocket import AsyncWebsocketConsumer
import google.generativeai as genai

# Load environment variables
load_dotenv('.env')

# Configure Gemini with your API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Warning: GOOGLE_API_KEY not found in environment variables")
    api_key = "your_google_api_key_here"  # Replace with your actual API key

genai.configure(api_key=api_key)

# Use Gemini 1.5 Flash (fast + text only)
model = genai.GenerativeModel('gemini-1.5-flash')


class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        data = json.loads(text_data)
        message = data.get("message")

        if not message:
            await self.send(text_data=json.dumps({"error": "No message provided"}))
            return

        try:
            # Send initial response to indicate processing
            await self.send(text_data=json.dumps({
                "event": "on_parser_start",
                "run_id": "response_1",
                "name": "Assistant"
            }))

            # Gemini is sync, run it in a separate thread
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, model.generate_content, message)

            # Send the response text
            await self.send(text_data=json.dumps({
                "event": "on_parser_stream",
                "run_id": "response_1",
                "data": {"chunk": response.text}
            }))

        except Exception as e:
            await self.send(text_data=json.dumps({
                "event": "error",
                "text": str(e)
            }))
