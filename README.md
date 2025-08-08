# Django React LangChain Stream

A real-time chatbot application using Django Channels, React, and Google's Gemini AI.

## Setup Instructions

### 1. Backend Setup (Django)

1. **Activate the virtual environment:**
   ```bash
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create a .env file in the project root:**
   ```bash
   echo "GOOGLE_API_KEY=your_actual_google_api_key_here" > .env
   ```
   
   **Important:** Replace `your_actual_google_api_key_here` with your real Google API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

4. **Run Django migrations:**
   ```bash
   python manage.py migrate
   ```

5. **Start the Django server:**
   ```bash
   python manage.py runserver
   ```

### 2. Frontend Setup (React)

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the React development server:**
   ```bash
   npm run dev
   ```

### 3. Usage

1. Open your browser and go to `http://localhost:5173` (React dev server)
2. The chatbot should connect to the Django backend via WebSocket
3. You should see a green "Connected" indicator in the top-right corner
4. Type a message and press Enter to chat with the AI

## Troubleshooting

### Common Issues:

1. **"GOOGLE_API_KEY not found" error:**
   - Make sure you have created a `.env` file with your Google API key
   - Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

2. **WebSocket connection fails:**
   - Ensure Django server is running on port 8000
   - Check that Django Channels is properly configured
   - Look for any error messages in the Django console

3. **Frontend shows "Disconnected":**
   - Check if Django server is running
   - Verify the WebSocket URL in `frontend/src/App.jsx` matches your Django server
   - Check browser console for WebSocket errors

4. **No response from chatbot:**
   - Verify your Google API key is valid and has sufficient quota
   - Check Django console for any error messages
   - Ensure you have internet connection for API calls

## Project Structure

```
Django_React_Langchain_Stream/
├── Django_React_Langchain_Stream/  # Django project settings
├── langchain_stream/               # Django app with WebSocket consumer
├── frontend/                       # React application
├── venv/                          # Python virtual environment
├── manage.py                      # Django management script
└── .env                          # Environment variables (create this)
```

## Technologies Used

- **Backend:** Django 5.0, Django Channels, Google Generative AI
- **Frontend:** React 18, Vite
- **Communication:** WebSocket (Django Channels)
- **AI:** Google Gemini 1.5 Flash
