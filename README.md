Snoonu Menu Extraction & Voice Assistant
========================================

Overview
--------
This project combines menu extraction with a LiveKit voice agent. It can extract
menu items from files, then let you add, edit, and delete menu items using spoken
commands. A web dashboard lets you review extracted data and send the current
menu context to the agent.

Key parts
---------
- Voice agent: `src/voice_assistant.py`
- API server (FastAPI): `src/api/main.py`
- Dashboard UI: `dashboard/index.html`
- Menu extraction (Datalab): `src/extract_menu.py`

Requirements
------------
- Python 3.12+
- A LiveKit project (Cloud or self-hosted)
- API keys for LiveKit, Groq, and Datalab

Setup
-----
1) Create and activate a virtual environment:
   `python -m venv .venv`
   `.\.venv\Scripts\Activate.ps1`

2) Install dependencies:
   `pip install -e .`

3) Create a `.env` file and set these variables (use your own values):
   - `LIVEKIT_URL` (e.g. `wss://your-livekit-url`)
   - `LIVEKIT_API_KEY`
   - `LIVEKIT_API_SECRET`
   - `LIVEKIT_AGENT_NAME` (optional)
   - `GROQ_API_KEY`
   - `GROQ_MODEL` (optional, defaults in code)
   - `DATALAB_API_KEY`
   - `CARTESIA_API_KEY` (if you use Cartesia TTS)
   - `DEEPGRAM_API_KEY` (only if you switch STT to Deepgram)

Run
---
1) Start the API server (serves the dashboard and tokens):
   `uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000`

2) Start the agent:
   `python src/voice_assistant.py dev`

3) Open the dashboard:
   `http://localhost:8000`

4) In the dashboard:
   - Load JSON or run extraction.
   - Click "Connect agent" and "Enable mic".
   - Speak commands to add/edit/delete items.

How it works
------------
- The dashboard sends the current menu as context to the agent (data channel
  and participant metadata).
- The agent asks for missing fields one at a time and confirms before applying
  changes.
- Updates are published back to the dashboard on topic `menu-edit`.
