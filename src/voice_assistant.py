import asyncio
import json
import os
import re

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    inference,
    room_io,
)
from livekit.agents.llm import function_tool
from livekit.plugins import groq
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.agents.voice.events import RunContext
from livekit.rtc import LocalParticipant

load_dotenv()


class MyAgent(Agent):
    def __init__(self) -> None:
        self._menu_context: dict | None = None
        self._flow: dict | None = None
        self._forced_reply: str | None = None
        super().__init__(
            instructions=(
                self._build_instructions()
            ),
        )
        self._local_participant: LocalParticipant | None = None

    def _build_instructions(self) -> str:
        menu_json = (
            json.dumps(self._menu_context, ensure_ascii=False)
            if self._menu_context
            else "[]"
        )
        instructions = (
            "You are a formal voice assistant. "
            "Speak clearly and concisely in formal tone. "
            "Respond in English only unless the user explicitly asks for Arabic. "
            "Ask one question at a time. Never ask multiple questions in a single response. "
            "After you ask a question, wait for the user's answer before asking the next one. "
            "Use the available tools to apply menu edits. "
            "Do not use emojis, asterisks, markdown, or special characters. "
            "If a request is unclear, ask a short clarifying question.\n"
            "When the user wants to add a menu item, ask these questions in order, "
            "one per turn, and wait for each answer:\n"
            "1) What will be the Arabic name?\n"
            "2) What will be the English name?\n"
            "3) What will be the Arabic description you want to add?\n"
            "4) What will be the English description you want to add?\n"
            "5) What will be the price you want to add to this item?\n"
            "After collecting the values, summarize the final values for each field "
            "(Arabic name, English name, Arabic description, English description, price) "
            "and ask for confirmation. "
            "Call add_menu_item only after the user confirms. "
            "If the user already provided some fields, skip those questions and ask only the next missing field. "
            "If the user says something like \"what about that\" or gives no value, repeat only the last question.\n"
            "When the user wants to edit an item, first ask: "
            "\"What is the name of the item you want to edit? You can answer in English or Arabic.\" "
            "Then ask which fields to change and collect each value one by one. "
            "Before applying the edit, summarize the final values for each field being updated and ask for confirmation. "
            "Call update_menu_item only after the user confirms and you have the target name and at least one field to change. "
            "If the name is ambiguous or missing, ask for clarification.\n"
            "When the user wants to delete an item, ask for the English name and confirm it, then call delete_menu_item.\n"
            "Use the menu JSON context to reference existing items and to disambiguate names. "
            "Do not invent items that are not in the menu context.\n"
            f"Menu JSON context: {menu_json}"
        )
        if self._menu_context and isinstance(self._menu_context, dict):
            item_count = len(self._menu_context.get("items", []))
            print(f"Agent prompt updated. Menu items: {item_count}")
        else:
            print("Agent prompt updated. Menu context is empty.")
        return instructions

    def _normalize_text(self, text: str) -> str:
        text = re.sub(r"[^\w\s\u0600-\u06ff]", " ", text.lower())
        return re.sub(r"\s+", " ", text.strip())

    def _is_yes(self, text: str) -> bool:
        text = self._normalize_text(text)
        return bool(
            re.search(
                r"\b(yes|y|yeah|yep|correct|confirm|ok|okay|sure|affirmative|نعم|ايوه|أيوه|تمام|موافق)\b",
                text,
            )
        )

    def _is_no(self, text: str) -> bool:
        text = self._normalize_text(text)
        return bool(re.search(r"\b(no|n|nope|cancel|stop|never|لا|مش|غير موافق)\b", text))

    def _has_arabic(self, text: str) -> bool:
        return bool(re.search(r"[\u0600-\u06ff]", text))

    def _normalize_for_match(self, text: str) -> str:
        return self._normalize_text(text)

    def _parse_price(self, text: str) -> float | None:
        match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
        if not match:
            return None
        try:
            return float(match.group(0))
        except Exception:
            return None

    def _detect_add_intent(self, text: str) -> bool:
        text = self._normalize_text(text)
        return "add" in text or "new item" in text or "add item" in text

    def _detect_edit_intent(self, text: str) -> bool:
        text = self._normalize_text(text)
        return "edit" in text or "update" in text or "modify" in text

    def _add_question(self, field: str) -> str:
        questions = {
            "name_ar": "What will be the Arabic name? Please say it in Arabic.",
            "name_en": "What will be the English name?",
            "description_ar": "What will be the Arabic description you want to add? Please say it in Arabic.",
            "description_en": "What will be the English description you want to add?",
            "price": "What will be the price you want to add to this item?",
        }
        return questions.get(field, "Please provide the value.")

    def _clean_value(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"^(the\s+)?(arabic|english)\s+(name|description)\s+(is|it will be|will be)\s+", "", text, flags=re.I)
        text = re.sub(r"^[^a-zA-Z\u0600-\u06ff0-9]+", "", text)
        text = re.sub(r"[.،,!?]+$", "", text).strip()
        return text

    def _find_menu_item_match(self, query: str, *, prefer_lang: str) -> dict | None:
        if not self._menu_context:
            return None
        items = self._menu_context.get("items", [])
        if not isinstance(items, list):
            return None
        query_norm = self._normalize_for_match(query)
        if not query_norm:
            return None
        best = None
        best_score = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            name_ar = self._normalize_for_match(item.get("name_ar") or "")
            name_en = self._normalize_for_match(item.get("name_en") or "")
            candidates = []
            if name_ar:
                candidates.append(("ar", name_ar))
            if name_en:
                candidates.append(("en", name_en))
            for lang, name_norm in candidates:
                if query_norm == name_norm:
                    return item
                score = 0
                if query_norm in name_norm or name_norm in query_norm:
                    score = 2
                else:
                    q_tokens = set(query_norm.split())
                    n_tokens = set(name_norm.split())
                    score = len(q_tokens & n_tokens)
                if lang == prefer_lang:
                    score += 1
                if score > best_score:
                    best = item
                    best_score = score
        return best if best_score > 0 else None

    def _next_missing_add_field(self, fields: dict) -> str | None:
        order = ["name_ar", "name_en", "description_ar", "description_en", "price"]
        for key in order:
            value = fields.get(key)
            if value in (None, ""):
                return key
        return None

    def _field_choice_from_text(self, text: str) -> str | None:
        text = self._normalize_text(text)
        if "arabic name" in text:
            return "name_ar"
        if "english name" in text:
            return "name_en"
        if "arabic description" in text:
            return "description_ar"
        if "english description" in text:
            return "description_en"
        if "price" in text:
            return "price"
        return None

    async def on_user_turn_completed(
        self, turn_ctx, new_message
    ) -> None:
        user_text = new_message.text_content or ""
        normalized = self._normalize_text(user_text)

        if normalized in {"cancel", "stop"}:
            self._flow = None
            self._forced_reply = "Okay, cancelled."
            return

        if self._flow:
            mode = self._flow["mode"]
            if mode == "add":
                if self._flow.get("awaiting_confirm"):
                    if self._is_yes(user_text):
                        fields = self._flow["fields"]
                        payload = {
                            "action": "add",
                            "fields": {
                                "name_ar": fields.get("name_ar") or None,
                                "name_en": fields.get("name_en") or None,
                                "description_ar": fields.get("description_ar") or None,
                                "description_en": fields.get("description_en") or None,
                                "price": fields.get("price"),
                            },
                        }
                        await self._send_menu_edit(payload)
                        self._flow = None
                        self._forced_reply = "Item added successfly thank you!."
                    elif self._is_no(user_text):
                        self._flow = None
                        self._forced_reply = "Okay, I will not add it."
                    else:
                        self._forced_reply = "Please confirm with yes or no."
                    return

                pending = self._flow.get("pending")
                if pending:
                    if pending == "price":
                        price = self._parse_price(user_text)
                        if price is not None:
                            self._flow["fields"]["price"] = price
                    else:
                        if normalized:
                            cleaned = self._clean_value(user_text)
                            self._flow["fields"][pending] = cleaned

                next_field = self._next_missing_add_field(self._flow["fields"])
                if next_field:
                    self._flow["pending"] = next_field
                else:
                    self._flow["pending"] = None
                    self._flow["awaiting_confirm"] = True
                return

            if mode == "edit":
                step = self._flow.get("step")
                if step == "target_name":
                    if normalized:
                        cleaned = self._clean_value(user_text)
                        match_lang = "ar" if self._has_arabic(cleaned) else "en"
                        matched = self._find_menu_item_match(cleaned, prefer_lang=match_lang)
                        if matched:
                            if match_lang == "ar" and matched.get("name_ar"):
                                cleaned = matched.get("name_ar")
                            elif match_lang == "en" and matched.get("name_en"):
                                cleaned = matched.get("name_en")
                            else:
                                cleaned = matched.get("name_en") or matched.get("name_ar") or cleaned
                            self._flow["matched_item"] = matched
                        self._flow["target_name"] = cleaned
                        self._flow["match_lang"] = match_lang
                        self._flow["step"] = "field_choice"
                    return
                if step == "field_choice":
                    choice = self._field_choice_from_text(user_text)
                    if choice:
                        self._flow["pending_field"] = choice
                        self._flow["step"] = "field_value"
                    return
                if step == "field_value":
                    field = self._flow.get("pending_field")
                    if field:
                        if field == "price":
                            price = self._parse_price(user_text)
                            if price is not None:
                                self._flow["fields"][field] = price
                        else:
                            if normalized:
                                self._flow["fields"][field] = user_text.strip()
                    self._flow["step"] = "confirm"
                    return
                if step == "confirm":
                    if self._is_yes(user_text):
                        fields = self._flow["fields"]
                        match_name = self._flow.get("target_name", "")
                        match_lang = self._flow.get("match_lang", "en")
                        match_key = "name_ar" if match_lang == "ar" else "name_en"
                        payload = {"action": "update", "match": {match_key: match_name}, "fields": fields}
                        await self._send_menu_edit(payload)
                        self._flow = None
                        self._forced_reply = "Item updated."
                    elif self._is_no(user_text):
                        self._flow = None
                        self._forced_reply = "Okay, I will not update it."
                    else:
                        self._forced_reply = "Please confirm with yes or no."
                    return

        if self._detect_add_intent(user_text):
            self._flow = {
                "mode": "add",
                "fields": {
                    "name_ar": None,
                    "name_en": None,
                    "description_ar": None,
                    "description_en": None,
                    "price": None,
                },
                "pending": "name_ar",
                "awaiting_confirm": False,
            }
            return

        if self._detect_edit_intent(user_text):
            self._flow = {
                "mode": "edit",
                "target_name": None,
                "fields": {},
                "step": "target_name",
                "pending_field": None,
            }
            return

    async def llm_node(self, chat_ctx, tools, model_settings):
        if self._forced_reply:
            reply = self._forced_reply
            self._forced_reply = None
            yield reply
            return

        if self._flow:
            mode = self._flow["mode"]
            if mode == "add":
                if self._flow.get("awaiting_confirm"):
                    fields = self._flow["fields"]
                    reply = (
                        "Please confirm the details:\n"
                        f"Arabic name: {fields.get('name_ar') or '(not provided)'}\n"
                        f"English name: {fields.get('name_en') or '(not provided)'}\n"
                        f"Arabic description: {fields.get('description_ar') or '(not provided)'}\n"
                        f"English description: {fields.get('description_en') or '(not provided)'}\n"
                        f"Price: {fields.get('price') if fields.get('price') is not None else '(not provided)'}\n"
                        "Do you want to add this item?"
                    )
                    yield reply
                    return

                pending = self._flow.get("pending") or "name_ar"
                yield self._add_question(pending)
                return

            if mode == "edit":
                step = self._flow.get("step")
                if step == "target_name":
                    yield "What is the name of the item you want to edit? You can answer in English or Arabic."
                    return
                if step == "field_choice":
                    yield "Which field do you want to change? Arabic name, English name, Arabic description, English description, or price?"
                    return
                if step == "field_value":
                    field = self._flow.get("pending_field")
                    if field == "name_ar":
                        yield "What is the new Arabic name?"
                    elif field == "name_en":
                        yield "What is the new English name?"
                    elif field == "description_ar":
                        yield "What is the new Arabic description?"
                    elif field == "description_en":
                        yield "What is the new English description?"
                    elif field == "price":
                        yield "What is the new price?"
                    else:
                        yield "Please provide the new value."
                    return
                if step == "confirm":
                    fields = self._flow.get("fields", {})
                    target = self._flow.get("target_name") or "(not provided)"
                    matched = self._flow.get("matched_item") or {}
                    matched_ar = matched.get("name_ar") or ""
                    matched_en = matched.get("name_en") or ""
                    match_line = ""
                    if matched_ar or matched_en:
                        match_line = f"Matched item: {matched_en or '(no English name)'} / {matched_ar or '(no Arabic name)'}\n"
                    reply = (
                        "Please confirm the update:\n"
                        f"Item: {target}\n"
                        f"{match_line}"
                        f"Arabic name: {fields.get('name_ar') or '(unchanged)'}\n"
                        f"English name: {fields.get('name_en') or '(unchanged)'}\n"
                        f"Arabic description: {fields.get('description_ar') or '(unchanged)'}\n"
                        f"English description: {fields.get('description_en') or '(unchanged)'}\n"
                        f"Price: {fields.get('price') if fields.get('price') is not None else '(unchanged)'}\n"
                        "Do you want to apply this update?"
                    )
                    yield reply
                    return

        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            yield chunk

    async def set_menu_context(self, payload: dict) -> None:
        self._menu_context = payload
        await self.update_instructions(self._build_instructions())

    def attach_local_participant(self, participant: LocalParticipant) -> None:
        self._local_participant = participant

    async def on_enter(self):
        self.session.generate_reply(allow_interruptions=False)

    async def _send_menu_edit(self, payload: dict) -> str:
        if not self._local_participant:
            return "Menu edit channel is not available yet."
        await self._local_participant.publish_data(
            json.dumps(payload, ensure_ascii=False),
            topic="menu-edit",
            reliable=True,
        )
        return "Done."

    @function_tool
    async def add_menu_item(
        self,
        context: RunContext,
        name_ar: str = "",
        name_en: str = "",
        description_ar: str = "",
        description_en: str = "",
        price: float | None = None,
    ):
        """Add a new menu item."""
        payload = {
            "action": "add",
            "fields": {
                "name_ar": name_ar or None,
                "name_en": name_en or None,
                "description_ar": description_ar or None,
                "description_en": description_en or None,
                "price": price,
            },
        }
        return await self._send_menu_edit(payload)

    @function_tool
    async def update_menu_item(
        self,
        context: RunContext,
        match_name: str,
        match_lang: str = "auto",
        name_ar: str = "",
        name_en: str = "",
        description_ar: str = "",
        description_en: str = "",
        price: float | None = None,
    ):
        """Update an existing menu item by name."""
        match = {}
        if match_lang == "ar":
            match["name_ar"] = match_name
        elif match_lang == "en":
            match["name_en"] = match_name
        else:
            match["name_ar"] = match_name
            match["name_en"] = match_name

        fields = {}
        if name_ar:
            fields["name_ar"] = name_ar
        if name_en:
            fields["name_en"] = name_en
        if description_ar:
            fields["description_ar"] = description_ar
        if description_en:
            fields["description_en"] = description_en
        if price is not None:
            fields["price"] = price

        payload = {"action": "update", "match": match, "fields": fields}
        return await self._send_menu_edit(payload)

    @function_tool
    async def delete_menu_item(
        self,
        context: RunContext,
        match_name: str,
        match_lang: str = "auto",
    ):
        """Delete a menu item by name."""
        match = {}
        if match_lang == "ar":
            match["name_ar"] = match_name
        elif match_lang == "en":
            match["name_en"] = match_name
        else:
            match["name_ar"] = match_name
            match["name_en"] = match_name
        payload = {"action": "delete", "match": match}
        return await self._send_menu_edit(payload)


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    groq_model = os.getenv("GROQ_MODEL", "llama3-8b-8192")
    agent = MyAgent()

    def _try_apply_metadata(metadata: str) -> None:
        if not metadata:
            return
        try:
            payload = json.loads(metadata)
        except Exception:
            return
        print("Menu context received from metadata.")
        asyncio.create_task(agent.set_menu_context(payload))

    def on_participant_connected(participant) -> None:
        try:
            meta_len = len(participant.metadata or "")
            print(f"Participant connected: {participant.identity}, metadata length: {meta_len}")
        except Exception:
            pass
        _try_apply_metadata(participant.metadata)

    def on_participant_metadata_changed(participant, old_metadata) -> None:
        try:
            meta_len = len(participant.metadata or "")
            print(f"Participant metadata changed: {participant.identity}, metadata length: {meta_len}")
        except Exception:
            pass
        _try_apply_metadata(participant.metadata)

    def on_data_received(packet) -> None:
        try:
            payload_preview = packet.data[:200]
            print(
                f"Data received. Topic: {packet.topic}, Bytes: {len(packet.data)}, "
                f"Preview: {payload_preview!r}"
            )
        except Exception:
            pass
        if packet.topic != "menu-context":
            return
        try:
            text = packet.data.decode("utf-8")
            payload = json.loads(text)
        except Exception:
            return
        asyncio.create_task(agent.set_menu_context(payload))

    ctx.room.on("data_received", on_data_received)
    ctx.room.on("participant_connected", on_participant_connected)
    ctx.room.on("participant_metadata_changed", on_participant_metadata_changed)

    async def apply_existing_participants_metadata():
        try:
            while ctx.room and not ctx.room.isconnected():
                await asyncio.sleep(0.1)
            if not ctx.room or not ctx.room.isconnected():
                return
            for participant in ctx.room.remote_participants.values():
                try:
                    meta_len = len(participant.metadata or "")
                    print(
                        f"Existing participant: {participant.identity}, metadata length: {meta_len}"
                    )
                except Exception:
                    pass
                _try_apply_metadata(participant.metadata)
        except Exception:
            pass

    asyncio.create_task(apply_existing_participants_metadata())

    async def attach_participant_when_ready():
        try:
            while ctx.room and not ctx.room.isconnected():
                await asyncio.sleep(0.1)
            if ctx.room and ctx.room.isconnected():
                agent.attach_local_participant(ctx.room.local_participant)
        except Exception:
            pass

    asyncio.create_task(attach_participant_when_ready())
    session = AgentSession(
        stt=groq.STT(model="whisper-large-v3", detect_language=True),
        llm=groq.LLM(model=groq_model),
        tts=inference.TTS("cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=False,
        resume_false_interruption=False,
        false_interruption_timeout=3.0,
    )

    await session.start(
        agent=agent,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
            ),
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)
