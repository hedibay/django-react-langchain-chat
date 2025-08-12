import json
import os
import asyncio
import base64
import io
import re
from dotenv import load_dotenv
from channels.generic.websocket import AsyncWebsocketConsumer
import google.generativeai as genai
from pypdf import PdfReader

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


def _build_cv_critic_prompt(
    resume_text: str,
    job_text: str | None = None,
    doc_meta: dict | None = None,
) -> str:
    """Create a targeted prompt for resume review with optional job description alignment.

    The model should return concise, high-signal feedback with clear, actionable suggestions.
    """
    header = (
        "You are an expert CV/Resume reviewer and hiring manager. "
        "Provide honest, specific, and actionable critique to improve interview chances. "
        "Be concise and high-signal. Avoid generic advice."
    )

    structure = (
        "Format your response in Markdown using EXACTLY these sections and headings (use '##'):\n\n"
        "## Summary\n- 2 to 4 bullet points highlighting the biggest wins and issues.\n\n"
        "## Scores\n- ATS Fit: <0-10> – one short reason\n- Clarity: <0-10> – one short reason\n- Impact: <0-10> – one short reason\n- Relevance: <0-10> – one short reason\n- Structure/Layout: <0-10> – one short reason\n\n"
        "## Structure & Layout\n- Section ordering (Header, Summary, Skills, Experience, Education, Projects, etc.)\n- One-page vs multi-page recommendation for this profile\n- Bullet style consistency and sentence style (verb-first, tense)\n- White space, scannability, and density (avoid walls of text)\n- ATS considerations (tables, columns, headers/footers, icons)\n- File naming and export guidance (PDF preferred)\n\n"
        "## Top Fixes\n- 3 to 6 bullets. Each: Before → After (concise, action + metric + impact).\n\n"
        "## Experience Rewrites\n- Rewrite 2 to 4 weak bullets using action + metric + impact.\n\n"
        "## Keywords To Add\n- Comma-separated keywords matched to the role/industry.\n\n"
        "## Proposed Structure\n- Provide a concise outline of recommended sections with 1–2 bullets per section.\n"
    )

    if job_text:
        context = (
            "Job Description (for alignment):\n---\n" + job_text.strip() + "\n---\n\n"
        )
        alignment = (
            "Focus on alignment gaps vs the job description. Call out missing keywords and quantifiable outcomes.\n"
        )
    else:
        context = ""
        alignment = (
            "If no job description is provided, optimize for strong general SWE/DS impacts, clarity, and ATS parsing.\n"
        )

    meta_lines: list[str] = []
    if doc_meta:
        if pages := doc_meta.get("pdf_pages"):
            meta_lines.append(f"Detected PDF pages: {pages}")
        if chars := doc_meta.get("char_count"):
            meta_lines.append(f"Approx. characters: {chars}")
    meta_block = ("Document Stats:\n" + "\n".join(f"- {l}" for l in meta_lines) + "\n\n") if meta_lines else ""

    resume = "Resume:\n---\n" + resume_text.strip() + "\n---\n"

    return "\n\n".join([header, alignment, structure, context, meta_block, resume])


def _build_career_mapper_prompt(profile: dict, detail_level: str = "summary") -> str:
    """Prompt to build a personalized career path plan with skill gap analysis and training."""
    detail_instructions = (
        "Be concise (high-signal bullets)." if detail_level == "summary" else
        "Be comprehensive with sub-bullets and examples."
    )
    p = profile or {}
    cur_title = p.get("current_title", "")
    industry = p.get("industry", "")
    skills = p.get("skills", [])
    educ = p.get("education", "")
    goals = p.get("goals", "")

    header = (
        "You are a senior career coach and labor market analyst. "
        "Produce realistic and region-agnostic advice. Include emerging roles."
    )
    structure = (
        "Return Markdown with these sections (use '##'):\n\n"
        "## Profile\n- Current Title: {cur_title}\n- Industry: {industry}\n- Skills: {skills}\n- Education: {educ}\n- Goals: {goals}\n\n"
        "## Suggested Career Paths\n- Short-term (6–12 months): 3–5 roles with 1-line rationale\n- Long-term (12–36 months): 3–5 roles with 1-line rationale\n- Emerging roles relevant to the profile\n\n"
        "## Skill Gap Analysis\n- Table listing: Skill | Current Level (Low/Med/High) | Required Level | Gap\n- Summarize top 5 gaps that unlock most roles\n\n"
        "## Skill Gap Chart (ASCII)\n- Provide an ASCII table or simple bars to visualize gaps\n\n"
        "## Training Recommendations\n- 6–10 items with mix of free/paid courses, certifications, or projects\n- Format: Resource | Provider | Cost | Why it helps\n\n"
        "## Next 30/60/90 Day Plan\n- Concrete weekly milestones\n\n"
        "## Customization\n- Note how to request more detail or focus areas\n"
    ).format(
        cur_title=cur_title or "(unspecified)",
        industry=industry or "(unspecified)",
        skills=", ".join(skills) if isinstance(skills, list) else (skills or "(unspecified)"),
        educ=educ or "(unspecified)",
        goals=goals or "(unspecified)",
    )

    return "\n\n".join([header, detail_instructions, structure])


def _build_career_role_detail_prompt(profile: dict, role: str) -> str:
    p = profile or {}
    header = "You are a senior career coach. Provide a practical role brief."
    details = (
        "Return Markdown with sections (use '##'):\n\n"
        f"## Role\n- Title: {role}\n\n"
        "## Description\n- 3–6 bullet overview\n\n"
        "## Typical Requirements\n- Skills, experience, and education\n\n"
        "## Salary Range (Approximate)\n- Junior | Mid | Senior – global ranges\n\n"
        "## Growth Outlook\n- Market trend and relevance\n\n"
        "## How Your Profile Maps\n- Key matches and gaps relative to user's profile\n"
    )
    return "\n\n".join([header, details])


def _extract_roles_from_markdown_json(text: str) -> list:
    """Best-effort extraction of role names from Markdown list structures."""
    roles = set()
    # Look for bullet lines that seem like roles
    for line in text.splitlines():
        m = re.match(r"^[\-\*]\s+(?:Short\-term|Long\-term)?\s*:?\s*(.+)", line, re.I)
        if m:
            candidate = m.group(1)
            # Strip rationale after dash or colon
            candidate = re.split(r"\s+\-|:\s+", candidate)[0].strip()
            if 2 <= len(candidate) <= 80:
                roles.add(candidate)
    return list(roles)[:12]


def _build_language_tutor_intro_prompt(language: str, mode: str) -> str:
    return (
        "You are a friendly yet precise language tutor. "
        "Confirm the target language and quickly assess the learner's level with 3 short questions.\n\n"
        f"Target Language: {language}\nMode: {mode}\n\n"
        "Return Markdown with sections (use '##'):\n\n"
        "## Warmup\n- Greet and confirm language + mode\n\n"
        "## Level Check\n- Ask 3 short questions or sample sentences to gauge beginner/intermediate/advanced\n\n"
        "## Mini Lesson\n- 3–5 vocab items with emoji + simple grammar tip + 2 example sentences\n\n"
        "## Practice\n- Provide one short exercise (translation, fill-the-blank, or roleplay)\n\n"
        "## Culture Tip\n- One short cultural insight or idiom\n\n"
        "Close by asking them to reply in the target language."
    )


def _build_language_tutor_message_prompt(state: dict, user_message: str) -> str:
    language = state.get("language") or "the target language"
    mode = state.get("mode") or "casual"
    level = state.get("level") or "unknown"
    learned = ", ".join(sorted(state.get("learned_words", []))) or "(none yet)"
    accuracy = 0
    t = max(1, state.get("answers_total", 0))
    accuracy = int(100 * (state.get("answers_correct", 0) / t)) if t else 0
    return (
        "You are a friendly language tutor. Keep messages concise and interactive.\n"
        f"Language: {language}\nMode: {mode}\nEstimated Level: {level}\n"
        f"Known words so far: {learned}\nAccuracy: {accuracy}%\n\n"
        "User said:\n" + user_message + "\n\n"
        "Return Markdown with sections (use '##'):\n\n"
        "## Feedback\n- Correct mistakes with brief explanation and alternatives\n\n"
        "## Practice\n- 1–2 exercises (translation/fill-blank/roleplay). Label with [Exercise].\n\n"
        "## Mini Lesson\n- Tiny vocab/grammar tip with examples\n\n"
        "## Culture Tip\n- One relevant cultural note\n\n"
        "Prompt the learner to reply in the target language."
    )


def _build_language_tutor_check_prompt(state: dict, exercise_context: str, answer: str) -> str:
    language = state.get("language") or "the target language"
    level = state.get("level") or "unknown"
    return (
        "Evaluate the student's answer and provide correction + score. Be supportive and concise.\n\n"
        f"Language: {language}\nEstimated Level: {level}\n\n"
        f"Exercise Context:\n{exercise_context}\n\n"
        f"Student Answer:\n{answer}\n\n"
        "Return Markdown with sections (use '##'):\n\n"
        "## Evaluation\n- Correct/Incorrect with 1–2 reasons\n\n"
        "## Suggested Answer\n- Provide a better version\n\n"
        "## Tip\n- Short tip to avoid this mistake next time\n\n"
        "Mark correctness with a clear token (✅ correct or ❌ incorrect)."
    )


def _update_language_state_from_text(state: dict, text: str) -> None:
    # Naive heuristic to set level based on model text
    if re.search(r"\b(beginner|a1|a2)\b", text, re.I):
        state["level"] = "beginner"
    elif re.search(r"\b(intermediate|b1|b2)\b", text, re.I):
        state["level"] = "intermediate"
    elif re.search(r"\b(advanced|c1|c2)\b", text, re.I):
        state["level"] = "advanced"


def _update_learned_words_from_text(state: dict, text: str) -> None:
    # Very rough extraction of backticked or bolded tokens as words
    words = set()
    words.update(re.findall(r"`([^`]+)`", text))
    words.update(re.findall(r"\*\*([^*]+)\*\*", text))
    # keep short tokens that look like words
    for w in list(words):
        if 2 <= len(w) <= 24 and re.match(r"^[\p{L}\- ]+$", w, re.UNICODE):
            state.setdefault("learned_words", set()).add(w.strip())


def _public_language_state(state: dict) -> dict:
    return {
        "language": state.get("language"),
        "mode": state.get("mode"),
        "level": state.get("level"),
        "learned_words": sorted(list(state.get("learned_words", [])))[:50],
        "answers_total": state.get("answers_total", 0),
        "answers_correct": state.get("answers_correct", 0),
        "accuracy_percent": int(100 * (state.get("answers_correct", 0) / max(1, state.get("answers_total", 1)))),
        "started": state.get("started", False),
    }


def _extract_text_from_pdf_bytes(
    pdf_bytes: bytes, max_pages: int = 10, max_chars: int = 20000
) -> tuple[str, int]:
    """Extract text from a PDF byte stream with simple limits.

    - Reads up to max_pages pages
    - Truncates to max_chars
    """
    with io.BytesIO(pdf_bytes) as file_like:
        reader = PdfReader(file_like)
        total_pages = len(reader.pages)
        pages_to_read = min(total_pages, max_pages)
        texts: list[str] = []
        for idx in range(pages_to_read):
            page = reader.pages[idx]
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if text:
                texts.append(text)
        combined = "\n\n".join(texts)
        if len(combined) > max_chars:
            combined = combined[:max_chars] + "\n\n[...truncated...]"
        return combined, total_pages


class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        # Per-connection state
        self.career_profile = {}
        self.language_state = {
            "language": None,
            "mode": "casual",  # casual | formal | exam
            "level": None,      # beginner | intermediate | advanced
            "learned_words": set(),
            "answers_total": 0,
            "answers_correct": 0,
            "started": False,
        }

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        data = json.loads(text_data)

        # Branch: CV critic agent
        if data.get("type") == "cv_critic":
            resume_text = (data.get("resume") or data.get("message") or "").strip()
            job_text = (data.get("job") or "").strip() or None

            if not resume_text:
                await self.send(text_data=json.dumps({"error": "No resume provided"}))
                return

            try:
                await self.send(text_data=json.dumps({
                    "event": "on_parser_start",
                    "run_id": "cv_response_1",
                    "name": "CV Critic"
                }))

                prompt = _build_cv_critic_prompt(
                    resume_text=resume_text,
                    job_text=job_text,
                    doc_meta={"char_count": len(resume_text)},
                )

                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, model.generate_content, prompt)

                await self.send(text_data=json.dumps({
                    "event": "on_parser_stream",
                    "run_id": "cv_response_1",
                    "data": {"chunk": response.text}
                }))

            except Exception as e:
                await self.send(text_data=json.dumps({
                    "event": "error",
                    "text": str(e)
                }))
            return

        # Branch: CV critic from PDF (base64)
        if data.get("type") == "cv_critic_pdf":
            b64 = (data.get("pdf_base64") or data.get("pdf") or "").strip()
            job_text = (data.get("job") or "").strip() or None

            if not b64:
                await self.send(text_data=json.dumps({"error": "No PDF data provided"}))
                return

            try:
                await self.send(text_data=json.dumps({
                    "event": "on_parser_start",
                    "run_id": "cv_pdf_response_1",
                    "name": "CV Critic"
                }))

                # Accept both plain base64 and data URL
                if b64.startswith("data:"):
                    b64 = b64.split(",", 1)[-1]
                pdf_bytes = base64.b64decode(b64)
                resume_text, total_pages = _extract_text_from_pdf_bytes(pdf_bytes)

                if not resume_text.strip():
                    await self.send(text_data=json.dumps({"error": "Could not extract text from PDF"}))
                    return

                prompt = _build_cv_critic_prompt(
                    resume_text=resume_text,
                    job_text=job_text,
                    doc_meta={
                        "pdf_pages": total_pages,
                        "char_count": len(resume_text),
                    },
                )

                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, model.generate_content, prompt)

                await self.send(text_data=json.dumps({
                    "event": "on_parser_stream",
                    "run_id": "cv_pdf_response_1",
                    "data": {"chunk": response.text}
                }))

            except Exception as e:
                await self.send(text_data=json.dumps({
                    "event": "error",
                    "text": str(e)
                }))
            return

        # Branch: Career Path Mapper - generate plan
        if data.get("type") == "career_mapper":
            profile = data.get("profile") or {}
            detail = (data.get("detail") or "summary").lower()
            # Persist minimal profile for follow-ups
            self.career_profile.update(profile)

            try:
                await self.send(text_data=json.dumps({
                    "event": "on_parser_start",
                    "run_id": "career_plan_1",
                    "name": "Career Mapper"
                }))

                prompt = _build_career_mapper_prompt(profile=self.career_profile, detail_level=detail)
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, model.generate_content, prompt)

                text = response.text
                await self.send(text_data=json.dumps({
                    "event": "on_parser_stream",
                    "run_id": "career_plan_1",
                    "data": {"chunk": text}
                }))

                roles = _extract_roles_from_markdown_json(text)
                if roles:
                    await self.send(text_data=json.dumps({
                        "event": "career_roles",
                        "roles": roles
                    }))
            except Exception as e:
                await self.send(text_data=json.dumps({
                    "event": "error",
                    "text": str(e)
                }))
            return

        # Branch: Career Path Mapper - role details
        if data.get("type") == "career_mapper_role_detail":
            role = (data.get("role") or "").strip()
            if not role:
                await self.send(text_data=json.dumps({"error": "No role provided"}))
                return
            try:
                await self.send(text_data=json.dumps({
                    "event": "on_parser_start",
                    "run_id": "career_role_detail_1",
                    "name": "Career Mapper"
                }))

                prompt = _build_career_role_detail_prompt(profile=self.career_profile, role=role)
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, model.generate_content, prompt)

                await self.send(text_data=json.dumps({
                    "event": "on_parser_stream",
                    "run_id": "career_role_detail_1",
                    "data": {"chunk": response.text}
                }))
            except Exception as e:
                await self.send(text_data=json.dumps({
                    "event": "error",
                    "text": str(e)
                }))
            return

        # Branch: Language Tutor - start session
        if data.get("type") == "language_tutor_start":
            lang = (data.get("language") or "").strip() or None
            mode = (data.get("lesson_mode") or data.get("mode") or "casual").strip().lower()
            if not lang:
                await self.send(text_data=json.dumps({"error": "No language provided"}))
                return
            self.language_state.update({
                "language": lang,
                "mode": mode if mode in ("casual", "formal", "exam") else "casual",
                "started": True,
            })
            try:
                await self.send(text_data=json.dumps({
                    "event": "on_parser_start",
                    "run_id": "lang_tutor_intro_1",
                    "name": "Language Tutor"
                }))
                prompt = _build_language_tutor_intro_prompt(lang, self.language_state["mode"])
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, model.generate_content, prompt)
                text = response.text
                _update_language_state_from_text(self.language_state, text)
                _update_learned_words_from_text(self.language_state, text)
                await self.send(text_data=json.dumps({
                    "event": "on_parser_stream",
                    "run_id": "lang_tutor_intro_1",
                    "data": {"chunk": text}
                }))
                await self.send(text_data=json.dumps({
                    "event": "language_tutor_state",
                    "state": _public_language_state(self.language_state),
                }))
            except Exception as e:
                await self.send(text_data=json.dumps({
                    "event": "error",
                    "text": str(e)
                }))
            return

        # Branch: Language Tutor - general message (conversation/lessons)
        if data.get("type") == "language_tutor_message":
            message = (data.get("message") or "").strip()
            if not self.language_state.get("started"):
                await self.send(text_data=json.dumps({"error": "Start the tutor first"}))
                return
            if not message:
                await self.send(text_data=json.dumps({"error": "No message provided"}))
                return
            try:
                await self.send(text_data=json.dumps({
                    "event": "on_parser_start",
                    "run_id": "lang_tutor_msg_1",
                    "name": "Language Tutor"
                }))
                prompt = _build_language_tutor_message_prompt(self.language_state, user_message=message)
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, model.generate_content, prompt)
                text = response.text
                _update_language_state_from_text(self.language_state, text)
                _update_learned_words_from_text(self.language_state, text)
                await self.send(text_data=json.dumps({
                    "event": "on_parser_stream",
                    "run_id": "lang_tutor_msg_1",
                    "data": {"chunk": text}
                }))
                await self.send(text_data=json.dumps({
                    "event": "language_tutor_state",
                    "state": _public_language_state(self.language_state),
                }))
            except Exception as e:
                await self.send(text_data=json.dumps({
                    "event": "error",
                    "text": str(e)
                }))
            return

        # Branch: Language Tutor - submit exercise answer
        if data.get("type") == "language_tutor_submit":
            answer = (data.get("answer") or data.get("message") or "").strip()
            exercise_context = (data.get("exercise") or data.get("context") or "").strip()
            if not self.language_state.get("started"):
                await self.send(text_data=json.dumps({"error": "Start the tutor first"}))
                return
            if not answer:
                await self.send(text_data=json.dumps({"error": "No answer provided"}))
                return
            try:
                await self.send(text_data=json.dumps({
                    "event": "on_parser_start",
                    "run_id": "lang_tutor_check_1",
                    "name": "Language Tutor"
                }))
                prompt = _build_language_tutor_check_prompt(self.language_state, exercise_context, answer)
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, model.generate_content, prompt)
                text = response.text
                # naive correctness heuristic
                correct = bool(re.search(r"\b(correct|✅)\b", text, re.I)) and not re.search(r"\b(incorrect|❌)\b", text, re.I)
                self.language_state["answers_total"] += 1
                if correct:
                    self.language_state["answers_correct"] += 1
                _update_learned_words_from_text(self.language_state, text)
                await self.send(text_data=json.dumps({
                    "event": "on_parser_stream",
                    "run_id": "lang_tutor_check_1",
                    "data": {"chunk": text}
                }))
                await self.send(text_data=json.dumps({
                    "event": "language_tutor_state",
                    "state": _public_language_state(self.language_state),
                }))
            except Exception as e:
                await self.send(text_data=json.dumps({
                    "event": "error",
                    "text": str(e)
                }))
            return

        # Default: general chat
        # Allow explicit type='chat' or no type
        if data.get("type") not in (None, "", "chat"):
            await self.send(text_data=json.dumps({"error": f"Unknown type: {data.get('type')}"}))
            return

        message = data.get("message")

        if not message:
            await self.send(text_data=json.dumps({"error": "No message provided"}))
            return

        try:
            await self.send(text_data=json.dumps({
                "event": "on_parser_start",
                "run_id": "response_1",
                "name": "Assistant"
            }))

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, model.generate_content, message)

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
