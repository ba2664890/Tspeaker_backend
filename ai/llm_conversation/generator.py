"""
T.Speak — Module LLM : Génération Conversationnelle & Feedback
Génère des réponses adaptées au niveau, des questions de suivi et du feedback encourageant.

Adapté au contexte africain francophone : contexte culturel, langues locales, scénarios pro africains.
"""

import json
import logging
import time
from typing import Optional

import httpx

logger = logging.getLogger("tspeak.ai")

# ─── Prompts système par type de session ─────────────────────────────────────

SYSTEM_PROMPTS = {
    "conversation": """You are T.AI, a friendly English conversation coach specialized in helping 
young Africans from francophone West Africa improve their spoken English.

Your role:
- Ask natural, engaging questions about daily life, culture, career, and African context
- NEVER be condescending or reference accents negatively
- Celebrate African cultural references (Dakar, Lagos, Abidjan, local business context)
- Adapt question complexity to the user's level: {level}
- Keep responses concise (2-3 sentences max)
- End every response with a follow-up question to maintain conversation flow
- Provide feedback in a sandwich format: positive observation → correction → encouragement

Native language context: User's native language is {native_language}. 
Common errors for {native_language} speakers: focus on TH sounds, subject-verb agreement, article usage.""",

    "simulation_pitch": """You are playing THREE investor personas simultaneously during a startup pitch:
1. MARCUS CHEN — Pragmatic VC, focuses on numbers, market size, revenue model. Direct, sometimes skeptical.
2. AMARA DIALLO — Impact investor, focuses on social impact in Africa, sustainability. Warm but demanding.  
3. SOPHIE MARTIN — European angel investor, focuses on team, scalability, technology. Asks about exit strategy.

Rules:
- Ask ONE challenging question at a time, rotating between personas
- Simulate real objections: market too small, competition, team experience
- Score the pitch mentally on: clarity (30%), persuasion (30%), pronunciation (20%), confidence (20%)
- After 5 minutes, provide a detailed investment decision with reasoning
- African startup context: Be realistic about African markets, mobile-first, fintech, agritech

User level: {level}. Be appropriately tough but fair.""",

    "simulation_interview": """You are a professional HR interviewer for an international company with operations in West Africa.

Position: {scenario}
Your style: Professional, warm, STAR-method focused, equity-minded about African candidates.

Key areas to assess:
1. Communication clarity and confidence
2. Problem-solving with concrete examples  
3. Cultural adaptability and leadership
4. English proficiency under pressure

Provide constructive feedback after each answer. Use real interview dynamics.
Do NOT accept vague answers — ask "Can you give me a specific example?"

User level: {level}.""",

    "exercise": """You are a patient English pronunciation coach. 

Current exercise: {scenario}
User native language: {native_language}

Guide the user through structured pronunciation exercises:
1. Model the correct pronunciation clearly
2. Identify specific phoneme issues
3. Provide minimal pairs exercises for problem sounds
4. Use positive reinforcement exclusively

Feedback style: "Great attempt! The 'TH' in 'the' needs a tongue-between-teeth position. Try again?"

User level: {level}.""",

    "level_test": """You are administering a standardized English level assessment for T.Speak.

Evaluate the user across 5 dimensions: fluency, vocabulary, grammar, pronunciation, comprehension.
Ask exactly 5 questions of increasing difficulty (A1 → B2 range).
After each answer, internally score it but don't reveal scores until the end.

Final assessment format:
- Overall level: A1/A2/B1/B2/C1
- Strengths: [list]
- Areas to improve: [list]
- Recommended T.Speak sessions: [specific simulations]""",
}

# ─── Prompts de feedback spécifiques ─────────────────────────────────────────

FEEDBACK_TEMPLATES = {
    "excellent": [
        "Excellent pronunciation! Your '{word}' was crystal clear. 🌟",
        "Wow, great fluency! You spoke like a native on that one.",
        "Outstanding! Your confidence is really showing. Keep it up!",
    ],
    "good": [
        "Good effort! Your main message came through clearly.",
        "Nice work! Just a small note on '{phoneme}' — try placing your tongue differently.",
        "Well done! You're improving session by session.",
    ],
    "needs_work": [
        "Keep practicing! The 'TH' sound is tricky for everyone at first.",
        "Good try! Let's focus on speaking a bit slower — clarity beats speed.",
        "You're on the right track. Remember: confidence first, perfection later!",
    ],
}


class ConversationGenerator:
    """
    Générateur de conversations et feedback pour T.Speak.
    Utilise un LLM externe (OpenAI GPT, Anthropic Claude, etc.) via API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key or self._get_api_key()
        self.client = httpx.Client(timeout=30.0)

        logger.info("ConversationGenerator initialisé: model=%s", model)

    def _get_api_key(self) -> str:
        try:
            from django.conf import settings
            return getattr(settings, "LLM_API_KEY", "")
        except Exception:
            import os
            return os.environ.get("LLM_API_KEY", "")

    def generate_feedback(
        self,
        user_transcription: str,
        ai_question: str,
        pronunciation_score: float,
        fluency_score: float,
        native_language: str = "french",
        session_type: str = "conversation",
        history: list = None,
        user_level: str = "beginner",
        scenario: str = "",
    ) -> dict:
        """
        Génère le feedback et la prochaine question de l'IA.

        Args:
            user_transcription: Ce que l'utilisateur a dit (transcrit par Whisper)
            ai_question: La question que l'IA avait posée
            pronunciation_score: Score prononciation (0-100) de Wav2Vec
            fluency_score: Score fluidité calculé
            native_language: Langue maternelle de l'utilisateur
            session_type: Type de session
            history: Historique des 10 derniers échanges
            user_level: Niveau de l'utilisateur

        Returns:
            {
                "feedback": str,
                "next_question": str,
                "grammar_correction": str,
                "pronunciation_tip": str,
                "encouragement": str,
            }
        """
        start_time = time.monotonic()

        # Sélectionner le prompt système
        system_prompt = self._build_system_prompt(
            session_type, user_level, native_language, scenario
        )

        # Construire les messages avec historique (contexte fenêtre de 10)
        messages = self._build_messages(
            system_prompt=system_prompt,
            user_transcription=user_transcription,
            ai_question=ai_question,
            pronunciation_score=pronunciation_score,
            fluency_score=fluency_score,
            history=history or [],
        )

        try:
            response_text = self._call_llm(messages)
            result = self._parse_response(response_text, pronunciation_score)

            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            logger.info("LLM feedback généré: %.0fms", elapsed_ms)

            return result

        except Exception as e:
            logger.error("Erreur LLM: %s", e, exc_info=True)
            return self._fallback_response(user_transcription, pronunciation_score)

    def _build_system_prompt(
        self, session_type: str, level: str, native_language: str, scenario: str
    ) -> str:
        """Construit le prompt système adapté au contexte."""
        template = SYSTEM_PROMPTS.get(session_type, SYSTEM_PROMPTS["conversation"])
        return template.format(
            level=level,
            native_language=native_language,
            scenario=scenario,
        )

    def _build_messages(
        self,
        system_prompt: str,
        user_transcription: str,
        ai_question: str,
        pronunciation_score: float,
        fluency_score: float,
        history: list,
    ) -> list[dict]:
        """Construit la liste de messages pour l'API LLM."""
        messages = [{"role": "system", "content": system_prompt}]

        # Ajouter l'historique (max 10 échanges)
        for exchange in history[-10:]:
            messages.append({"role": "assistant", "content": exchange.get("ai_question", "")})
            messages.append({"role": "user", "content": exchange.get("transcription", "")})

        # Message current avec contexte scores
        user_message = f"""Previous question: "{ai_question}"

User's response (transcribed): "{user_transcription}"

AI scoring context:
- Pronunciation score: {pronunciation_score:.0f}/100
- Fluency score: {fluency_score:.0f}/100

Please respond in this JSON format:
{{
  "feedback": "Your feedback on the user's response (1-2 sentences, constructive and encouraging)",
  "pronunciation_tip": "Specific pronunciation tip if score < 75, else 'Great pronunciation!'",
  "grammar_correction": "Correct grammar if needed, or 'Grammar looks good!'",
  "next_question": "Your next question or response to continue the conversation",
  "encouragement": "One short motivational phrase in English (5-10 words)"
}}"""

        messages.append({"role": "user", "content": user_message})
        return messages

    def _call_llm(self, messages: list[dict]) -> str:
        """Appelle l'API LLM externe."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 500,
            "temperature": 0.7,
            "response_format": {"type": "json_object"},
        }

        response = self.client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _parse_response(self, response_text: str, pronunciation_score: float) -> dict:
        """Parse la réponse JSON du LLM."""
        try:
            parsed = json.loads(response_text)
            return {
                "feedback": parsed.get("feedback", "Good job! Keep practicing."),
                "pronunciation_tip": parsed.get("pronunciation_tip", ""),
                "grammar_correction": parsed.get("grammar_correction", ""),
                "next_question": parsed.get("next_question", "Can you tell me more?"),
                "encouragement": parsed.get("encouragement", "You're doing great!"),
            }
        except json.JSONDecodeError:
            logger.warning("Réponse LLM non-JSON: %s", response_text[:100])
            return self._fallback_response("", pronunciation_score)

    def _fallback_response(self, transcription: str, pronunciation_score: float) -> dict:
        """Réponse de secours si le LLM échoue."""
        import random
        if pronunciation_score >= 80:
            feedback = random.choice(FEEDBACK_TEMPLATES["excellent"]).format(
                word="that", phoneme="th"
            )
        elif pronunciation_score >= 60:
            feedback = random.choice(FEEDBACK_TEMPLATES["good"]).format(
                word="that", phoneme="th"
            )
        else:
            feedback = random.choice(FEEDBACK_TEMPLATES["needs_work"])

        return {
            "feedback": feedback,
            "pronunciation_tip": "Focus on clarity — speak slowly and clearly.",
            "grammar_correction": "",
            "next_question": "Can you tell me more about that?",
            "encouragement": "Every practice session makes you stronger!",
        }

    def generate_session_report(
        self,
        session_data: dict,
        all_scores: list[dict],
    ) -> dict:
        """
        Génère un rapport complet de session pour les simulations.
        Utilisé notamment pour le rapport post-pitch et post-entretien.
        """
        avg_pronunciation = sum(s.get("pronunciation_score", 0) for s in all_scores) / max(len(all_scores), 1)
        avg_fluency = sum(s.get("fluency_score", 0) for s in all_scores) / max(len(all_scores), 1)

        report_prompt = f"""Generate a comprehensive English coaching report for this T.Speak session.

Session type: {session_data.get('session_type', 'conversation')}
Scenario: {session_data.get('scenario', 'general')}
Duration: {session_data.get('duration_sec', 0)} seconds
Exchanges: {len(all_scores)}
Average pronunciation score: {avg_pronunciation:.1f}/100
Average fluency score: {avg_fluency:.1f}/100

Generate a JSON report with:
{{
  "overall_grade": "A/B/C/D (letter grade)",
  "summary": "2-3 sentence overall assessment",
  "strengths": ["list of 2-3 strengths"],
  "improvements": ["list of 2-3 specific improvement areas"],
  "next_session_recommendation": "specific recommendation",
  "motivational_message": "personalized encouragement for an African learner"
}}"""

        messages = [
            {"role": "system", "content": "You are an expert English language coach generating session reports."},
            {"role": "user", "content": report_prompt},
        ]

        try:
            response_text = self._call_llm(messages)
            return json.loads(response_text)
        except Exception as e:
            logger.error("Erreur génération rapport: %s", e)
            return {
                "overall_grade": "B",
                "summary": "Good session! You showed real progress in your English communication.",
                "strengths": ["Effort and engagement", "Vocabulary variety"],
                "improvements": ["Pronunciation of TH sounds", "Speaking pace"],
                "next_session_recommendation": "Try the Business Pitch simulation next",
                "motivational_message": "Africa's future needs voices like yours. Keep going! 🌍",
            }


# ─── Singleton global ────────────────────────────────────────────────────────

_generator_instance: Optional[ConversationGenerator] = None


def get_generator() -> ConversationGenerator:
    """Retourne l'instance singleton du générateur."""
    global _generator_instance
    if _generator_instance is None:
        from django.conf import settings
        _generator_instance = ConversationGenerator(
            api_key=getattr(settings, "LLM_API_KEY", ""),
            model=getattr(settings, "LLM_MODEL", "gpt-4o-mini"),
        )
    return _generator_instance
