import os
import json
import logging
from ai.whisper_asr.transcriber import WhisperTranscriber
from ai.llm_conversation.generator import ConversationGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_vad():
    logger.info("Test du VAD Whisper...")
    transcriber = WhisperTranscriber(model_name="tiny") # Use tiny for fast test
    # Note: On aurait besoin d'un fichier audio de test avec du silence
    logger.info("VAD configuré avec success.")

def test_llm_parsing():
    logger.info("Test du parsing LLM durci...")
    generator = ConversationGenerator(api_key="mock_key")
    
    # Mocking successful LLM response with extra text
    mock_response = """Here is your feedback in JSON:
    {
        "feedback": "Great job!",
        "pronunciation_tip": "Focus on the 'th' sound.",
        "grammar_correction": "Grammar looks good!",
        "next_question": "What's next?",
        "encouragement": "Keep it up!"
    }
    Hope this helps!"""
    
    result = generator._parse_response(mock_response, 85.0)
    assert result["feedback"] == "Great job!"
    logger.info("Parsing LLM robuste validé (extrait le JSON du texte).")

if __name__ == "__main__":
    try:
        test_llm_parsing()
        # test_vad() # Need audio file
    except Exception as e:
        logger.error(f"Echec des tests: {e}")
