"""
T.Speak — Analyse NLP : Grammaire & Vocabulaire
Évalue la qualité grammaticale et la richesse lexicale des réponses utilisateurs.

Grammaire : spaCy (analyse syntaxique)
Vocabulaire : niveau CEFR (A1 → C2) + richesse lexicale (TTR)
"""

import logging
import re
from typing import Optional

logger = logging.getLogger("tspeak.ai")


class GrammarAnalyzer:
    """
    Analyseur grammatical basé sur spaCy.
    Détecte les erreurs syntaxiques et évalue la complexité des phrases.
    """

    COMMON_ERRORS = {
        # Erreurs fréquentes des francophones en anglais
        "subject_verb_agreement": ["he go", "she go", "it go", "they goes", "we goes"],
        "missing_article": ["i go to school", "i have car"],
        "tense_errors": ["yesterday i go", "tomorrow i went"],
        "double_negative": ["i don't have no", "i haven't no"],
        "wrong_preposition": ["i am agree", "i am interesting"],
    }

    def __init__(self):
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
            except (ImportError, OSError):
                logger.warning("spaCy non disponible — fallback sur analyse basique")
                self._nlp = None
        return self._nlp

    def analyze(self, text: str) -> dict:
        """
        Analyse grammaticale complète.

        Returns:
            {
                "grammar_score": float (0-100),
                "errors": [{"type": str, "description": str, "suggestion": str}],
                "sentence_complexity": float,
                "sentence_count": int,
            }
        """
        if not text or len(text.split()) < 3:
            return {
                "grammar_score": 50.0,
                "errors": [],
                "sentence_complexity": 0.0,
                "sentence_count": 0,
            }

        errors = []

        # Analyse via spaCy si disponible
        if self.nlp:
            errors.extend(self._spacy_analysis(text))
        else:
            errors.extend(self._rule_based_analysis(text))

        # Score (pénalité par erreur)
        error_penalty = min(len(errors) * 8, 50)
        grammar_score = max(50.0, 100.0 - error_penalty)

        # Complexité syntaxique
        sentence_complexity = self._compute_complexity(text)
        # Bonus pour phrases complexes (subordonnées, etc.)
        grammar_score = min(100.0, grammar_score + sentence_complexity * 5)

        return {
            "grammar_score": round(grammar_score, 2),
            "errors": errors[:5],  # Top 5 erreurs à corriger
            "sentence_complexity": round(sentence_complexity, 2),
            "sentence_count": len(re.split(r"[.!?]+", text.strip())),
            "word_count": len(text.split()),
        }

    def _spacy_analysis(self, text: str) -> list[dict]:
        """Analyse syntaxique via spaCy."""
        doc = self.nlp(text)
        errors = []

        for sent in doc.sents:
            tokens = list(sent)

            # Vérifier accord sujet-verbe
            for token in tokens:
                if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                    subj = token
                    verb = token.head
                    if subj.text.lower() in ("he", "she", "it") and \
                       verb.morph.get("Number") == ["Plur"]:
                        errors.append({
                            "type": "subject_verb_agreement",
                            "description": f"'{verb.text}' devrait être conjugué à la 3ème personne du singulier",
                            "suggestion": f"Utilisez '{verb.lemma_}s' ou '{verb.lemma_}es'",
                            "position": verb.idx,
                        })

            # Vérifier les déterminants manquants
            for token in tokens:
                if token.pos_ == "NOUN" and token.dep_ in ("dobj", "pobj"):
                    has_det = any(c.dep_ == "det" for c in token.children)
                    is_proper = token.pos_ == "PROPN"
                    if not has_det and not is_proper and not token.is_stop:
                        errors.append({
                            "type": "missing_article",
                            "description": f"Article manquant avant '{token.text}'",
                            "suggestion": f"Essayez 'a {token.text}', 'an {token.text}' ou 'the {token.text}'",
                            "position": token.idx,
                        })

        return errors

    def _rule_based_analysis(self, text: str) -> list[dict]:
        """Analyse basée sur des règles (fallback sans spaCy)."""
        errors = []
        text_lower = text.lower()

        # Vérifications simples par regex
        patterns = [
            (r"\bhe (go|come|eat|drink|sleep|work)\b", "Accord sujet-verbe manquant",
             "Utilisez 'goes', 'comes', etc. pour he/she/it"),
            (r"\bi am agree\b", "Mauvaise structure", "Utilisez 'I agree'"),
            (r"\bdon't have no\b", "Double négation", "Utilisez 'don't have any'"),
            (r"\byesterday.{0,10}\b(go|come|eat|see)\b", "Erreur de temps",
             "Utilisez le prétérit (went, came, ate, saw)"),
        ]

        for pattern, error_type, suggestion in patterns:
            if re.search(pattern, text_lower):
                errors.append({
                    "type": "grammar_error",
                    "description": error_type,
                    "suggestion": suggestion,
                    "position": 0,
                })

        return errors

    def _compute_complexity(self, text: str) -> float:
        """
        Mesure la complexité syntaxique (0-1).
        Basée sur la longueur moyenne des phrases et la variété de structures.
        """
        sentences = re.split(r"[.!?]+", text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0.0

        # Longueur moyenne des phrases
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences)

        # Indicateurs de complexité
        complex_indicators = [
            r"\bbecause\b", r"\balthough\b", r"\bwhereas\b",
            r"\bhowever\b", r"\btherefore\b", r"\bfurthermore\b",
            r"\bnevertheless\b", r"\bif .+ then\b",
        ]
        text_lower = text.lower()
        complexity_bonus = sum(
            0.15 for p in complex_indicators if re.search(p, text_lower)
        )

        # Normaliser (phrases de 10+ mots = complexes)
        base_complexity = min(avg_length / 15.0, 1.0)
        return float(min(1.0, base_complexity + complexity_bonus))


class VocabularyAnalyzer:
    """
    Analyseur de vocabulaire.
    Évalue la richesse lexicale et le niveau CEFR des mots utilisés.
    """

    # Vocabulaire de base par niveau CEFR (mots fréquents)
    CEFR_VOCABULARY = {
        "A1": {
            "score": 30, "label": "Débutant",
            "examples": ["go", "eat", "have", "be", "see", "like", "want", "good", "big"]
        },
        "A2": {
            "score": 45, "label": "Élémentaire",
            "examples": ["understand", "believe", "discuss", "important", "different"]
        },
        "B1": {
            "score": 60, "label": "Intermédiaire",
            "examples": ["achieve", "strategy", "develop", "experience", "successful"]
        },
        "B2": {
            "score": 75, "label": "Intermédiaire avancé",
            "examples": ["implement", "innovative", "sustainable", "perspective", "negotiate"]
        },
        "C1": {
            "score": 88, "label": "Avancé",
            "examples": ["leverage", "paradigm", "substantiate", "implications", "sophisticated"]
        },
        "C2": {
            "score": 98, "label": "Maîtrise",
            "examples": ["nuanced", "elucidation", "pragmatic", "epitomize", "perspicacious"]
        },
    }

    # Marqueurs de niveau (liste simplifiée)
    ADVANCED_MARKERS = {
        "leverage", "paradigm", "sophisticated", "nuanced", "pragmatic",
        "sustainable", "innovative", "implement", "strategy", "optimize",
        "collaborate", "facilitate", "comprehensive", "substantial",
        "perspective", "stakeholder", "efficiency", "initiative",
    }

    INTERMEDIATE_MARKERS = {
        "achieve", "develop", "experience", "improve", "understand",
        "believe", "important", "different", "possible", "necessary",
        "communicate", "organize", "prepare", "manage", "create",
    }

    def analyze(self, text: str) -> dict:
        """
        Analyse du vocabulaire.

        Returns:
            {
                "vocabulary_score": float (0-100),
                "cefr_level": str,
                "type_token_ratio": float,
                "unique_words": int,
                "advanced_words_found": [str],
                "suggestions": [str],
            }
        """
        if not text:
            return {
                "vocabulary_score": 40.0,
                "cefr_level": "A1",
                "type_token_ratio": 0.0,
                "unique_words": 0,
                "advanced_words_found": [],
                "suggestions": [],
            }

        words = re.findall(r"\b[a-z]+\b", text.lower())
        unique_words = set(words)

        # Type-Token Ratio (richesse lexicale)
        ttr = len(unique_words) / len(words) if words else 0.0

        # Détecter mots avancés
        advanced_found = list(unique_words & self.ADVANCED_MARKERS)
        intermediate_found = list(unique_words & self.INTERMEDIATE_MARKERS)

        # Calculer score vocabulaire
        base_score = 45.0  # A2 par défaut
        base_score += len(advanced_found) * 8
        base_score += len(intermediate_found) * 4
        base_score += (ttr - 0.5) * 30  # Bonus/malus TTR

        vocabulary_score = round(min(100.0, max(20.0, base_score)), 2)

        # Déterminer niveau CEFR
        cefr_level = self._score_to_cefr(vocabulary_score)

        # Suggestions de mots à apprendre
        suggestions = self._generate_suggestions(text, cefr_level)

        return {
            "vocabulary_score": vocabulary_score,
            "cefr_level": cefr_level,
            "type_token_ratio": round(ttr, 3),
            "total_words": len(words),
            "unique_words": len(unique_words),
            "advanced_words_found": advanced_found[:5],
            "suggestions": suggestions,
        }

    def _score_to_cefr(self, score: float) -> str:
        """Convertit un score numérique en niveau CEFR."""
        if score >= 90:
            return "C2"
        elif score >= 80:
            return "C1"
        elif score >= 65:
            return "B2"
        elif score >= 50:
            return "B1"
        elif score >= 38:
            return "A2"
        return "A1"

    def _generate_suggestions(self, text: str, current_level: str) -> list[str]:
        """Génère des suggestions de vocabulaire pour progresser."""
        level_up = {
            "A1": "A2", "A2": "B1", "B1": "B2", "B2": "C1", "C1": "C2", "C2": "C2"
        }
        next_level = level_up.get(current_level, "B1")
        level_data = self.CEFR_VOCABULARY.get(next_level, {})
        examples = level_data.get("examples", [])
        return [f"Try using '{w}' instead of simpler alternatives" for w in examples[:3]]
