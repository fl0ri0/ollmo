"""Shared request phase-graph helpers for Ghost planning, execution resolving, and frames."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any, Optional

from helpers.model_capabilities import (
    CAPABILITY_CHAT,
    CAPABILITY_IMAGE_GENERATION,
    CAPABILITY_SPEECH_TO_TEXT,
    CAPABILITY_TEXT_TO_SPEECH,
    CAPABILITY_VISION_ANALYSIS,
    normalize_capability,
)
from ollmo_core.inference import (
    _json_text_artifact_intent,
    _text_artifact_format_match_is_negated,
    detect_text_artifact_requests,
    normalize_text_artifact_extension,
)
from ollmo_g.intent import (
    analyze_prompt_intent,
    intent_span_is_literal_payload,
    mask_intent_literal_payloads,
    materialization_negation_match_is_artifact_fulfillment_only,
    materialization_negation_match_is_output_contrast,
    normalize_intent_text,
    visual_action_is_negated,
)
from ollmo_g.intent_obligations import (
    apply_intent_obligation_dependency_edges,
    build_intent_obligation_ledger,
    required_intent_obligations,
    summarize_required_intent_obligations,
)
from ollmo_g.request_ir import build_request_ir
from ollmo_g.request_meta import extract_request_meta
from ollmo_services.responses import extract_responses_current_turn_prompt
from ollmo_services.tts_source import resolve_explicit_tts_source

REQUEST_PHASE_GRAPH_VERSION = 3
_GHOST_PREPARE_FIRST_MATERIALIZATION_CAPABILITIES = {
    CAPABILITY_IMAGE_GENERATION,
    CAPABILITY_SPEECH_TO_TEXT,
    CAPABILITY_TEXT_TO_SPEECH,
}
_STT_RESULT_FEEDS_AUDIO_RE = re.compile(
    r'\b('
    r'daraus|davon|hieraus|'
    r'aus\s+(?:der\s+)?(?:transkription|transkript|übersetzung|uebersetzung|zusammenfassung)|'
    r'(?:from|using)\s+(?:that|it|the\s+(?:transcript|transcription|translation|summary))|'
    r'read\s+(?:that|it|the\s+(?:transcript|transcription|translation|summary))\s+aloud|'
    r'audio\s+(?:from|using|of)\s+(?:that|it|the\s+(?:transcript|transcription|translation|summary))'
    r')\b',
    re.IGNORECASE,
)
_CANDIDATE_CONTRACT_STATES = {
    'candidate',
    'reserved',
    'possible',
    'draft',
    'optional',
    'not_promoted',
    'not-promoted',
    'unpromoted',
    'discarded',
    'rejected',
}
_PROMOTED_CONTRACT_STATES = {
    'promoted',
    'promoted_to_obligation',
    'promotion_accepted',
}
_ASSISTANT_OUTPUT_STRONG_FOLLOW_UP_RE = re.compile(
    r'\b('
    r'phase\s*\d+|status\s*[:=\-]?\s*(?:pending|queued|scheduled)|pending|queued|queuing|queue|trigger|'
    r'branch|late\s*fill|follow[-\s]?up|continuation|'
    r'action|action[_\-\s]?input|tool[_\-\s]?call|function[_\-\s]?call|'
    r'i\s+(?:will|am\s+going\s+to|shall)\s+(?:now\s+)?(?:generate|create|render|make|read|speak)|'
    r'i[\'’]?ll\s+(?:now\s+)?(?:generate|create|render|make|read|speak)|'
    r'ich\s+(?:werde|starte)|wird\s+jetzt|jetzt\s+(?:starte|generiere|lese)'
    r')\b',
    re.IGNORECASE,
)
_ASSISTANT_OUTPUT_TTS_CLAIM_RE = re.compile(
    r'\b('
    r'text[_\-\s]?to[_\-\s]?speech|tts|audio\s+generation|audio\s+branch|voice\s+clip|'
    r'spoken\s+version|read(?:\s+\w+){0,5}\s+(?:aloud|out\s+loud)|'
    r'vorlesen|lies(?:\s+\w+){0,5}\s+vor|lese(?:\s+\w+){0,5}\s+vor|audio(?:\s|-)?datei'
    r')\b',
    re.IGNORECASE,
)
_ASSISTANT_OUTPUT_IMAGE_CLAIM_RE = re.compile(
    r'\b('
    r'image[_\-\s]?generation|image\s+branch|image\s+artifact|generate(?:\s+\w+){0,4}\s+image|'
    r'picture\s+generation|render(?:\s+\w+){0,4}\s+image|'
    r'bild(?:er)?(?:\s+\w+){0,5}\s+generier|generier(?:e|en)?(?:\s+\w+){0,5}\s+bild'
    r')\b',
    re.IGNORECASE,
)
_ASSISTANT_OUTPUT_HYPOTHETICAL_RE = re.compile(
    r'\b('
    r'for\s+example|hypothetical|if\s+(?:the|a|someone|you|i)\s+(?:ask|asks|asked)|'
    r'zum\s+beispiel|beispiel|wenn\s+(?:der|ein|ich|du)|'
    r'could\s+(?:then|also)?|would\s+(?:then|also)?|konnte|koennte|wurde|wuerde'
    r')\b',
    re.IGNORECASE,
)
_CURRENT_INPUT_AUDIO_REFERENCE_RE = re.compile(
    r'\b('
    r'audio(?:\s|-)?datei|audiodatei|audio\s+file|audiofile|aufnahme|recording|voice\s+note|'
    r'audio|sound|clip|sprachversion|gesprochene(?:n|r|s)?\s+version|spoken\s+version|stimme|tonlage|'
    r'datei|file|upload|hochgeladen|angehangt|angehaengt|angefugt|angefuegt|attachment|attachement|attached'
    r')\b',
    re.IGNORECASE,
)
_AUDIO_ARTIFACT_TYPES = {'audio', 'wav', 'mp3', 'm4a', 'flac', 'aac', 'ogg', 'opus'}
_TEXT_SOURCE_FENCED_BLOCK_RE = re.compile(
    r'```(?P<lang>[A-Za-z0-9_+.-]*)(?:[^\n`]*)?\n(?P<body>.*?)(?:\n```|```)',
    re.DOTALL,
)
_TEXT_SOURCE_FILENAME_RE = re.compile(
    r'(?im)^\s*(?:<!--\s*|/\*\s*|//\s*|#\s*)?'
    r'(?P<name>[A-Za-z0-9][A-Za-z0-9._-]{0,80})\.(?P<ext>[A-Za-z0-9]{1,12})\b'
)
_TEXT_SOURCE_LANGUAGE_EXTENSIONS = {
    'bash': 'sh',
    'cjs': 'js',
    'css': 'css',
    'htm': 'html',
    'html': 'html',
    'javascript': 'js',
    'js': 'js',
    'json': 'json',
    'jsx': 'jsx',
    'markdown': 'md',
    'md': 'md',
    'mjs': 'js',
    'py': 'py',
    'python': 'py',
    'sh': 'sh',
    'shell': 'sh',
    'svg': 'svg',
    'text': 'txt',
    'ts': 'ts',
    'tsx': 'tsx',
    'txt': 'txt',
    'xml': 'xml',
    'yaml': 'yaml',
    'yml': 'yaml',
}
_TEXT_SOURCE_EDIT_CUE_RE = re.compile(
    r'\b('
    r'change|modify|update|edit|revise|alter|replace|restyle|'
    r'fix|correct|repair|debug|broken|link|bind|embed|insert|include|'
    r'aendere|ändere|veraendere|verändere|anpassen|passe|ersetze|'
    r'fuege|füge|hinzufuegen|hinzufügen|entferne|loesche|lösche|'
    r'verlinke|verlinken|verknuepfe|verknüpfe|einbinden|binde|'
    r'farbe|farbschema|schrift|design|layout|struktur|updated|aktualisiert'
    r')\b',
    re.IGNORECASE,
)
_JSON_SOURCE_EDIT_TARGET_RE = re.compile(
    r'(?:'
    r'\b(?:change|modify|update|edit|revise|alter|replace|fix|correct|repair|'
    r'aendere|ändere|veraendere|verändere|ersetze|korrigiere|repariere)\w*\b\s+'
    r'(?:(?:this|that|the|selected|current|existing|attached|uploaded|provided|'
    r'dies(?:e|es|en)|das|die|ausgewahlte|ausgewählte|aktuelle|bestehende|'
    r'angehangte|angehängte|hochgeladene)\s+){0,3}'
    r'json(?:\s+object|\s+objekt|\s+file|\s+datei)?\b|'
    r'\b(?:change|modify|update|edit|revise|alter|replace|fix|correct|repair|'
    r'aendere|ändere|veraendere|verändere|ersetze|korrigiere|repariere)\w*\b\s+'
    r'(?:(?:the|this|selected|current|die|das|den|diese|ausgewahlte|ausgewählte|aktuelle)\s+)?'
    r'(?:field|fields|key|keys|value|values|property|properties|feld|felder|schlussel|schlüssel|wert|werte|status)\b'
    r'[^.;!?]{0,40}\b(?:in|inside|within|of|im|in\s+dem|in\s+der|des|der)\s+'
    r'(?:(?:this|that|the|selected|current|existing|dies(?:e|es|en)|das|die|ausgewahlte|ausgewählte|aktuelle|bestehende)\s+){0,3}'
    r'json(?:\s+object|\s+objekt|\s+file|\s+datei)?\b|'
    r'\b(?:in|inside|within|im|in\s+dem|in\s+der)\s+'
    r'(?:(?:this|that|the|selected|current|existing|dies(?:e|es|en)|das|die|ausgewahlte|ausgewählte|aktuelle|bestehende)\s+){0,3}'
    r'json(?:\s+object|\s+objekt|\s+file|\s+datei)?\b'
    r'[^.;!?]{0,24}\b(?:change|modify|update|edit|revise|alter|replace|fix|correct|repair|'
    r'aendere|ändere|veraendere|verändere|ersetze|korrigiere|repariere)\w*\b|'
    r'\b(?:this|that|selected|current|existing|attached|uploaded|provided|'
    r'dies(?:e|es|en)|ausgewahlte|ausgewählte|aktuelle|bestehende|angehangte|angehängte|hochgeladene)\s+'
    r'json(?:\s+object|\s+objekt|\s+file|\s+datei)?\b\s*[:,]\s*'
    r'(?:change|modify|update|edit|revise|alter|replace|fix|correct|repair|'
    r'aendere|ändere|veraendere|verändere|ersetze|korrigiere|repariere)\w*\b'
    r')',
    re.IGNORECASE,
)
_TEXT_SOURCE_LINKED_CSS_CUE_RE = re.compile(
    r'\b(?:linked|verlinkt(?:e|en|er|es)?|eingebunden(?:e|en|er|es)?|zugehoerig(?:e|en|er|es)?|zugehörig(?:e|en|er|es)?)\b'
    r'[\s\S]{0,80}\bcss\b|'
    r'\bcss\b[\s\S]{0,80}\b(?:linked|verlinkt(?:e|en|er|es)?|eingebunden(?:e|en|er|es)?|zugehoerig(?:e|en|er|es)?|zugehörig(?:e|en|er|es)?)\b',
    re.IGNORECASE,
)
_TEXT_SOURCE_HTML_STYLESHEET_LINK_RE = re.compile(
    r'<link\b(?P<attrs>[^>]*\bhref\s*=\s*(?P<quote>["\'])(?P<href>[^"\']+\.css(?:[?#][^"\']*)?)(?P=quote)[^>]*)>',
    re.IGNORECASE,
)
_GENERATE_IMAGE_ACTION_RE = re.compile(
    r'\b('
    r'generate|create|render|show|make|generier|generiere|generieren|erzeuge|erstelle|erstellen|'
    r'zeige|mach|mache|machen'
    r')\b[^.;!?]{0,100}?\b(?:image|images|picture|pictures|photo(?:s)?|foto(?:s)?|illustration(?:s)?|posterbild(?:er)?|bildidee(?:n)?|bildkandidat(?:en)?|bildvariante(?:n)?|bildversion(?:en)?|bild|bilder)\b|'
    r'\b(?:image|images|picture|pictures|photo(?:s)?|foto(?:s)?|illustration(?:s)?|posterbild(?:er)?|bildidee(?:n)?|bildkandidat(?:en)?|bildvariante(?:n)?|bildversion(?:en)?|bild|bilder)\b[^.;!?]{0,100}?\b('
    r'generate|create|render|show|make|generiere|generieren|erzeuge|erstelle|erstellen|zeige|mach|mache|machen'
    r')\b|'
    r'\b(?:image\s+ideas?|image\s+candidates?|bildidee(?:n)?|bildkandidat(?:en)?)\b[^.;!?]{0,140}\b('
    r'generate|create|render|show|make|generier|generiere|generieren|erzeuge|erstelle|erstellen|zeige'
    r')\b[^.;!?]{0,80}\b(?:only\s+the\s+)?(?:first|second|third|fourth|fifth|sixth|erste|zweite|dritte|vierte|fuenfte|fünfte|sechste)\b|'
    r'\b('
    r'generate|create|render|show|make|generier|generiere|generieren|erzeuge|erstelle|erstellen|zeige'
    r')\b[^.;!?]{0,80}\b(?:only\s+the\s+)?(?:first|second|third|fourth|fifth|sixth|erste|zweite|dritte|vierte|fuenfte|fünfte|sechste)\b'
    r'[^.;!?]{0,140}\b(?:image\s+ideas?|image\s+candidates?|bildidee(?:n)?|bildkandidat(?:en)?)\b'
    r'|'
    r'\b(?:also|additionally|zusatzlich|zusaetzlich|zusätzlich)\b[^.;!?]{0,64}\b'
    r'(?:an?|ein(?:e|en|es)?)?\s*(?:posterbild(?:er)?|poster|bild|bilder|image|images|picture|pictures|photo(?:s)?|foto(?:s)?|illustration(?:s)?)\b',
    re.IGNORECASE,
)
_GENERATE_AUDIO_ACTION_RE = re.compile(
    r'\b('
    r'read|speak|narrate|voice|lies|lese|sprich|erzähle|erzaehle'
    r')\b[^.;!?]{0,100}?\b(?:aloud|audio|spoken|vor|mp3|wav)\b|'
    r'\b(?:generate|create|make|generier|generiere|erzeuge|mach|mache|machen)\b[^.;!?]{0,64}?\b(?:audio(?:s)?|audio file|audio ad|audiofassung(?:en)?|audioversion(?:en)?|voice clip|spoken|mp3|wav)\b|'
    r'\b(?:turn|convert|transform|wandle|wandel|verwandle|mach|mache|machen)\b[^.;!?]{0,64}?\b(?:it|this|that|story|text|script|slogan|tagline|reply|daraus|davon|ihn|sie|es)?\b[^.;!?]{0,32}?\b(?:into|to|as|in|als)\b[^.;!?]{0,32}?\b(?:audio|speech|spoken|mp3|wav)\b|'
    r'\b(?:replace|ersetz(?:e|en|t|st)?)\b[^.;!?]{0,120}\b(?:audio[\s-]?branch|audiozweig(?:e)?)\b'
    r'[^.;!?]{0,80}\b(?:with|by|durch)\b[^.;!?]{0,80}\b(?:audio[\s-]?(?:version(?:s)?|variant(?:s)?|fassung(?:en)?))\b|'
    r'\b(?:audio generation|audiogenerierung)\b[^.;!?]{0,80}\b(?:start|starts|run|runs|starte|läuft|laeuft)\b',
    re.IGNORECASE,
)
_DIRECT_IMAGE_DESCRIPTION_PREFIX_RE = re.compile(
    r'(?i)^\s*(?:of|showing|depicting|featuring|with)\s+'
)
_DIRECT_MEDIA_DEICTIC_PAYLOAD_RE = re.compile(
    r'^\s*(?:it|this|that|these|those|them|him|her|daraus|davon|dies(?:e|er|es|en|em)?|'
    r'das|sie|es)(?:\b|$)|'
    r'^\s*(?:(?:the\s+)?(?:previous|prepared|generated|written|described|referenced)\s+)?'
    r'(?:story|text|description|prompt|reply|response|content)\b|'
    r'^\s*the\s+(?:story|text|description|prompt|reply|response|content)\s+(?:above|before)\b',
    re.IGNORECASE,
)
_GERMAN_ORIGINAL_AUDIO_VARIANT_RE = re.compile(
    r'\b(?:ursprunglich\w*|original\w*)\b[^.;!?]{0,80}\bdeutsch\w*\b'
    r'[^.;!?]{0,64}\b(?:erzahlung|narration|text|version)\b|'
    r'\bdeutsch\w*\b[^.;!?]{0,64}\b(?:ursprunglich\w*|original\w*)\b'
    r'[^.;!?]{0,64}\b(?:erzahlung|narration|text|version)\b',
    re.IGNORECASE,
)
_ENGLISH_TRANSLATION_AUDIO_VARIANT_RE = re.compile(
    r'\b(?:getreu\w*\s+)?englisch\w*\b[^.;!?]{0,64}\b(?:ubersetzung|translation)\b|'
    r'\b(?:faithful\w*\s+)?english\b[^.;!?]{0,64}\btranslation\b',
    re.IGNORECASE,
)
_AUDIO_VARIANT_LANGUAGE_MENTION_RE = re.compile(
    r'\b(?P<english>english|englisch\w*)\b|\b(?P<german>german|deutsch\w*)\b',
    re.IGNORECASE,
)
_POST_IMAGE_TEXT_RE = re.compile(
    r'\b(?:after|afterwards|then|danach|anschliessend|anschließend|nach)\b.{0,80}\b('
    r'image|images|picture|pictures|bild(?:er|es|ern|e)?|bildgenerierung|image generation|generated images?|'
    r'erzeugte(?:n|s|m|r)? bild(?:er|es|ern|e)?|generierte(?:n|s|m|r)? bild(?:er|es|ern|e)?'
    r')\b.{0,120}\b('
    r'write|describe|caption|summari[sz]e|confirm|explain|compare|analy[sz]e|analyse|analysier|inspect|examine|'
    r'visible\s+details|sichtbare\s+details|sichtbaren\s+details|'
    r'schreib|schreibe|beschreib|beschreibe|'
    r'bestaetig|bestätig|erklaer|erklar|erklär|name|list|enumerate|nenn|nenne|auflist|aufzähl|aufzaehl'
    r')\b|'
    r'\b('
    r'write|describe|caption|summari[sz]e|confirm|explain|compare|analy[sz]e|analyse|analysier|inspect|examine|'
    r'visible\s+details|sichtbare\s+details|sichtbaren\s+details|'
    r'schreib|schreibe|beschreib|beschreibe|'
    r'bestaetig|bestätig|erklaer|erklar|erklär|name|list|enumerate|nenn|nenne|auflist|aufzähl|aufzaehl'
    r')\b.{0,120}\b(?:after|afterwards|then|danach|anschliessend|anschließend|nach)\b.{0,80}\b('
    r'image|images|picture|pictures|bild(?:er|es|ern|e)?|bildgenerierung|image generation|generated images?|'
    r'erzeugte(?:n|s|m|r)? bild(?:er|es|ern|e)?|generierte(?:n|s|m|r)? bild(?:er|es|ern|e)?'
    r')\b|'
    r'\b(?:after|afterwards|then|danach|anschliessend|anschließend|nach)\b.{0,80}\b('
    r'compare|summari[sz]e|caption|describe|write|explain|confirm|analy[sz]e|analyse|analysier|inspect|examine|'
    r'visible\s+details|sichtbare\s+details|sichtbaren\s+details'
    r')\b.{0,120}\b('
    r'image|images|picture|pictures|bild(?:er|es|ern|e)?|bildgenerierung|image generation|generated images?|'
    r'erzeugte(?:n|s|m|r)? bild(?:er|es|ern|e)?|generierte(?:n|s|m|r)? bild(?:er|es|ern|e)?'
    r')\b',
    re.IGNORECASE,
)
_POST_AUDIO_TEXT_RE = re.compile(
    r'\b(?:after|afterwards|then|danach|anschliessend|anschließend|nach)\b.{0,80}\b('
    r'audio(?:s)?|audiofassung(?:en)?|audioversion(?:en)?|audiogenerierung|audio generation|spoken text|spoken version|'
    r'gesprochen(?:e|en|er|es)?(?:\s+text|\s+version)?'
    r')\b.{0,120}\b('
    r'write|confirm|summari[sz]e|transcribe|transkribier\w*|say|give|check|verify|evaluate|review|assess|'
    r'schreib|schreibe|gib|gebe|bestaetig|bestätig|fass|notier|pruef(?:e|st|t|en)?|pruf(?:e|st|t|en)?|prüf(?:e|st|t|en)?|bewert|kontrollier|markier'
    r')\b|'
    r'\b('
    r'write|confirm|summari[sz]e|transcribe|transkribier\w*|say|give|check|verify|evaluate|review|assess|'
    r'schreib|schreibe|gib|gebe|bestaetig|bestätig|fass|notier|pruef(?:e|st|t|en)?|pruf(?:e|st|t|en)?|prüf(?:e|st|t|en)?|bewert|kontrollier|markier'
    r')\b.{0,120}\b(?:after|nach)\b.{0,80}\b('
    r'audio(?:s)?|audiofassung(?:en)?|audioversion(?:en)?|audiogenerierung|audio generation|spoken text|spoken version|'
    r'gesprochen(?:e|en|er|es)?(?:\s+text|\s+version)?'
    r')\b|'
    r'\b('
    r'audio(?:s)?|audiofassung(?:en)?|audioversion(?:en)?|audiogenerierung|audio generation|spoken text|spoken version|'
    r'gesprochen(?:e|en|er|es)?(?:\s+text|\s+version)?'
    r')\b.{0,120}\b(?:then|afterwards|danach|anschliessend|anschließend|und)\b.{0,80}\b('
    r'write|confirm|summari[sz]e|transcribe|transkribier\w*|say|give|check|verify|evaluate|review|assess|'
    r'schreib|schreibe|gib|gebe|bestaetig|bestätig|fass|notier|pruef(?:e|st|t|en)?|pruf(?:e|st|t|en)?|prüf(?:e|st|t|en)?|bewert|kontrollier|markier'
    r')\b|'
    r'\b('
    r'audio(?:s)?|audiofassung(?:en)?|audioversion(?:en)?|audiogenerierung|audio generation|spoken text|spoken version|'
    r'gesprochen(?:e|en|er|es)?(?:\s+text|\s+version)?'
    r')\b.{0,80}\b('
    r'write|confirm|summari[sz]e|transcribe|transkribier\w*|say|give|check|verify|evaluate|review|assess|'
    r'schreib|schreibe|gib|gebe|bestaetig|bestätig|fass|notier|pruef(?:e|st|t|en)?|pruf(?:e|st|t|en)?|prüf(?:e|st|t|en)?|bewert|kontrollier|markier'
    r')\b.{0,80}\b(?:after|afterwards|then|danach|anschliessend|anschließend|nach)\b',
    re.IGNORECASE,
)
_POST_GENERATED_ARTIFACT_TEXT_RE = re.compile(
    r'\b(?:after|afterwards|then|danach|anschliessend|anschließend|nach)\b.{0,120}\b('
    r'write|describe|caption|summari[sz]e|confirm|explain|compare|reference|'
    r'schreib|schreibe|beschreib|beschreibe|bestaetig|bestätig|erklaer|erklar|erklär|'
    r'name|list|enumerate|nenn|nenne|auflist|aufzähl|aufzaehl'
    r')\b.{0,160}\b('
    r'(?:both\s+)?generated\s+artifacts?|(?:both\s+)?generated\s+outputs?|'
    r'both\s+artifacts?|both\s+outputs?|'
    r'artifact(?:s)?|artefact(?:s)?|artefakt(?:e)?|'
    r'audio.{0,60}(?:image|poster)|(?:image|poster).{0,60}audio'
    r')\b|'
    r'\b('
    r'write|describe|caption|summari[sz]e|confirm|explain|compare|reference|'
    r'schreib|schreibe|beschreib|beschreibe|bestaetig|bestätig|erklaer|erklar|erklär|'
    r'name|list|enumerate|nenn|nenne|auflist|aufzähl|aufzaehl'
    r')\b.{0,160}\b('
    r'(?:both\s+)?generated\s+artifacts?|(?:both\s+)?generated\s+outputs?|'
    r'both\s+artifacts?|both\s+outputs?|'
    r'audio.{0,60}(?:image|poster)|(?:image|poster).{0,60}audio'
    r')\b',
    re.IGNORECASE,
)
_POST_GENERATED_ARTIFACT_INVENTORY_RE = re.compile(
    r'\b('
    r'which|what|name|list|enumerate|'
    r'welche|was|nenn|nenne|auflist|aufzähl|aufzaehl'
    r')\w*\b.{0,140}\b('
    r'generated\s+)?(?:artifacts?|artefacts?|outputs?|'
    r'artefakt(?:e|en)?|ausgaben?)\b.{0,80}\b('
    r'generated|created|produced|erzeugt|generiert|erstellt'
    r')?\w*\b',
    re.IGNORECASE,
)
_COMPARE_WITH_ORIGINAL_TEXT_RE = re.compile(
    r'\b('
    r'compare|check|verify|review|'
    r'vergleich(?:e|en|st|t)?|pruef(?:e|en|st|t)?|prüf(?:e|en|st|t)?|'
    r'kontrollier(?:e|en|st|t)?'
    r')\b.{0,120}\b('
    r'original|source\s+text|original\s+text|'
    r'original(?:text)?|ursprungstext|ausgangstext'
    r')\b|'
    r'\b(?:transcript|transkript|transkription)\b.{0,80}\b(?:mit|with)\b.{0,80}\b('
    r'original|originaltext|ursprungstext|ausgangstext'
    r')\b',
    re.IGNORECASE,
)
_POST_TEXT_TO_AUDIO_RE = re.compile(
    r'\b('
    r'read|speak|narrate|voice|lies|lese|sprich|verton|erzähle|erzaehle'
    r')\b.{0,120}?\b('
    r'this|that|description|caption|summary|text|beschreibung|bestaetigung|bestätigung|satz|zeile'
    r')\b.{0,120}?\b(?:aloud|audio|spoken|vor|mp3|wav)\b|'
    r'\b('
    r'this|that|description|caption|summary|text|beschreibung|bestaetigung|bestätigung|satz|zeile'
    r')\b.{0,120}?\b(?:as|als)\b.{0,40}\b(?:audio|mp3|wav)\b|'
    r'\b(?:turn|convert|transform)\b.{0,48}?\b(?:it|this|that|story|text|script|slogan|tagline|reply)?\b'
    r'.{0,24}?\b(?:into|to|as)\b.{0,24}?\b(?:audio|speech|spoken|mp3|wav)\b',
    re.IGNORECASE,
)
_POST_TEXT_TO_IMAGE_RE = re.compile(
    r'\b('
    r'generate|create|render|show|make|generiere|generieren|erzeuge|erstellen|zeige'
    r')\b.{0,120}\b(?:second|another|zweite(?:s|n)?|weiteres|noch ein(?:es)?)\b.{0,80}\b('
    r'image|picture|bild'
    r')\b.{0,120}\b('
    r'description|caption|summary|text|beschreibung|bestaetigung|bestätigung|satz|zeile'
    r')\b|'
    r'\b('
    r'image|picture|bild'
    r')\b.{0,80}\b(?:based on|auf basis|basierend auf)\b.{0,80}\b('
    r'description|caption|summary|text|beschreibung|bestaetigung|bestätigung|satz|zeile'
    r')\b',
    re.IGNORECASE,
)
_TEXT_OUTPUT_ACTION_RE = re.compile(
    r'\b('
    r'write|describe|caption|summari[sz]e|confirm|explain|translate|compare|'
    r'visible\s+details|sichtbare\s+details|sichtbaren\s+details|'
    r'check|verify|evaluate|review|assess|name|list|enumerate|'
    r'vergleich(?:e|en|st|t)?|'
    r'nenn(?:e|en|st|t)?|auflist(?:e|en|est|et)?|aufzaehl(?:e|en|st|t)?|aufzähl(?:e|en|st|t)?|'
    r'schreib(?:e|en|st|t)?|beschreib(?:e|en|st|t)?|'
    r'gib(?:st)?|gebe|liefer(?:e|n|st|t)?|'
    r'bestaetig(?:e|en|st|t)?|bestätig(?:e|en|st|t)?|'
    r'fass(?:e|en|t)?|notier(?:e|en|st|t)?|'
    r'erklaer(?:e|en|st|t)?|erklar(?:e|en|st|t)?|erklär(?:e|en|st|t)?|'
    r'uebersetz(?:e|en|st|t)?|ubersetz(?:e|en|st|t)?|übersetz(?:e|en|st|t)?|'
    r'pruef(?:e|en|st|t)?|pruf(?:e|en|st|t)?|prüf(?:e|en|st|t)?|bewert(?:e|en|est|et)?|'
    r'kontrollier(?:e|en|st|t)?|markier(?:e|en|st|t)?'
    r')\b',
    re.IGNORECASE,
)
_TRANSCRIBE_ACTION_RE = re.compile(
    r'\b('
    r'transcribe|speech[-\s]?to[-\s]?text|stt|'
    r'transkribier|verschriftlich'
    r')\w*\b',
    re.IGNORECASE,
)
_DIRECT_INPUT_AUDIO_TRANSCRIPTION_TARGET_RE = re.compile(
    r'\b(?:transcribe|transkribier|verschriftlich)\w*\b'
    r'[^.;!?]{0,140}\b(?:'
    r'(?:attached|uploaded|selected|provided|input|source|'
    r'angehangt\w*|angehaengt\w*|hochgeladen\w*|ausgewahlt\w*|ausgewählt\w*|eingabe)\s+'
    r'(?:audio(?:s)?|audio\s*file|audiofile|recording(?:s)?|aufnahme(?:n)?|audiodatei|datei|file)|'
    r'(?:audio(?:s)?|audio\s*file|audiofile|recording(?:s)?|aufnahme(?:n)?|audiodatei|datei|file)'
    r'[^.;!?]{0,48}\b(?:attached|uploaded|selected|provided|input|source|'
    r'angehangt\w*|angehaengt\w*|hochgeladen\w*|ausgewahlt\w*|ausgewählt\w*|eingabe)'
    r')\b',
    re.IGNORECASE,
)
_SELECTED_AUDIO_TARGET_QUALIFIER_RE = re.compile(
    r'\b(?:selected|chosen|ausgewahlt\w*|ausgewählt\w*)\s+'
    r'(?:audio(?:s)?|audio\s*file|audiofile|recording(?:s)?|aufnahme(?:n)?|audiodatei|datei|file)\b|'
    r'\b(?:audio(?:s)?|audio\s*file|audiofile|recording(?:s)?|aufnahme(?:n)?|audiodatei|datei|file)'
    r'[^.;!?]{0,48}\b(?:selected|chosen|ausgewahlt\w*|ausgewählt\w*)\b',
    re.IGNORECASE,
)
_ANALYZE_IMAGE_ACTION_RE = re.compile(
    r'\b('
    r'analy[sz]e|analyse|analysier|inspect|examine|evaluate|review|'
    r'untersuch|pruef|prüf|bewert'
    r')\w*\b[^.;!?]{0,120}?\b('
    r'(?:actual\s+|attached\s+|generated\s+|created\s+|rendered\s+|'
    r'erzeugte(?:n|s|m|r)?\s+|generierte(?:n|s|m|r)?\s+)?'
    r'(?:image|images|picture|pictures|photo(?:s)?|foto(?:s)?|illustration(?:s)?|poster|posterbild(?:er)?|bildanalyse|bild(?:er|es|ern|e)?)|'
    r'(?:both|all|them|these|beide|alle|sie|diese)'
    r')\b',
    re.IGNORECASE,
)
_VISUAL_EVIDENCE_TEXT_TRIGGER_RE = re.compile(
    r'\b(?:actual\s+visual\s+evidence|visual\s+evidence|image\s+evidence)\b',
    re.IGNORECASE,
)
_MATERIALIZATION_ACTION_VERB_RE = re.compile(
    r'\b(?:generate|create|render|show|make|generier|generiere|generieren|erzeuge|erstelle|erstellen|zeige|'
    r'mach|mache|machen|read|speak|narrate|voice|lies|lese|sprich|erzähle|erzaehle|turn|convert|transform)\b',
    re.IGNORECASE,
)
_POST_AUDIO_TEXT_REQUIRES_STT_RE = re.compile(
    r'\b('
    r'exact\s+spoken\s+text|spoken\s+text|what\s+(?:was|is)\s+said|transcribe|transcription|'
    r'(?:text|audio).{0,80}\b(?:match|matches|fit|fits)|(?:match|matches|fit|fits).{0,80}\b(?:text|audio)|'
    r'confirm.{0,80}\b(?:spoken|text|said)|'
    r'(?:spoken\s+version|voice|audio|recording).{0,96}\b(?:urgent|serious|convincing|clear|tone|pace|intonation|emphasis|delivery)|'
    r'(?:urgent|serious|convincing|clear|tone|pace|intonation|emphasis|delivery).{0,96}\b(?:spoken\s+version|voice|audio|recording)|'
    r'gesprochen(?:e|en|er|es)?\s+text|transkribier|verschriftlich|'
    r'(?:text|audio).{0,80}\b(?:zusammenpass\w*|passen\s+zusammen)|(?:zusammenpass\w*|passen\s+zusammen).{0,80}\b(?:text|audio)|'
    r'bestaetig.{0,80}\b(?:gesprochen|text)|bestätig.{0,80}\b(?:gesprochen|text)|'
    r'(?:gesprochen(?:e|en|er|es)?\s+version|audio|aufnahme|stimme|tonlage).{0,96}\b(?:eindringlich|ernsthaft|dringlich|klar|betonung|sprechtempo|akustisch)|'
    r'(?:eindringlich|ernsthaft|dringlich|klar|betonung|sprechtempo|akustisch).{0,96}\b(?:gesprochen(?:e|en|er|es)?\s+version|audio|aufnahme|stimme|tonlage)'
    r')\b',
    re.IGNORECASE,
)
_POST_AUDIO_TEXT_REQUIRES_CHAT_AFTER_STT_RE = re.compile(
    r'\b('
    r'translate|translation|summari[sz]e|summary|explain|rewrite|rephrase|compare|'
    r'match|matches|fit|fits|review|assess|evaluate|judge|check|confirm|'
    r'übersetz|uebersetz|ubersetz|fasse|zusammenfassung|erklaer|erklar|erklär|'
    r'umschreib|vergleiche|prüf|pruef|bewert|beurteil|eindringlich|ernsthaft|dringlich|betonung|sprechtempo|akustisch|'
    r'zusammenpass\w*|passen\s+zusammen'
    r')\w*\b',
    re.IGNORECASE,
)
_FINAL_OUTPUT_MARKER_RE = re.compile(
    r'\b(?:finally|at\s+the\s+end|abschlie(?:ß|ss)end|am\s+ende|zum\s+schluss)\b',
    re.IGNORECASE,
)
_STRUCTURED_FINAL_OUTPUT_RE = re.compile(
    r'\b(?:json(?:[\s-]+)?(?:object|objekt)|structured\s+(?:object|output)|'
    r'strukturiert(?:e|en|er|es)?\s+(?:objekt|ausgabe))\b',
    re.IGNORECASE,
)
_FINAL_OUTPUT_BINDING_RE = re.compile(
    r'\b(?:bind\w*|combin\w*|connect\w*|join\w*|integrat\w*|'
    r'verbind\w*|verknupf\w*|zusammenfuhr\w*)\b',
    re.IGNORECASE,
)
_IMAGE_ARTIFACT_REF_RE = re.compile(
    r'\b(?:image|bild)[\s_-]*(?:artifact[\s_-]*ref|artefakt[\s_-]*referenz)\w*\b',
    re.IGNORECASE,
)
_AUDIO_ARTIFACT_REF_RE = re.compile(
    r'\baudio[\s_-]*(?:artifact[\s_-]*ref|artefakt[\s_-]*referenz)\w*\b',
    re.IGNORECASE,
)
_VISUAL_EVIDENCE_JOIN_RE = re.compile(
    r'\b(?:visual\s+evidence|image\s+evidence|bildevidenz|bildanalyse|'
    r'sichtbar\w*\s+(?:bild)?(?:evidenz|details?))\b',
    re.IGNORECASE,
)
_TRANSCRIPT_JOIN_RE = re.compile(
    r'\b(?:transcript\w*|transkript\w*|transkription\w*)\b',
    re.IGNORECASE,
)
_FINAL_JOIN_PLURAL_BINDING_RE = re.compile(
    r'\b(?:both|all|each|beide|beiden|alle|jede(?:n|r|s)?)\b',
    re.IGNORECASE,
)
_FINAL_JOIN_SELECTED_MEDIA_RE = re.compile(
    r'\b(?:only(?:\s+the)?|ausschlie(?:ß|ss)lich(?:\s+(?:den|die|das))?)\s+'
    r'(?P<ordinal>first|second|erst(?:e|en|er|es)?|zweit(?:e|en|er|es)?)\s+'
    r'(?P<media>image(?:\s+branch)?|bild(?:zweig)?|audio(?:\s+branch|zweig)?)\b',
    re.IGNORECASE,
)
_STRUCTURED_FINAL_QUOTE_SPAN_PATTERNS = (
    re.compile(r'(?<!\\)"[^"\n]{1,1000}(?<!\\)"'),
    re.compile(r'“[^”\n]{1,1000}”'),
    re.compile(r'„[^“\n]{1,1000}“'),
    re.compile(r'«[^»\n]{1,1000}»'),
    re.compile(r'‹[^›\n]{1,1000}›'),
    re.compile(r'(?<!`)`[^`\n]{1,1000}`(?!`)'),
    re.compile(r"(?<![\\\w])'[^'\n]{1,1000}'(?!\w)"),
)
_FINAL_JOIN_BINDING_PREFIX_NEGATION_RE = re.compile(
    r'\b(?:nicht|not|never)\s+(?:ausdrucklich|ausdrücklich|explicitly)?\s*$',
    re.IGNORECASE,
)
_FINAL_JOIN_BINDING_SUFFIX_NEGATION_RE = re.compile(
    r'^\s*(?:ausdrucklich|ausdrücklich|explicitly)?\s*(?:nicht|not|never)\b',
    re.IGNORECASE,
)
_FINAL_JOIN_SELECTOR_NEGATION_RE = re.compile(
    r'\b(?:nicht|not)\s*$',
    re.IGNORECASE,
)
_EXECUTABLE_BRANCH_CONTRACT_STATES = {
    *_PROMOTED_CONTRACT_STATES,
    'accepted',
    'active',
    'completed',
    'pending',
    'planned',
    'queued',
    'ready',
    'required',
    'running',
    'scheduled',
}
_EXECUTABLE_BRANCH_RESOLUTIONS = {
    'active',
    'completed',
    'pending',
    'pending_dependency',
    'planned',
    'queued',
    'ready',
    'running',
    'scheduled',
}
_DEPENDENT_CHAIN_MARKER_RE = re.compile(
    r'\b('
        r'after|afterwards|then|finally|lastly|based\s+on|using\s+(?:that|this|it)|from\s+(?:that|this|it)|'
        r'danach|anschliessend|anschließend|abschliessend|abschließend|zuletzt|nach|basierend\s+auf|auf\s+basis|'
    r'daraus|davon|hieraus|diese(?:r|s|n)?\s+(?:beschreibung|bestaetigung|bestätigung|text|satz|zeile)|'
    r'erzeugte(?:n|s|m|r)?\s+(?:bild(?:er|es|ern|e)?|audio)|generierte(?:n|s|m|r)?\s+(?:bild(?:er|es|ern|e)?|audio)|'
    r'generated\s+(?:image|images|audio)|spoken\s+text|gesprochenen\s+text'
    r')\b',
    re.IGNORECASE,
)
_NEGATED_IMAGE_MATERIALIZATION_RE = re.compile(
    r'\b(?:do\s+not|don[\'’]?t|not|no|without|kein|keine|keinen|nicht|ohne|noch\s+kein)\b'
    r'[^.;!?]{0,80}\b(?:generate|create|render|show|make|generier|erzeug|erstell|zeig)?\w*'
    r'[^.;!?]{0,80}\b(?:image|picture|bild|bilder)\b|'
    r'\b(?:generate|create|render|show|make|generier|erzeug|erstell|zeig)\w*'
    r'[^.;!?]{0,80}\b(?:do\s+not|don[\'’]?t|not|no|without|kein|keine|keinen|nicht|ohne|noch\s+kein)\b'
    r'[^.;!?]{0,80}\b(?:image|picture|bild|bilder)\b',
    re.IGNORECASE,
)
_IMAGE_MATERIALIZATION_VERB_RE = re.compile(
    r'\b(?:generate|create|render|show|make|generier|erzeug|erstell|zeig)\w*\b',
    re.IGNORECASE,
)
_DIRECT_IMAGE_NEGATION_RE = re.compile(
    r'\b(?:do\s+not|don[\'’]?t|not|no|without|kein|keine|keinen|nicht|ohne|noch\s+kein)\b'
    r'[^.;!?]{0,36}\b(?:image|picture|bild|bilder)\b',
    re.IGNORECASE,
)
_IMAGE_PATH_OR_LINK_CONSTRAINT_RE = re.compile(
    r'\b(?:'
    r'image\s*path|image\s*paths|asset\s*path|asset\s*paths|'
    r'bildpfad(?:e|en)?|dateipfad(?:e|en)?|pfad(?:e|en)?|link|links|verweis|verweisen|verlink|'
    r'href|src|url|saved|gespeichert(?:e|en|er|es)?|lokal|local|'
    r'platzhalter|placeholder|erfunden(?:e|en|er|es)?|extern(?:e|en|er|es)?|external|remote'
    r')\b',
    re.IGNORECASE,
)
_IMAGE_QUALITY_NEGATION_CONTEXT_RE = re.compile(
    r'\b(?:'
    r'marketing[\s-]?sprech|billig(?:e|en|er|es)?|'
    r'unnotig(?:e|en|er|es)?|unnoetig(?:e|en|er|es)?|unnötig(?:e|en|er|es)?|'
    r'effekt(?:e|en)?|typografie|hover[\s-]?effekt(?:e|en)?|copy|text'
    r')\b',
    re.IGNORECASE,
)
_NEGATED_AUDIO_MATERIALIZATION_RE = re.compile(
    r'\b(?:do\s+not|don[\'’]?t|not|no|without|kein|keine|keinen|nicht|ohne|noch\s+kein)\b'
    r'[^.,;!?]{0,80}\b(?:generate|create|read|speak|voice|generier|erzeug|lies|lese|sprich|verton)?\w*'
    r'[^.,;!?]{0,80}\b(?:audio|mp3|wav|spoken|vorlesen)\b|'
    r'\b(?:generate|create|read|speak|voice|generier|erzeug|lies|lese|sprich|verton)\w*'
    r'[^.,;!?]{0,80}\b(?:do\s+not|don[\'’]?t|not|no|without|kein|keine|keinen|nicht|ohne|noch\s+kein)\b'
    r'[^.,;!?]{0,80}\b(?:audio|mp3|wav|spoken|vorlesen)\b',
    re.IGNORECASE,
)
_RESERVED_IMAGE_MATERIALIZATION_RE = re.compile(
    r'\b(?:image|images|picture|pictures|posterbild(?:er)?|bildidee(?:n)?|bildkandidat(?:en)?|bild|bilder)\b'
    r'[^.;!?]{0,120}\b(?:reserved|reserve|option|optional|reserviert(?:e|en|er|es)?|'
    r'vormerk(?:e|en)?|merk(?:e|en)?|halte(?:n)?|festhalten)\b|'
    r'\b(?:reserved|reserve|option|optional|reserviert(?:e|en|er|es)?|'
    r'vormerk(?:e|en)?|merk(?:e|en)?|halte(?:n)?|festhalten)\b'
    r'[^.;!?]{0,120}\b(?:image|images|picture|pictures|posterbild(?:er)?|bildidee(?:n)?|bildkandidat(?:en)?|bild|bilder)\b',
    re.IGNORECASE,
)
_RESERVED_OPTION_ONLY_RE = re.compile(
    r'\b(?:reserved|reserve|keep|note|remember|option|optional|reserviert(?:e|en|er|es)?|'
    r'vormerk(?:e|en)?|merk(?:e|en)?|halte(?:n)?|festhalten)\b'
    r'[^.;!?]{0,120}\b(?:reserved|option|optional|candidate|kandidat(?:en)?|reserviert(?:e|en|er|es)?|option(?:en)?)\b',
    re.IGNORECASE,
)
_REJECTED_IMAGE_CANDIDATE_PLANNING_RE = re.compile(
    r'\b(?:do\s+not|don[\'’]?t|never|without|nicht|nie|ohne)\b'
    r'[^.;!?]{0,120}\b(?:plan|queue|schedule|reserve|propose|plan(?:e|en|t)?|vormerk|reservier)\w*\b'
    r'[^.;!?]{0,80}\b(?:image(?:s)?|picture(?:s)?|photo(?:s)?|illustration(?:s)?|bild(?:er)?|foto(?:s)?)\b|'
    r'\b(?:no|kein(?:e|en|er|es)?)\b[^.;!?]{0,32}'
    r'\b(?:image|picture|photo|illustration|bild|foto)[\s-]?(?:plan|candidate|option|planung|kandidat|option)\w*\b',
    re.IGNORECASE,
)
_SELECTED_IMAGE_CANDIDATE_INDEX_RE = re.compile(
    r'\b(?:generat(?:e|es|ed|ing)|create|render|show|make|generier(?:e|en)?|erzeuge|erstelle|zeige)\b'
    r'[^.;!?]{0,80}\b(?:only\s+the\s+|nur\s+(?:die|den|das)\s+)?'
    r'(?P<ordinal>first|second|third|fourth|fifth|sixth|erste|zweite|dritte|vierte|fuenfte|fünfte|sechste)\b',
    re.IGNORECASE,
)
_SELECTED_SPEAKABLE_CANDIDATE_INDEX_RE = re.compile(
    r'\b(?:read|speak|say|narrate|voice|lies|lese|sprich|sag(?:e)?|vorlesen)\b'
    r'[^.;!?]{0,100}\b(?:only\s+the\s+|nur\s+(?:die|den|das)\s+)?'
    r'(?P<ordinal>first|second|third|fourth|fifth|sixth|erste|zweite|dritte|vierte|fuenfte|fünfte|sechste)\b|'
    r'\b(?:only\s+the\s+|nur\s+(?:die|den|das)\s+)?'
    r'(?P<ordinal_prefix>first|second|third|fourth|fifth|sixth|erste|zweite|dritte|vierte|fuenfte|fünfte|sechste)\b'
    r'[^.;!?]{0,100}\b(?:read|speak|say|narrate|voice|lies|lese|sprich|sag(?:e)?|vorlesen)\b',
    re.IGNORECASE,
)
_SELECTED_SPEAKABLE_CANDIDATE_RE = re.compile(
    r'\b(?:best|beste(?:n|r|s)?|chosen|selected|gewahlt(?:e|en|er|es)?|gewaehlt(?:e|en|er|es)?|gewählte(?:n|r|s)?)\b'
    r'[^.;!?]{0,120}\b(?:read|speak|say|narrate|voice|audio|lies|lese|sprich|sag(?:e)?|vorlesen|vor)\b|'
    r'\b(?:read|speak|say|narrate|voice|audio|lies|lese|sprich|sag(?:e)?|vorlesen|vor)\b'
    r'[^.;!?]{0,120}\b(?:best|beste(?:n|r|s)?|chosen|selected|gewahlt(?:e|en|er|es)?|gewaehlt(?:e|en|er|es)?|gewählte(?:n|r|s)?)\b',
    re.IGNORECASE,
)
_IMAGE_CANDIDATE_CONTEXT_RE = re.compile(
    r'\b(?:image\s+ideas?|image\s+candidates?|bildidee(?:n)?|bildkandidat(?:en)?|bildoption(?:en)?|bilder?)\b',
    re.IGNORECASE,
)
_ORDINAL_INDEXES = {
    'first': 1,
    'erste': 1,
    'second': 2,
    'zweite': 2,
    'third': 3,
    'dritte': 3,
    'fourth': 4,
    'vierte': 4,
    'fifth': 5,
    'fuenfte': 5,
    'fünfte': 5,
    'sixth': 6,
    'sechste': 6,
}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _accepted_learning_hints_from_payloads(
    request: Mapping[str, Any],
    route: Mapping[str, Any],
    response: Mapping[str, Any],
) -> dict[str, Any]:
    for source in (route, request, response):
        hints = source.get('accepted_learning_hints') if isinstance(source, Mapping) else None
        if isinstance(hints, Mapping):
            return dict(hints)
        runtime = source.get('route_runtime') if isinstance(source, Mapping) and isinstance(source.get('route_runtime'), Mapping) else {}
        hints = runtime.get('accepted_learning_hints') if isinstance(runtime, Mapping) else None
        if isinstance(hints, Mapping):
            return dict(hints)
        runtime = source.get('runtime') if isinstance(source, Mapping) and isinstance(source.get('runtime'), Mapping) else {}
        hints = runtime.get('accepted_learning_hints') if isinstance(runtime, Mapping) else None
        if isinstance(hints, Mapping):
            return dict(hints)
    return {}


def _semantic_role_profile_from_payloads(
    request: Mapping[str, Any],
    route: Mapping[str, Any],
    response: Mapping[str, Any],
) -> dict[str, Any]:
    for source in (route, request, response):
        if not isinstance(source, Mapping):
            continue
        direct = source.get('semantic_role_profile')
        if isinstance(direct, Mapping):
            return dict(direct)
        route_runtime = source.get('route_runtime') if isinstance(source.get('route_runtime'), Mapping) else {}
        route_profile = route_runtime.get('semantic_role_profile') if isinstance(route_runtime, Mapping) else None
        if isinstance(route_profile, Mapping):
            return dict(route_profile)
        runtime = source.get('runtime') if isinstance(source.get('runtime'), Mapping) else {}
        runtime_profile = runtime.get('semantic_role_profile') if isinstance(runtime, Mapping) else None
        if isinstance(runtime_profile, Mapping):
            return dict(runtime_profile)
    return {}


def _clean_text(value: Any) -> str:
    return str(value or '').strip()


def _current_turn_prompt_from_request_payload(request_payload: Mapping[str, Any]) -> str:
    if not isinstance(request_payload, Mapping):
        return ''
    try:
        current = extract_responses_current_turn_prompt(dict(request_payload))
    except Exception:
        current = ''
    return _clean_text(current)


def _looks_like_serialized_responses_history(value: Any) -> bool:
    text = _clean_text(value)
    if not text:
        return False
    if isinstance(value, list):
        return True
    return (
        text.startswith('[{')
        and "'role'" in text
        and "'content'" in text
    ) or (
        text.startswith('[{')
        and '"role"' in text
        and '"content"' in text
    ) or (
        '[assistant]' in text.lower()
        and '[user]' in text.lower()
    )


def _response_output_text(response_payload: Mapping[str, Any]) -> str:
    for key in ('output_text', 'content', 'text', 'response'):
        value = response_payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ''


def _assistant_output_claimed_downstream_capabilities(
    prompt_analysis: Mapping[str, Any],
    response_payload: Mapping[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    if bool(prompt_analysis.get('meta_execution_explanation_request')):
        return [], []
    text = _response_output_text(response_payload)
    if not text:
        return [], []
    normalized = normalize_intent_text(text)
    if not normalized or _ASSISTANT_OUTPUT_HYPOTHETICAL_RE.search(normalized):
        return [], []
    if not _ASSISTANT_OUTPUT_STRONG_FOLLOW_UP_RE.search(normalized):
        return [], []

    capabilities: list[str] = []
    refinements: list[dict[str, Any]] = []
    for capability, pattern in (
        (CAPABILITY_TEXT_TO_SPEECH, _ASSISTANT_OUTPUT_TTS_CLAIM_RE),
        (CAPABILITY_IMAGE_GENERATION, _ASSISTANT_OUTPUT_IMAGE_CLAIM_RE),
    ):
        if not pattern.search(normalized):
            continue
        capabilities.append(capability)
        refinements.append(
            {
                'source': 'assistant_output_claim',
                'capability': capability,
                'reason': 'assistant_output_explicitly_claimed_pending_materialization',
            }
        )
    return _unique_capabilities(capabilities), refinements


def _refine_prompt_analysis_from_response_output(
    prompt_analysis: Mapping[str, Any],
    response_payload: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    updated = dict(prompt_analysis or {})
    claimed_capabilities, refinements = _assistant_output_claimed_downstream_capabilities(
        updated,
        response_payload,
    )
    if not claimed_capabilities:
        return updated, []

    existing = _unique_capabilities(
        updated.get('downstream_follow_up_capabilities')
        if isinstance(updated.get('downstream_follow_up_capabilities'), list)
        else []
    )
    new_capabilities = [capability for capability in claimed_capabilities if capability not in existing]
    if not new_capabilities:
        return updated, []

    downstream = list(existing)
    downstream.extend(new_capabilities)
    updated['downstream_follow_up_capabilities'] = _unique_capabilities(downstream)
    capability_scores = (
        dict(updated.get('capability_scores'))
        if isinstance(updated.get('capability_scores'), Mapping)
        else {}
    )
    for capability in new_capabilities:
        capability_scores[capability] = max(int(capability_scores.get(capability) or 0), 4)
        if not normalize_capability(updated.get('primary_capability')):
            updated['primary_capability'] = capability
        if capability == CAPABILITY_TEXT_TO_SPEECH:
            updated['requests_audio_output'] = True
            updated['has_audio_follow_up_request'] = True
            updated['text_preparation_before_audio_output'] = True
            if _coerce_positive_int(updated.get('requested_audio_output_count')) <= 0:
                updated['requested_audio_output_count'] = 1
            updated['text_first_follow_up_capability'] = (
                normalize_capability(updated.get('text_first_follow_up_capability'))
                or CAPABILITY_TEXT_TO_SPEECH
            )
        if capability == CAPABILITY_IMAGE_GENERATION:
            updated['requests_visual_output'] = True
            updated['has_visual_follow_up_request'] = True
            updated['text_preparation_before_visual_output'] = True
            if _coerce_positive_int(updated.get('requested_visual_output_count')) <= 0:
                updated['requested_visual_output_count'] = 1
            updated['text_first_follow_up_capability'] = (
                normalize_capability(updated.get('text_first_follow_up_capability'))
                or CAPABILITY_IMAGE_GENERATION
            )
    updated['capability_scores'] = capability_scores
    return updated, [
        refinement
        for refinement in refinements
        if refinement.get('capability') in new_capabilities
    ]


def _output_type_for_capability(capability: Any) -> Optional[str]:
    token = normalize_capability(capability)
    if token == CAPABILITY_TEXT_TO_SPEECH:
        return 'audio'
    if token == CAPABILITY_IMAGE_GENERATION:
        return 'image'
    if token == CAPABILITY_SPEECH_TO_TEXT:
        return 'text'
    if token == CAPABILITY_VISION_ANALYSIS:
        return 'text'
    if token == CAPABILITY_CHAT:
        return 'text'
    return None


def _phase_status_from_late_fill(status: str) -> str:
    normalized = _clean_text(status).lower()
    if normalized == 'completed':
        return 'completed'
    if normalized == 'failed':
        return 'blocked'
    if normalized in {'running', 'queued', 'pending', 'scheduled', 'accepted'}:
        return 'pending'
    return 'pending'


def _artifact_matches_output_type(artifacts: Any, output_type: str) -> bool:
    if not isinstance(artifacts, list):
        return False
    for raw_artifact in artifacts:
        if not isinstance(raw_artifact, Mapping):
            continue
        artifact_type = _clean_text(raw_artifact.get('type') or raw_artifact.get('kind')).lower()
        if not artifact_type:
            continue
        if output_type == 'audio' and artifact_type in {'audio', 'wav', 'mp3', 'm4a', 'flac'}:
            return True
        if output_type == 'image' and artifact_type in {'image', 'png', 'jpg', 'jpeg', 'webp'}:
            return True
        if output_type == 'text' and artifact_type in {'text', 'markdown', 'md', 'json', 'csv', 'message'}:
            return True
    return False


def _coerce_positive_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def _requested_audio_output_count(prompt_analysis: Mapping[str, Any]) -> int:
    if bool(prompt_analysis.get('audio_output_count_exceeds_bound')):
        return 0
    requested_count = _coerce_positive_int(prompt_analysis.get('requested_audio_output_count'))
    maximum = _coerce_positive_int(prompt_analysis.get('requested_audio_output_count_max')) or 6
    return requested_count if 0 < requested_count <= maximum else 0


def _visual_execution_is_preserved(
    prompt_analysis: Mapping[str, Any],
    capability: Any,
) -> bool:
    normalized = normalize_capability(capability)
    if normalized == CAPABILITY_IMAGE_GENERATION:
        if 'visual_artifact_execution_suppressed_by_preservation' in prompt_analysis:
            return bool(prompt_analysis.get('visual_artifact_execution_suppressed_by_preservation'))
        return bool(prompt_analysis.get('visual_artifact_preservation_without_regeneration'))
    if normalized == CAPABILITY_VISION_ANALYSIS:
        if 'visual_analysis_execution_suppressed_by_preservation' in prompt_analysis:
            return bool(prompt_analysis.get('visual_analysis_execution_suppressed_by_preservation'))
        return bool(prompt_analysis.get('visual_analysis_preservation_without_reanalysis'))
    return False


def _clean_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _input_artifact_records(request_payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    for key in (
        'input_artifacts',
        'reference_artifacts',
        'selected_reference_artifacts',
        'selectedReferenceArtifacts',
    ):
        raw_items = request_payload.get(key)
        if isinstance(raw_items, list):
            records.extend(item for item in raw_items if isinstance(item, Mapping))
    raw_item = request_payload.get('selected_reference_artifact') or request_payload.get('selectedReferenceArtifact')
    if isinstance(raw_item, Mapping):
        records.append(raw_item)
    return records


def _preserved_visual_reference_input_refs(
    prompt_analysis: Mapping[str, Any],
    request_payload: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    """Return exact selected-reference handles for a preserved visual world.

    A carried image is reference truth, not a fresh image-generation phase.  Bind
    the concrete image plus its same-message evidence source into the terminal
    join without carrying unrelated predecessor audio or transcript artifacts.
    """

    if not (
        bool(prompt_analysis.get('visual_artifact_preservation_without_regeneration'))
        or bool(prompt_analysis.get('visual_analysis_preservation_without_reanalysis'))
    ):
        return [], ''

    records = _input_artifact_records(request_payload)
    image_records: list[Mapping[str, Any]] = []
    seen_images: set[tuple[str, str]] = set()
    for record in records:
        artifact_type = _clean_text(record.get('type') or record.get('kind')).lower()
        if artifact_type not in {'image', 'png', 'jpg', 'jpeg', 'webp'}:
            continue
        artifact_ref = _clean_text(record.get('artifact_ref') or record.get('ref'))
        path = _clean_text(record.get('path') or record.get('artifact_path'))
        if not artifact_ref and not path:
            continue
        identity = (artifact_ref, path)
        if identity in seen_images:
            continue
        seen_images.add(identity)
        image_records.append(record)

    # Preservation of a singular selected visual must not silently choose among
    # several carried images.  The caller can still expose the ambiguity as an
    # ordinary open contract instead of regenerating one of them.
    if len(image_records) != 1:
        return (
            [],
            'preserved_visual_reference_missing'
            if not image_records
            else 'preserved_visual_reference_ambiguous',
        )

    image = image_records[0]
    artifact_ref = _clean_text(image.get('artifact_ref') or image.get('ref'))
    path = _clean_text(image.get('path') or image.get('artifact_path'))
    source_response_id = _clean_text(image.get('source_response_id'))
    source_message_id = _clean_text(
        image.get('source_message_id') or image.get('message_id')
    )
    refs: list[dict[str, Any]] = [
        {
            key: value
            for key, value in {
                'kind': 'selected_reference',
                'role': 'preserved_visual_artifact',
                'artifact_type': 'image',
                'artifact_ref': artifact_ref or None,
                'path': path or None,
                'source_response_id': source_response_id or None,
                'source_message_id': source_message_id or None,
            }.items()
            if value not in (None, '')
        }
    ]

    evidence_messages = [
        record
        for record in records
        if _clean_text(record.get('type') or record.get('kind')).lower() == 'message'
        and (
            (source_message_id and _clean_text(record.get('message_id')) == source_message_id)
            or (
                source_response_id
                and _clean_text(record.get('source_response_id')) == source_response_id
            )
        )
    ]
    if len(evidence_messages) == 1:
        message = evidence_messages[0]
        refs.append(
            {
                key: value
                for key, value in {
                    'kind': 'selected_reference',
                    'role': 'preserved_visual_evidence',
                    'message_id': _clean_text(message.get('message_id')) or None,
                    'source_response_id': _clean_text(
                        message.get('source_response_id')
                    ) or None,
                }.items()
                if value not in (None, '')
            }
        )
    elif bool(prompt_analysis.get('visual_analysis_preservation_without_reanalysis')):
        return [], (
            'preserved_visual_evidence_missing'
            if not evidence_messages
            else 'preserved_visual_evidence_ambiguous'
        )
    return refs, ''


def _source_artifact_records(*payloads: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    for request_payload in payloads:
        if not isinstance(request_payload, Mapping):
            continue
        for key in (
            'input_artifacts',
            'reference_artifacts',
            'selected_reference_artifacts',
            'selectedReferenceArtifacts',
        ):
            raw_items = request_payload.get(key)
            if not isinstance(raw_items, list):
                continue
            records.extend(item for item in raw_items if isinstance(item, Mapping))
        raw_item = request_payload.get('selected_reference_artifact') or request_payload.get('selectedReferenceArtifact')
        if isinstance(raw_item, Mapping):
            records.append(raw_item)
        for key in (
            'route_artifact_path',
            'route_artifact_ref',
            'artifact_path',
            'artifact_ref',
            'file_path',
            'saved_image_path',
            'saved_audio_path',
            'saved_text_path',
        ):
            value = _clean_text(request_payload.get(key))
            if value:
                records.append({'path': value, 'artifact_ref': value})
    return records


def _iter_payload_text_values(value: Any) -> list[str]:
    texts: list[str] = []
    if isinstance(value, str):
        if value.strip():
            texts.append(value)
        return texts
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key in {
                'text',
                'content',
                'input',
                'messages',
                'ghost_messages',
                'output_text',
            }:
                texts.extend(_iter_payload_text_values(nested))
        return texts
    if isinstance(value, list):
        for item in value:
            texts.extend(_iter_payload_text_values(item))
    return texts


def _source_name_from_filename(filename: str, extension: str) -> str:
    stem = _clean_text(filename).rsplit('/', 1)[-1]
    normalized_extension = _clean_text(extension).lower()
    suffix = f'.{normalized_extension}' if normalized_extension else ''
    if suffix and stem.lower().endswith(suffix):
        stem = stem[: -len(suffix)]
    return stem or f'updated-{normalized_extension or "text"}'


def _text_source_prompt_mentions_extension(prompt: str, extension: str) -> bool:
    normalized_extension = normalize_text_artifact_extension(extension or '') or ''
    if not normalized_extension:
        return False
    text = _clean_text(prompt).lower()
    if not text:
        return False
    aliases = {
        'html': (r'\bhtml\b', r'\bhtm\b', r'\bweb\s?page\b', r'\bwebseite\b'),
        'css': (r'\bcss\b', r'\bstylesheet\b', r'\bstyles?\b', r'\bstil(?:e|es)?\b'),
        'js': (r'\bjs\b', r'\bjavascript\b', r'\bscript\b'),
        'md': (r'\bmd\b', r'\bmarkdown\b', r'\breadme\b'),
        'txt': (r'\btxt\b', r'\bplain\s+text\b', r'\btext\b'),
    }
    patterns = aliases.get(normalized_extension, (rf'\b{re.escape(normalized_extension)}\b',))
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _text_source_has_explicit_extension_target(prompt: str) -> bool:
    return any(
        _text_source_prompt_mentions_extension(prompt, extension)
        for extension in ('html', 'css', 'js', 'md', 'txt', 'json', 'svg', 'py', 'sh')
    )


def _text_source_request_from_path(
    path_value: str,
    *,
    source: str,
) -> Optional[dict[str, str]]:
    path_token = _clean_text(path_value)
    if not path_token:
        return None
    extension = normalize_text_artifact_extension(Path(path_token).suffix)
    if not extension:
        return None
    return {
        'extension': extension,
        'source': source,
        'source_name': _source_name_from_filename(Path(path_token).name, extension),
        'target_path': path_token,
    }


def _linked_css_source_requests_from_html(
    path_value: str,
    prompt: str,
) -> list[dict[str, str]]:
    if not _TEXT_SOURCE_LINKED_CSS_CUE_RE.search(_clean_text(prompt)):
        return []
    request = _text_source_request_from_path(path_value, source='selected_source_edit')
    if not request or request.get('extension') not in {'html', 'htm'}:
        return []
    try:
        html_path = Path(path_value).expanduser()
        if not html_path.exists() or not html_path.is_file() or html_path.stat().st_size > 400_000:
            return []
        html = html_path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return []
    requests: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for match in _TEXT_SOURCE_HTML_STYLESHEET_LINK_RE.finditer(html):
        href = _clean_text(match.group('href')).split('#', 1)[0].split('?', 1)[0]
        if not href or re.match(r'^(?:https?:|//|data:|blob:|mailto:|tel:|javascript:)', href, re.IGNORECASE):
            continue
        try:
            css_path = (html_path.parent / href).expanduser().resolve(strict=False)
        except OSError:
            continue
        if not css_path.exists() or not css_path.is_file():
            continue
        css_path_token = str(css_path)
        if css_path_token in seen_paths:
            continue
        seen_paths.add(css_path_token)
        css_request = _text_source_request_from_path(css_path_token, source='selected_source_edit')
        if css_request and css_request.get('extension') == 'css':
            requests.append(css_request)
    return requests


def _text_artifact_source_requests_from_history(
    prompt: str,
    *payloads: Mapping[str, Any],
) -> list[dict[str, str]]:
    if not _TEXT_SOURCE_EDIT_CUE_RE.search(_clean_text(prompt)):
        return []
    requests: list[dict[str, str]] = []
    seen_keys: set[tuple[str, str]] = set()

    def append_request(request: dict[str, str]) -> None:
        extension = _clean_text(request.get('extension')).lower()
        target_path = _clean_text(request.get('target_path'))
        source_name = _clean_text(request.get('source_name'))
        key = (extension, target_path or source_name)
        if not extension or key in seen_keys:
            return
        seen_keys.add(key)
        requests.append(request)

    explicit_extension_target = _text_source_has_explicit_extension_target(prompt)
    json_intent = _json_text_artifact_intent(prompt)
    explicit_json_source_edit = bool(
        json_intent.has_materialization
        or _JSON_SOURCE_EDIT_TARGET_RE.search(_clean_text(prompt))
    )
    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        for text in _iter_payload_text_values(payload):
            for match in _TEXT_SOURCE_FENCED_BLOCK_RE.finditer(text):
                language = re.sub(r'[^a-z0-9]+', '', _clean_text(match.group('lang')).lower())
                body = str(match.group('body') or '')
                filename_match = _TEXT_SOURCE_FILENAME_RE.search(body)
                extension = ''
                source_name = ''
                if filename_match:
                    extension = normalize_text_artifact_extension(filename_match.group('ext') or '') or ''
                    source_name = _source_name_from_filename(filename_match.group('name') or '', extension)
                if not extension:
                    extension = _TEXT_SOURCE_LANGUAGE_EXTENSIONS.get(language) or ''
                    extension = normalize_text_artifact_extension(extension) or ''
                if not extension:
                    continue
                if (
                    extension == 'json'
                    and json_intent.suppress_generic_fallback
                    and not explicit_json_source_edit
                ):
                    # A fenced JSON predecessor remains bounded evidence when the
                    # current turn asks for a JSON response object, not a new file.
                    continue
                if explicit_extension_target and not _text_source_prompt_mentions_extension(prompt, extension):
                    continue
                append_request(
                    {
                        'extension': extension,
                        'source': 'history_source_edit',
                        'source_name': source_name or f'updated-{extension}',
                    }
                )
        for artifact in _source_artifact_records(payload):
            artifact_type = _clean_text(artifact.get('type') or artifact.get('kind')).lower()
            path_token = _clean_text(artifact.get('path') or artifact.get('source_path'))
            if artifact_type and artifact_type not in {'text', 'document', 'file'}:
                continue
            request = _text_source_request_from_path(path_token, source='selected_source_edit')
            if not request:
                continue
            extension = _clean_text(request.get('extension')).lower()
            if (
                extension == 'json'
                and json_intent.suppress_generic_fallback
                and not explicit_json_source_edit
            ):
                continue
            if explicit_extension_target and not _text_source_prompt_mentions_extension(prompt, extension):
                continue
            append_request(request)
            for linked_request in _linked_css_source_requests_from_html(path_token, prompt):
                append_request(linked_request)
    return requests


def _merge_text_artifact_requests(
    explicit_requests: list[dict[str, str]],
    source_requests: list[dict[str, str]],
) -> list[dict[str, str]]:
    if not source_requests:
        return explicit_requests
    merged: list[dict[str, str]] = []
    for request in explicit_requests:
        if (
            request.get('extension') == 'txt'
            and request.get('source') == 'explicit_file_cue'
            and request.get('source_name') == 'generated-text'
        ):
            continue
        merged.append(request)
    existing_extensions = {
        _clean_text(request.get('extension')).lower()
        for request in merged
        if _clean_text(request.get('extension'))
    }
    for request in source_requests:
        extension = _clean_text(request.get('extension')).lower()
        if not extension or extension in existing_extensions:
            continue
        existing_extensions.add(extension)
        merged.append(request)
    return merged


def _apply_implicit_web_page_visual_binding(
    prompt_analysis: Mapping[str, Any],
    text_artifact_requests: list[dict[str, str]],
) -> dict[str, Any]:
    updated = dict(prompt_analysis or {})
    has_implicit_html_page = any(
        _clean_text(request.get('extension')).lower() in {'html', 'htm'}
        and _clean_text(request.get('source')) == 'implicit_web_page_cue'
        for request in text_artifact_requests
    )
    if not has_implicit_html_page:
        return updated
    has_visual_work = bool(
        updated.get('requests_visual_output')
        or updated.get('has_visual_follow_up_request')
        or updated.get('text_preparation_before_visual_output')
        or _coerce_positive_int(updated.get('requested_visual_output_count')) > 0
    )
    if not has_visual_work:
        return updated
    updated['local_visual_asset_requirement'] = True
    cues = _clean_string_list(updated.get('local_visual_asset_cues'))
    if 'implicit_web_page_generated_images' not in cues:
        cues.append('implicit_web_page_generated_images')
    updated['local_visual_asset_cues'] = cues
    return updated


def _current_predecessor_image_prompts(
    request_payload: Mapping[str, Any],
) -> list[str]:
    context = (
        request_payload.get('current_predecessor_context')
        if isinstance(request_payload.get('current_predecessor_context'), Mapping)
        else {}
    )
    if (
        _clean_text(context.get('status')).lower() != 'authorized'
        or _clean_text(context.get('authorization'))
        != 'canonical_same_conversation_predecessor'
    ):
        return []
    return _clean_string_list(context.get('batch_prompts'))


def _apply_current_predecessor_image_prompt_contract(
    prompt_analysis: Mapping[str, Any],
    request_payload: Mapping[str, Any],
    text_artifact_requests: list[dict[str, str]],
) -> tuple[dict[str, Any], list[str]]:
    prompts = _current_predecessor_image_prompts(request_payload)
    updated = dict(prompt_analysis or {})
    if not prompts or not bool(updated.get('requests_visual_output')):
        return updated, []
    updated['requested_visual_output_count'] = max(
        _coerce_positive_int(updated.get('requested_visual_output_count')),
        len(prompts),
    )
    updated['counted_visual_output_obligation'] = True
    updated['has_visual_follow_up_request'] = True
    updated['separate_visual_generation_request'] = True
    if any(
        _clean_text(item.get('extension')).lower() in {'html', 'htm', 'css'}
        for item in text_artifact_requests
    ):
        updated['local_visual_asset_requirement'] = True
        cues = _clean_string_list(updated.get('local_visual_asset_cues'))
        if 'current_predecessor_prompt_page_binding' not in cues:
            cues.append('current_predecessor_prompt_page_binding')
        updated['local_visual_asset_cues'] = cues
    return updated, prompts


def _bind_current_predecessor_image_prompts(
    branches: list[dict[str, Any]],
    prompts: list[str],
    graph_refinements: list[dict[str, Any]],
) -> None:
    if not prompts:
        return
    image_branches = sorted(
        (
            branch
            for branch in branches
            if normalize_capability(branch.get('capability'))
            == CAPABILITY_IMAGE_GENERATION
            and not _is_unpromoted_candidate_record(branch)
        ),
        key=lambda item: (
            _coerce_positive_int(item.get('queue_index')),
            _clean_text(item.get('phase_id')),
        ),
    )
    if len(image_branches) < len(prompts):
        return
    for index, prompt in enumerate(prompts, start=1):
        branch = image_branches[index - 1]
        branch['artifact_prompt'] = prompt
        branch['artifact_prompt_source'] = (
            'canonical_current_predecessor_batch_prompts'
        )
        branch['batch_prompts'] = list(prompts)
    graph_refinements.append(
        {
            'source': 'canonical_current_predecessor_batch_prompts',
            'refinement': 'current_predecessor_prompt_batch_binding',
            'capability': CAPABILITY_IMAGE_GENERATION,
            'bound_prompt_count': len(prompts),
            'bound_branch_ids': [
                _clean_text(item.get('branch_id'))
                for item in image_branches[:len(prompts)]
                if _clean_text(item.get('branch_id'))
            ],
        }
    )


def _bounded_directive_clause_end(prompt_text: str, offset: int) -> int:
    tail = prompt_text[max(0, offset):]
    boundaries = [
        match.start()
        for pattern in (
            re.compile(r'[.;!?\n]'),
            re.compile(r'(?i),\s*(?:and\s+)?then\b|\band\s+then\b|\bthen\b'),
        )
        for match in [pattern.search(tail)]
        if match is not None
    ]
    return max(0, offset) + min(boundaries) if boundaries else len(prompt_text)


def _direct_clause_local_media_payloads(
    prompt_text: str,
    *,
    prompt_analysis: Mapping[str, Any],
) -> dict[str, list[str]]:
    """Extract only self-contained media payloads from their own action clauses."""

    prompt = str(prompt_text or '').strip()
    if not prompt:
        return {'image_generation': [], 'text_to_speech': []}
    masked_prompt = mask_intent_literal_payloads(prompt)

    image_payloads: list[str] = []
    for action in _GENERATE_IMAGE_ACTION_RE.finditer(masked_prompt):
        clause_end = _bounded_directive_clause_end(masked_prompt, action.end())
        tail = prompt[action.end():clause_end]
        prefix = _DIRECT_IMAGE_DESCRIPTION_PREFIX_RE.match(tail)
        if not prefix:
            continue
        description = tail[prefix.end():].strip(' \t,:-')
        if description and not _DIRECT_MEDIA_DEICTIC_PAYLOAD_RE.match(description):
            image_payloads.append(description)

    tts_source = resolve_explicit_tts_source(prompt)
    explicit_spoken_text = _clean_text(tts_source.get('text'))
    spoken_payloads = (
        [explicit_spoken_text]
        if explicit_spoken_text
        and bool(prompt_analysis.get('direct_audio_materialization_request'))
        and not bool(prompt_analysis.get('text_preparation_before_audio_output'))
        else []
    )

    return {
        CAPABILITY_IMAGE_GENERATION: list(dict.fromkeys(image_payloads)),
        CAPABILITY_TEXT_TO_SPEECH: list(dict.fromkeys(spoken_payloads)),
    }


def _bind_direct_clause_local_media_payloads(
    branches: list[dict[str, Any]],
    *,
    prompt_text: str,
    prompt_analysis: Mapping[str, Any],
    graph_refinements: list[dict[str, Any]],
) -> None:
    payloads = _direct_clause_local_media_payloads(
        prompt_text,
        prompt_analysis=prompt_analysis,
    )
    bound: list[dict[str, str]] = []

    image_branches = [
        branch
        for branch in branches
        if not _is_unpromoted_candidate_record(branch)
        if normalize_capability(branch.get('capability'))
        == CAPABILITY_IMAGE_GENERATION
    ]
    image_payloads = payloads.get(CAPABILITY_IMAGE_GENERATION) or []
    if (
        len(image_branches) == 1
        and len(image_payloads) == 1
        and not _clean_text(image_branches[0].get('artifact_prompt'))
    ):
        image_branches[0]['artifact_prompt'] = image_payloads[0]
        image_branches[0]['artifact_prompt_source'] = (
            'current_turn_direct_image_clause'
        )
        bound.append(
            {
                'branch_id': _clean_text(image_branches[0].get('branch_id')),
                'capability': CAPABILITY_IMAGE_GENERATION,
                'payload_source': 'current_turn_direct_image_clause',
            }
        )

    tts_branches = [
        branch
        for branch in branches
        if not _is_unpromoted_candidate_record(branch)
        if normalize_capability(branch.get('capability'))
        == CAPABILITY_TEXT_TO_SPEECH
    ]
    spoken_payloads = payloads.get(CAPABILITY_TEXT_TO_SPEECH) or []
    if (
        len(tts_branches) == 1
        and len(spoken_payloads) == 1
        and not _clean_text(tts_branches[0].get('content_payload'))
        and not _clean_text(tts_branches[0].get('selection_policy'))
    ):
        tts_branches[0]['content_payload'] = spoken_payloads[0]
        tts_branches[0]['content_payload_source'] = (
            'current_turn_direct_spoken_clause'
        )
        bound.append(
            {
                'branch_id': _clean_text(tts_branches[0].get('branch_id')),
                'capability': CAPABILITY_TEXT_TO_SPEECH,
                'payload_source': 'current_turn_direct_spoken_clause',
            }
        )

    if bound:
        graph_refinements.append(
            {
                'source': 'current_turn_direct_media_payload_binding',
                'refinement': 'clause_local_media_payload_binding',
                'bindings': bound,
            }
        )


def _has_source_artifact_available(*payloads: Mapping[str, Any]) -> bool:
    return any(
        _clean_text(
            artifact.get('path')
            or artifact.get('source_path')
            or artifact.get('url')
            or artifact.get('artifact_ref')
            or artifact.get('ref')
            or artifact.get('content')
        )
        for artifact in _source_artifact_records(*payloads)
    )


def _artifact_type_token(artifact: Mapping[str, Any]) -> str:
    token = _clean_text(artifact.get('type') or artifact.get('kind')).lower()
    if token:
        return token
    path = _clean_text(artifact.get('path') or artifact.get('source_path')).lower()
    return path.rsplit('.', 1)[-1] if '.' in path else ''


def _has_audio_input_artifact(request_payload: Mapping[str, Any]) -> bool:
    return any(
        _artifact_type_token(artifact) in _AUDIO_ARTIFACT_TYPES
        for artifact in _input_artifact_records(request_payload)
    )


def _has_current_audio_input_artifact(request_payload: Mapping[str, Any]) -> bool:
    raw_items = request_payload.get('input_artifacts')
    return bool(
        isinstance(raw_items, list)
        and any(
            isinstance(artifact, Mapping)
            and _artifact_type_token(artifact) in _AUDIO_ARTIFACT_TYPES
            for artifact in raw_items
        )
    )


def _has_selected_reference_audio_artifact(request_payload: Mapping[str, Any]) -> bool:
    records: list[Mapping[str, Any]] = []
    for key in (
        'reference_artifacts',
        'selected_reference_artifacts',
        'selectedReferenceArtifacts',
    ):
        raw_items = request_payload.get(key)
        if isinstance(raw_items, list):
            records.extend(item for item in raw_items if isinstance(item, Mapping))
    raw_item = (
        request_payload.get('selected_reference_artifact')
        or request_payload.get('selectedReferenceArtifact')
    )
    if isinstance(raw_item, Mapping):
        records.append(raw_item)
    return any(
        _artifact_type_token(artifact) in _AUDIO_ARTIFACT_TYPES
        for artifact in records
    )


def _prompt_references_current_input_audio(prompt_analysis: Mapping[str, Any]) -> bool:
    normalized_prompt = _clean_text(prompt_analysis.get('normalized_prompt')).lower()
    return bool(normalized_prompt and _CURRENT_INPUT_AUDIO_REFERENCE_RE.search(normalized_prompt))


def _unique_capabilities(values: list[Any]) -> list[str]:
    items: list[str] = []
    for value in values:
        capability = normalize_capability(value)
        if not capability or capability in items:
            continue
        items.append(capability)
    return items


def _contract_state(record: Mapping[str, Any]) -> str:
    for key in ('contract_state', 'contract_status', 'obligation_state', 'intent_state'):
        token = _clean_text(record.get(key)).lower()
        if token:
            return token
    return _clean_text(record.get('status')).lower()


def _explicit_required_flag(record: Mapping[str, Any]) -> Optional[bool]:
    if 'required' not in record:
        return None
    value = record.get('required')
    if isinstance(value, bool):
        return value
    token = _clean_text(value).lower()
    if token in {'true', 'yes', '1', 'required'}:
        return True
    if token in {'false', 'no', '0', 'optional'}:
        return False
    return None


def _is_unpromoted_candidate_record(record: Mapping[str, Any]) -> bool:
    state = _contract_state(record)
    if state in _PROMOTED_CONTRACT_STATES or _clean_text(record.get('promoted_from_candidate_id')):
        return False
    if _clean_text(record.get('candidate_id')):
        return True
    if state in _CANDIDATE_CONTRACT_STATES:
        return True
    return _explicit_required_flag(record) is False


def _phase_records(phase_graph: Optional[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(phase_graph, Mapping):
        return []
    phases = phase_graph.get('phases')
    if not isinstance(phases, list):
        return []
    return [dict(item) for item in phases if isinstance(item, Mapping)]


def _downstream_phase_records(
    phase_graph: Optional[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(phase_graph, Mapping):
        return []
    current_phase_id = _clean_text(phase_graph.get('current_phase_id'))
    records: list[dict[str, Any]] = []
    for phase in _phase_records(phase_graph):
        phase_id = _clean_text(phase.get('phase_id'))
        if not phase_id or phase_id == current_phase_id:
            continue
        if _is_unpromoted_candidate_record(phase):
            continue
        records.append(phase)
    return records


def _promoted_branch_records(branches: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(branch)
        for branch in branches
        if isinstance(branch, Mapping) and not _is_unpromoted_candidate_record(branch)
    ]


def _workload_task_lookup(workload_graph: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    tasks = workload_graph.get('tasks') if isinstance(workload_graph.get('tasks'), list) else []
    for raw_task in tasks:
        if not isinstance(raw_task, Mapping):
            continue
        task = dict(raw_task)
        for key in ('phase_id', 'branch_id', 'task_id', 'workload_task_id'):
            token = _clean_text(task.get(key))
            if token:
                lookup[token] = task
    return lookup


def _project_workload_task_fields(
    record: Mapping[str, Any],
    workload_task_lookup: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    updated = dict(record or {})
    task = (
        workload_task_lookup.get(_clean_text(updated.get('phase_id')))
        or workload_task_lookup.get(_clean_text(updated.get('branch_id')))
        or workload_task_lookup.get(_clean_text(updated.get('task_id') or updated.get('workload_task_id')))
    )
    if not isinstance(task, Mapping):
        return updated
    for key in (
        'semantic_intent',
        'objective',
        'deliverable',
        'rationale',
        'advisory_role',
        'decision_notes',
        'promotion_policy',
        'reconsideration_policy',
        'review_criteria',
        'evidence_requirements',
        'reconsideration_triggers',
        'semantic_review_criteria',
        'promotion_suggestions',
        'waiver_candidates',
        'repair_candidates',
        'supersession_candidates',
        'learning_hint_refs',
        'input_refs',
        'execution_contract',
        'workload_task_ref',
        'output_obligation_ref',
        'output_contract',
        'accepted_proposals',
        'stage_direction',
        'content_payload_source',
        'artifact_prompt_source',
        'candidate_selection_index',
        'candidate_selection_count',
        'selection_policy',
        'selection_reason',
        'lang_code',
        'audio_variant_index',
        'audio_variant_role',
        'audio_variant_contract_source',
        'structured_output_contract',
        'branch_contract_error',
        'audio_variant_contract_conflicting_fields',
        'requires_artifact',
        'text_artifact_extension',
        'text_artifact_source_name',
        'text_artifact_source',
        'text_artifact_target_path',
        'artifact_request',
        'repair_action',
        'recovery_action',
        'repair_action_reason',
        'blocked_by_dependency_input',
        'blocked_by_branch_contract',
    ):
        value = task.get(key)
        if value not in (None, '', [], {}):
            if key == 'input_refs' and isinstance(value, list):
                merged_refs = [
                    dict(item)
                    for item in (updated.get('input_refs') or [])
                    if isinstance(item, Mapping)
                ]
                for item in value:
                    if isinstance(item, Mapping) and dict(item) not in merged_refs:
                        merged_refs.append(dict(item))
                updated[key] = merged_refs
            else:
                updated[key] = value
    task_id = _clean_text(task.get('task_id'))
    if task_id and updated.get('task_id') in (None, '', [], {}):
        updated['task_id'] = task_id
        updated['workload_task_id'] = task_id
    return updated


def _phase_id_for_capability(
    phase_graph: Optional[Mapping[str, Any]],
    capability: Any,
) -> Optional[str]:
    normalized_capability = normalize_capability(capability)
    if not normalized_capability:
        return None
    for phase in _phase_records(phase_graph):
        if normalize_capability(phase.get('capability')) != normalized_capability:
            continue
        phase_id = _clean_text(phase.get('phase_id'))
        if phase_id:
            return phase_id
    return None


def _phase_ids_for_capability(
    phase_graph: Optional[Mapping[str, Any]],
    capability: Any,
) -> list[str]:
    normalized_capability = normalize_capability(capability)
    if not normalized_capability:
        return []
    phase_ids: list[str] = []
    for phase in _phase_records(phase_graph):
        if normalize_capability(phase.get('capability')) != normalized_capability:
            continue
        phase_id = _clean_text(phase.get('phase_id'))
        if phase_id:
            phase_ids.append(phase_id)
    return phase_ids


def _completed_phase_ids(phase_graph: Optional[Mapping[str, Any]]) -> set[str]:
    completed: set[str] = set()
    for phase in _phase_records(phase_graph):
        phase_id = _clean_text(phase.get('phase_id'))
        status = _clean_text(phase.get('status')).lower()
        if phase_id and status == 'completed':
            completed.add(phase_id)
    return completed


def _failed_phase_ids(phase_graph: Optional[Mapping[str, Any]]) -> set[str]:
    failed: set[str] = set()
    for phase in _phase_records(phase_graph):
        phase_id = _clean_text(phase.get('phase_id'))
        status = _clean_text(phase.get('status')).lower()
        if phase_id and status in {'blocked', 'failed'}:
            failed.add(phase_id)
    return failed


def _downstream_capabilities(
    prompt_analysis: Mapping[str, Any],
    *,
    request_meta: Mapping[str, Any],
    request_payload: Mapping[str, Any],
    route_payload: Mapping[str, Any],
) -> list[str]:
    if bool(prompt_analysis.get('explicit_defer_materialization')) and not (
        bool(prompt_analysis.get('requests_audio_output'))
        or bool(prompt_analysis.get('requests_visual_output'))
        or bool(prompt_analysis.get('requests_speech_to_text_output'))
        or bool(prompt_analysis.get('has_audio_follow_up_request'))
        or bool(prompt_analysis.get('has_visual_follow_up_request'))
        or bool(prompt_analysis.get('text_preparation_before_audio_output'))
        or bool(prompt_analysis.get('text_preparation_before_visual_output'))
    ):
        return []
    wants_audio_follow_up = bool(
        prompt_analysis.get('requests_audio_output')
        or prompt_analysis.get('has_audio_follow_up_request')
        or prompt_analysis.get('text_preparation_before_audio_output')
    )
    primary_capability = normalize_capability(prompt_analysis.get('primary_capability'))
    wants_speech_to_text = bool(
        prompt_analysis.get('requests_speech_to_text_output')
        or primary_capability == CAPABILITY_SPEECH_TO_TEXT
        or (
            _has_audio_input_artifact(request_payload)
            and _prompt_references_current_input_audio(prompt_analysis)
        )
    )
    capabilities: list[Any] = []
    raw_list = (
        prompt_analysis.get('downstream_follow_up_capabilities')
        if isinstance(prompt_analysis.get('downstream_follow_up_capabilities'), list)
        else []
    )
    capabilities.extend(raw_list)
    explicit_client_batch_prompts = _clean_string_list(request_payload.get('batch_prompts'))
    explicit_client_image_batch_contract = len(explicit_client_batch_prompts) > 1
    ghost_owned_request = bool(request_payload.get('ghost_route')) or (
        _clean_text(route_payload.get('route_source')).lower() == 'ghost_carried'
    )
    requested_visual_output_count = max(
        _coerce_positive_int(prompt_analysis.get('requested_visual_output_count')),
        _coerce_positive_int(request_payload.get('batch_count')),
    )
    wants_visual_follow_up = bool(
        prompt_analysis.get('requests_visual_output')
        or prompt_analysis.get('has_visual_follow_up_request')
        or prompt_analysis.get('text_preparation_before_visual_output')
        or requested_visual_output_count > 0
    )
    hinted_capability = normalize_capability(request_meta.get('capability_hint'))
    route_capability = normalize_capability(route_payload.get('capability'))
    if (
        not explicit_client_image_batch_contract
        and requested_visual_output_count > 1
        and wants_visual_follow_up
    ):
        capabilities.append(CAPABILITY_IMAGE_GENERATION)
    if ghost_owned_request:
        if wants_audio_follow_up:
            appended_audio = False
            for candidate in (route_capability, hinted_capability, primary_capability):
                if candidate != CAPABILITY_TEXT_TO_SPEECH:
                    continue
                capabilities.append(candidate)
                appended_audio = True
                break
            if not appended_audio:
                capabilities.append(CAPABILITY_TEXT_TO_SPEECH)
        if wants_visual_follow_up:
            appended_visual = False
            for candidate in (route_capability, hinted_capability, primary_capability):
                if candidate != CAPABILITY_IMAGE_GENERATION:
                    continue
                if explicit_client_image_batch_contract:
                    continue
                capabilities.append(candidate)
                appended_visual = True
                break
            if not appended_visual and not explicit_client_image_batch_contract:
                capabilities.append(CAPABILITY_IMAGE_GENERATION)
        if wants_speech_to_text:
            appended_stt = False
            for candidate in (route_capability, hinted_capability, primary_capability):
                if candidate != CAPABILITY_SPEECH_TO_TEXT:
                    continue
                capabilities.append(candidate)
                appended_stt = True
                break
            if not appended_stt:
                capabilities.append(CAPABILITY_SPEECH_TO_TEXT)
    if (
        hinted_capability == CAPABILITY_TEXT_TO_SPEECH
        and wants_audio_follow_up
        and (not route_capability or route_capability == CAPABILITY_CHAT)
    ):
        capabilities.append(hinted_capability)
    if (
        hinted_capability == CAPABILITY_IMAGE_GENERATION
        and wants_visual_follow_up
        and (not route_capability or route_capability == CAPABILITY_CHAT)
    ):
        capabilities.append(hinted_capability)
    if (
        hinted_capability == CAPABILITY_SPEECH_TO_TEXT
        and wants_speech_to_text
        and (not route_capability or route_capability == CAPABILITY_CHAT)
    ):
        capabilities.append(hinted_capability)
    ordered_capabilities = _unique_capabilities(capabilities)
    if (
        CAPABILITY_SPEECH_TO_TEXT in ordered_capabilities
        and _has_audio_input_artifact(request_payload)
    ):
        ordered_capabilities = [
            CAPABILITY_SPEECH_TO_TEXT,
            *[
                capability
                for capability in ordered_capabilities
                if capability != CAPABILITY_SPEECH_TO_TEXT
            ],
        ]
    return [
        capability
        for capability in ordered_capabilities
        if not _prompt_negates_materialization_capability(prompt_analysis, capability)
        and not _prompt_reserves_entire_materialization_capability(prompt_analysis, capability)
        and not _visual_execution_is_preserved(prompt_analysis, capability)
        and not (
            capability == CAPABILITY_TEXT_TO_SPEECH
            and bool(prompt_analysis.get('audio_output_count_exceeds_bound'))
        )
    ]


def _speech_to_text_result_feeds_capability(
    prompt_analysis: Mapping[str, Any],
    capability: Any,
) -> bool:
    normalized_capability = normalize_capability(capability)
    if normalized_capability != CAPABILITY_TEXT_TO_SPEECH:
        return False
    prompt_text = _clean_text(prompt_analysis.get('normalized_prompt'))
    return bool(
        prompt_analysis.get('text_preparation_before_audio_output')
        or _STT_RESULT_FEEDS_AUDIO_RE.search(prompt_text)
    )


def _image_materialization_negation_match_is_quality_or_path_constraint(
    prompt_text: str,
    start: int,
    end: int,
) -> bool:
    match_text = str(prompt_text or '')[max(0, int(start or 0)):max(0, int(end or 0))]
    if not match_text:
        return False
    if _IMAGE_MATERIALIZATION_VERB_RE.search(match_text):
        return False
    if (
        _DIRECT_IMAGE_NEGATION_RE.search(match_text)
        and not _IMAGE_QUALITY_NEGATION_CONTEXT_RE.search(match_text)
        and not _IMAGE_PATH_OR_LINK_CONSTRAINT_RE.search(match_text)
    ):
        return False
    return bool(
        _IMAGE_QUALITY_NEGATION_CONTEXT_RE.search(match_text)
        or _IMAGE_PATH_OR_LINK_CONSTRAINT_RE.search(match_text)
    )


def _prompt_negates_materialization_capability(
    prompt_analysis: Mapping[str, Any],
    capability: Any,
) -> bool:
    normalized = normalize_capability(capability)
    prompt_text = _clean_text(prompt_analysis.get('normalized_prompt'))
    if not prompt_text:
        return False
    if normalized == CAPABILITY_IMAGE_GENERATION:
        if bool(prompt_analysis.get('separate_visual_generation_request')):
            return False
        pattern = _NEGATED_IMAGE_MATERIALIZATION_RE
    elif normalized == CAPABILITY_TEXT_TO_SPEECH:
        pattern = _NEGATED_AUDIO_MATERIALIZATION_RE
    else:
        return False
    for match in pattern.finditer(prompt_text):
        if materialization_negation_match_is_artifact_fulfillment_only(
            prompt_text,
            match.start(),
            match.end(),
        ):
            continue
        if materialization_negation_match_is_output_contrast(
            prompt_text,
            match.start(),
            match.end(),
        ):
            continue
        if (
            normalized == CAPABILITY_IMAGE_GENERATION
            and _image_materialization_negation_match_is_quality_or_path_constraint(
                prompt_text,
                match.start(),
                match.end(),
            )
        ):
            continue
        return True
    return False


def _prompt_reserves_materialization_capability(
    prompt_analysis: Mapping[str, Any],
    capability: Any,
) -> bool:
    normalized = normalize_capability(capability)
    prompt_text = _clean_text(prompt_analysis.get('normalized_prompt'))
    if not prompt_text:
        return False
    if normalized == CAPABILITY_IMAGE_GENERATION:
        return bool(
            _RESERVED_IMAGE_MATERIALIZATION_RE.search(prompt_text)
            or (
                _IMAGE_CANDIDATE_CONTEXT_RE.search(prompt_text)
                and _RESERVED_OPTION_ONLY_RE.search(prompt_text)
            )
        )
    return False


def _selected_image_candidate_index(prompt_analysis: Mapping[str, Any]) -> int:
    prompt_text = _clean_text(prompt_analysis.get('normalized_prompt'))
    if not prompt_text or not _IMAGE_CANDIDATE_CONTEXT_RE.search(prompt_text):
        return 0
    match = _SELECTED_IMAGE_CANDIDATE_INDEX_RE.search(prompt_text)
    if not match:
        return 0
    return _ORDINAL_INDEXES.get(_clean_text(match.group('ordinal')).lower(), 0)


def _selected_speakable_candidate_fields(prompt_analysis: Mapping[str, Any]) -> dict[str, Any]:
    prompt_text = _clean_text(prompt_analysis.get('normalized_prompt'))
    if not prompt_text or not _SELECTED_SPEAKABLE_CANDIDATE_RE.search(prompt_text):
        return {}
    match = _SELECTED_SPEAKABLE_CANDIDATE_INDEX_RE.search(prompt_text)
    ordinal = ''
    if match:
        ordinal = _clean_text(match.group('ordinal') or match.group('ordinal_prefix')).lower()
    index = _ORDINAL_INDEXES.get(ordinal, 0) if ordinal else 0
    return {
        'candidate_selection_index': index or 1,
        'selection_policy': 'selected_candidate_only' if index else 'best_candidate_only',
        'selection_reason': 'speak only the requested best/selected candidate',
        'content_payload_source': 'selected_candidate_from_phase_output',
    }


def _counted_audio_variant_selection_fields(
    requested_count: int,
    occurrence: int,
    prompt_analysis: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Bind each counted audio branch to one independently prepared text variant."""

    if requested_count <= 1 or occurrence <= 0 or occurrence > requested_count:
        return {}
    fields: dict[str, Any] = {
        'candidate_selection_index': occurrence,
        'candidate_selection_count': requested_count,
        'selection_policy': 'selected_candidate_only',
        'selection_reason': 'materialize the independently requested audio text variant',
        'content_payload_source': 'selected_candidate_from_phase_output',
        'stage_direction': f'materialize_requested_audio_variant_{occurrence}',
        'audio_variant_index': occurrence,
        'audio_variant_role': 'requested_audio_variant',
    }
    analysis = prompt_analysis if isinstance(prompt_analysis, Mapping) else {}
    prompt_text = _clean_text(analysis.get('normalized_prompt'))
    explicit_contracts: list[tuple[int, str, str]] = []
    german_original = _GERMAN_ORIGINAL_AUDIO_VARIANT_RE.search(prompt_text)
    if german_original:
        explicit_contracts.append((german_original.start(), 'de', 'original_narration'))
    english_translation = _ENGLISH_TRANSLATION_AUDIO_VARIANT_RE.search(prompt_text)
    if english_translation:
        explicit_contracts.append((english_translation.start(), 'en', 'faithful_translation'))
    explicit_contracts.sort(key=lambda item: item[0])
    if len(explicit_contracts) != requested_count:
        explicit_contracts = [
            (
                match.start(),
                'en' if match.group('english') else 'de',
                'language_variant',
            )
            for match in _AUDIO_VARIANT_LANGUAGE_MENTION_RE.finditer(prompt_text)
        ]
    if len(explicit_contracts) == requested_count and occurrence <= len(explicit_contracts):
        _, lang_code, role = explicit_contracts[occurrence - 1]
        fields.update(
            {
                'lang_code': lang_code,
                'audio_variant_role': role,
                'audio_variant_contract_source': 'explicit_language_role_sequence',
            }
        )
    return fields


def _retain_counted_audio_variant_contracts(
    branches: list[dict[str, Any]],
    *,
    prompt_analysis: Mapping[str, Any],
) -> None:
    """Restore deterministic variant identity on explicit/refined TTS branches.

    Explicit planner branches outrank synthesized branches, but older or reduced
    planner projections may omit the selection fields derived from the current
    prompt.  Missing fields are recoverable from current intent; conflicting
    fields are not and therefore fail closed at the branch contract boundary.
    """

    requested_count = _requested_audio_output_count(prompt_analysis)
    if requested_count <= 1:
        return
    tts_branches = [
        branch
        for branch in branches
        if normalize_capability(branch.get('capability')) == CAPABILITY_TEXT_TO_SPEECH
        and not _is_unpromoted_candidate_record(branch)
    ]
    if len(tts_branches) != requested_count:
        return

    ordered = sorted(
        enumerate(tts_branches, start=1),
        key=lambda item: (
            _coerce_positive_int(item[1].get('queue_index')) or item[0],
            item[0],
        ),
    )
    for occurrence, (_position, branch) in enumerate(ordered, start=1):
        expected = _counted_audio_variant_selection_fields(
            requested_count,
            occurrence,
            prompt_analysis,
        )
        authoritative_fields = {
            'candidate_selection_index',
            'candidate_selection_count',
            'selection_policy',
            'content_payload_source',
            'stage_direction',
            'lang_code',
            'audio_variant_index',
            'audio_variant_role',
            'audio_variant_contract_source',
        }

        def values_conflict(key: str, actual: Any, wanted: Any) -> bool:
            if key in {
                'candidate_selection_index',
                'candidate_selection_count',
                'audio_variant_index',
            }:
                return _coerce_positive_int(actual) != _coerce_positive_int(wanted)
            if isinstance(actual, str) and isinstance(wanted, str):
                return actual.strip().casefold() != wanted.strip().casefold()
            return actual != wanted

        conflicting_fields = [
            key
            for key, expected_value in expected.items()
            if key in authoritative_fields
            if branch.get(key) not in (None, '', [], {})
            and values_conflict(key, branch.get(key), expected_value)
        ]
        if conflicting_fields:
            branch['branch_contract_error'] = 'ambiguous_audio_variant_contract'
            branch['blocked_by_branch_contract'] = True
            branch['repair_action'] = 'repair_branch_contract'
            branch['resolution'] = 'blocked_branch_contract'
            branch['audio_variant_contract_conflicting_fields'] = conflicting_fields
            continue
        for key, value in expected.items():
            if branch.get(key) in (None, '', [], {}):
                branch[key] = value


def _prompt_reserves_entire_materialization_capability(
    prompt_analysis: Mapping[str, Any],
    capability: Any,
) -> bool:
    if not _prompt_reserves_materialization_capability(prompt_analysis, capability):
        return False
    if normalize_capability(capability) == CAPABILITY_IMAGE_GENERATION:
        return _selected_image_candidate_index(prompt_analysis) <= 0
    return True


def _post_artifact_continuation_sequence(prompt_analysis: Mapping[str, Any]) -> list[str]:
    """Return explicit dependent phase order requested by the current prompt."""

    if bool(prompt_analysis.get('explicit_defer_materialization')) and not (
        bool(prompt_analysis.get('requests_audio_output'))
        or bool(prompt_analysis.get('requests_visual_output'))
        or bool(prompt_analysis.get('requests_speech_to_text_output'))
        or bool(prompt_analysis.get('has_audio_follow_up_request'))
        or bool(prompt_analysis.get('has_visual_follow_up_request'))
        or bool(prompt_analysis.get('text_preparation_before_audio_output'))
        or bool(prompt_analysis.get('text_preparation_before_visual_output'))
    ):
        return []
    if bool(prompt_analysis.get('meta_execution_explanation_request')):
        return []
    prompt_text = _clean_text(prompt_analysis.get('normalized_prompt'))
    if not prompt_text:
        return []
    if not (
        _DEPENDENT_CHAIN_MARKER_RE.search(prompt_text)
        or _POST_IMAGE_TEXT_RE.search(prompt_text)
        or _POST_AUDIO_TEXT_RE.search(prompt_text)
        or _POST_GENERATED_ARTIFACT_TEXT_RE.search(prompt_text)
        or _POST_TEXT_TO_AUDIO_RE.search(prompt_text)
        or _POST_TEXT_TO_IMAGE_RE.search(prompt_text)
        or _ANALYZE_IMAGE_ACTION_RE.search(prompt_text)
        or _VISUAL_EVIDENCE_TEXT_TRIGGER_RE.search(prompt_text)
    ):
        return []
    action_prompt_text = mask_intent_literal_payloads(prompt_text)

    actions: list[tuple[int, int, str]] = []
    for capability, pattern in (
        (CAPABILITY_CHAT, _TEXT_OUTPUT_ACTION_RE),
        (CAPABILITY_IMAGE_GENERATION, _GENERATE_IMAGE_ACTION_RE),
        (CAPABILITY_TEXT_TO_SPEECH, _GENERATE_AUDIO_ACTION_RE),
        (CAPABILITY_SPEECH_TO_TEXT, _TRANSCRIBE_ACTION_RE),
        (CAPABILITY_VISION_ANALYSIS, _ANALYZE_IMAGE_ACTION_RE),
    ):
        if _visual_execution_is_preserved(prompt_analysis, capability):
            continue
        if (
            capability == CAPABILITY_IMAGE_GENERATION
            and bool(prompt_analysis.get('explicit_visual_defer_materialization'))
        ):
            continue
        if (
            capability == CAPABILITY_TEXT_TO_SPEECH
            and bool(prompt_analysis.get('explicit_audio_defer_materialization'))
        ):
            continue
        if (
            capability == CAPABILITY_TEXT_TO_SPEECH
            and bool(prompt_analysis.get('audio_output_count_exceeds_bound'))
        ):
            continue
        for match in pattern.finditer(action_prompt_text):
            matched_text = prompt_text[match.start():match.end()].lower()
            if (
                capability == CAPABILITY_IMAGE_GENERATION
                and visual_action_is_negated(
                    prompt_text,
                    match.start(),
                    match.end(),
                )
            ):
                continue
            if _prompt_negates_materialization_capability(
                {'normalized_prompt': matched_text},
                capability,
            ):
                continue
            if _prompt_reserves_materialization_capability(
                {'normalized_prompt': matched_text},
                capability,
            ):
                continue
            start = match.start()
            if capability in {CAPABILITY_IMAGE_GENERATION, CAPABILITY_TEXT_TO_SPEECH}:
                verb_matches = list(_MATERIALIZATION_ACTION_VERB_RE.finditer(matched_text))
                if len(verb_matches) > 1:
                    start = match.start() + verb_matches[-1].start()
            if capability == CAPABILITY_IMAGE_GENERATION:
                additive_match = re.search(r'\b(?:also|additionally|zusatzlich|zusaetzlich|zusätzlich)\b', matched_text)
                if additive_match:
                    start = match.start() + additive_match.start()
            actions.append((start, match.end(), capability))
    if not actions:
        return []

    actions.sort(key=lambda item: (item[0], item[1]))
    sequence: list[str] = []
    last_end = -1
    last_start = -1
    last_capability = ''
    for start, end, capability in actions:
        same_verb_media_pair = (
            start == last_start
            and capability != last_capability
            and capability in {CAPABILITY_IMAGE_GENERATION, CAPABILITY_TEXT_TO_SPEECH}
            and last_capability in {CAPABILITY_IMAGE_GENERATION, CAPABILITY_TEXT_TO_SPEECH}
        )
        if start < last_end and not same_verb_media_pair:
            continue
        if sequence and sequence[-1] == capability:
            last_end = max(last_end, end)
            last_start = start
            last_capability = capability
            continue
        sequence.append(capability)
        last_end = max(last_end, end)
        last_start = start
        last_capability = capability

    if sequence and sequence[0] == CAPABILITY_CHAT:
        sequence = sequence[1:]
    if len(sequence) < 2:
        return []
    return _sequence_with_media_analysis_dependencies(sequence, prompt_text)


def _sequence_binds_stt_to_generated_audio(sequence: list[Any]) -> bool:
    """Return whether a requested TTS producer precedes an STT consumer."""

    has_audio_producer = False
    for item in sequence:
        capability = normalize_capability(item)
        if capability == CAPABILITY_TEXT_TO_SPEECH:
            has_audio_producer = True
        elif capability == CAPABILITY_SPEECH_TO_TEXT and has_audio_producer:
            return True
    return False


def _prompt_targets_direct_input_audio_for_stt(
    prompt_analysis: Mapping[str, Any],
) -> bool:
    prompt_text = _clean_text(prompt_analysis.get('normalized_prompt'))
    return bool(
        prompt_text
        and _DIRECT_INPUT_AUDIO_TRANSCRIPTION_TARGET_RE.search(prompt_text)
    )


def _stt_source_contract_around_tts(
    prompt_analysis: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    """Return pre-TTS STT presence and ordered source classes after TTS."""

    prompt_text = _clean_text(prompt_analysis.get('normalized_prompt'))
    if not prompt_text:
        return False, []
    producers = list(_GENERATE_AUDIO_ACTION_RE.finditer(prompt_text))
    transcriptions = list(_TRANSCRIBE_ACTION_RE.finditer(prompt_text))
    if not producers or not transcriptions:
        return False, []
    action_starts = sorted(
        {
            match.start()
            for match in [*producers, *transcriptions]
        }
    )
    has_pre_tts_transcription = any(
        transcription.start() < min(producer.start() for producer in producers)
        for transcription in transcriptions
    )
    source_sequence: list[str] = []
    for transcription in transcriptions:
        if not any(producer.end() <= transcription.start() for producer in producers):
            continue
        segment_end = next(
            (
                action_start
                for action_start in action_starts
                if action_start > transcription.start()
            ),
            len(prompt_text),
        )
        action_segment = prompt_text[transcription.start():segment_end]
        if _DIRECT_INPUT_AUDIO_TRANSCRIPTION_TARGET_RE.search(action_segment):
            source_sequence.append(
                'selected_reference'
                if _SELECTED_AUDIO_TARGET_QUALIFIER_RE.search(action_segment)
                else 'current_input'
            )
        else:
            source_sequence.append('generated_audio')
    return has_pre_tts_transcription, source_sequence


def _post_tts_stt_source_targets(
    prompt_analysis: Mapping[str, Any],
) -> tuple[bool, bool]:
    """Return direct-input and generated-audio STT targets after a TTS action."""

    _has_pre_tts_transcription, source_sequence = _stt_source_contract_around_tts(
        prompt_analysis
    )
    return (
        any(source != 'generated_audio' for source in source_sequence),
        'generated_audio' in source_sequence,
    )


def _bind_post_tts_direct_input_stt_to_request_audio(
    branches: list[dict[str, Any]],
    *,
    prompt_analysis: Mapping[str, Any],
    request_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Keep an explicitly reselected request audio out of a generated-audio edge."""

    has_pre_tts_transcription, source_sequence = _stt_source_contract_around_tts(
        prompt_analysis
    )
    if not any(source != 'generated_audio' for source in source_sequence):
        return branches
    leading_stt_phase_id = next(
        (
            _clean_text(branch.get('phase_id'))
            for branch in branches
            if normalize_capability(branch.get('capability')) == CAPABILITY_SPEECH_TO_TEXT
        ),
        '',
    )
    if not leading_stt_phase_id:
        return branches
    branch_source_sequence = (
        list(source_sequence)
        if has_pre_tts_transcription
        else [source for source in source_sequence if source == 'generated_audio']
    )
    if not branch_source_sequence:
        return branches
    tts_phase_ids: list[str] = []
    post_tts_stt_index = 0
    if not any(
        normalize_capability(branch.get('capability')) == CAPABILITY_TEXT_TO_SPEECH
        for branch in branches
    ):
        return branches
    updated: list[dict[str, Any]] = []
    for branch in branches:
        record = dict(branch)
        capability = normalize_capability(record.get('capability'))
        phase_id = _clean_text(record.get('phase_id'))
        if capability == CAPABILITY_TEXT_TO_SPEECH and phase_id:
            tts_phase_ids.append(phase_id)
        elif (
            capability == CAPABILITY_SPEECH_TO_TEXT
            and phase_id != leading_stt_phase_id
            and tts_phase_ids
        ):
            source_class = branch_source_sequence[
                min(post_tts_stt_index, len(branch_source_sequence) - 1)
            ]
            post_tts_stt_index += 1
            if source_class != 'generated_audio':
                record['depends_on'] = [leading_stt_phase_id]
                current_input_available = _has_current_audio_input_artifact(request_payload)
                selected_reference_available = _has_selected_reference_audio_artifact(
                    request_payload
                )
                prefers_selected_reference = source_class == 'selected_reference'
                bind_selected_reference = (
                    selected_reference_available
                    and (prefers_selected_reference or not current_input_available)
                )
                record['content_payload_source'] = (
                    'selected_reference_audio_artifact'
                    if bind_selected_reference
                    else 'current_input_audio_artifact'
                )
            else:
                depends_on = {
                    _clean_text(item)
                    for item in (record.get('depends_on') or [])
                    if _clean_text(item)
                }
                if not depends_on.intersection(tts_phase_ids):
                    record['depends_on'] = [tts_phase_ids[-1]]
                if _clean_text(record.get('content_payload_source')) in {
                    'current_input_audio_artifact',
                    'selected_reference_audio_artifact',
                }:
                    record.pop('content_payload_source', None)
        updated.append(record)
    return updated


def _sequence_with_media_analysis_dependencies(sequence: list[str], prompt_text: str) -> list[str]:
    """Insert evidence-extraction phases needed by later text branches."""

    enriched: list[str] = []
    prompt = _clean_text(prompt_text)
    artifact_inventory_only = bool(_POST_GENERATED_ARTIFACT_INVENTORY_RE.search(prompt))
    for capability in sequence:
        normalized = normalize_capability(capability)
        if not normalized:
            continue
        previous = enriched[-1] if enriched else ''
        if (
            normalized == CAPABILITY_CHAT
            and previous == CAPABILITY_IMAGE_GENERATION
            and not artifact_inventory_only
            and CAPABILITY_VISION_ANALYSIS not in enriched[-1:]
        ):
            enriched.append(CAPABILITY_VISION_ANALYSIS)
        if (
            normalized == CAPABILITY_CHAT
            and previous == CAPABILITY_TEXT_TO_SPEECH
            and _POST_AUDIO_TEXT_REQUIRES_STT_RE.search(prompt)
        ):
            enriched.append(CAPABILITY_SPEECH_TO_TEXT)
            if not _POST_AUDIO_TEXT_REQUIRES_CHAT_AFTER_STT_RE.search(prompt):
                continue
        if enriched and enriched[-1] == normalized:
            continue
        enriched.append(normalized)
    if (
        enriched
        and enriched[-1] == CAPABILITY_VISION_ANALYSIS
        and _POST_IMAGE_TEXT_RE.search(prompt)
    ):
        enriched.append(CAPABILITY_CHAT)
    return enriched


def _next_phase_number(branches: list[Mapping[str, Any]]) -> int:
    highest = 1
    for branch in branches:
        phase_id = _clean_text(branch.get('phase_id') or branch.get('branch_id'))
        match = re.fullmatch(r'phase-(\d+)', phase_id)
        if not match:
            continue
        try:
            highest = max(highest, int(match.group(1)))
        except ValueError:
            continue
    return highest + 1


def _text_artifact_request_branches(
    text_artifact_requests: list[dict[str, str]],
    *,
    existing_branches: list[Mapping[str, Any]],
    prompt_analysis: Optional[Mapping[str, Any]] = None,
) -> list[dict[str, Any]]:
    branches: list[dict[str, Any]] = []
    if not text_artifact_requests:
        return branches
    remaining_requests = list(text_artifact_requests)
    json_request_indexes = [
        index
        for index, request in enumerate(text_artifact_requests)
        if _clean_text(request.get('extension')).lower() == 'json'
    ]
    if len(json_request_indexes) == 1:
        json_request_index = json_request_indexes[0]
        request = text_artifact_requests[json_request_index]
        extension = 'json'
        structured_final_join_branches = [
            branch
            for branch in existing_branches
            if normalize_capability(branch.get('capability')) == CAPABILITY_CHAT
            and _clean_text(branch.get('dependency_contract')).lower()
            == 'structured_multi_evidence_join'
        ]
        terminal_json_branches = [
            branch
            for branch in existing_branches
            if normalize_capability(branch.get('capability')) == CAPABILITY_CHAT
            and _clean_text(branch.get('role')).lower() == 'post_artifact_text_follow_up'
            and any(
                _clean_text(dependency) != 'phase-1'
                for dependency in (branch.get('depends_on') or [])
                if _clean_text(dependency)
            )
        ]
        artifact_target = (
            structured_final_join_branches[0]
            if len(structured_final_join_branches) == 1
            else (
                terminal_json_branches[0]
                if len(terminal_json_branches) == 1
                else None
            )
        )
        if artifact_target is not None:
            source_name = _clean_text(request.get('source_name')) or 'generated-json'
            source = _clean_text(request.get('source')) or 'explicit_text_artifact_request'
            target_path = _clean_text(request.get('target_path'))
            artifact_request = {
                'extension': extension,
                'source_name': source_name,
                'source': source,
                **({'target_path': target_path} if target_path else {}),
            }
            artifact_target.update(
                {
                    'kind': 'materialize',
                    'required': True,
                    'requires_artifact': True,
                    'text_artifact_extension': extension,
                    'text_artifact_source_name': source_name,
                    'text_artifact_source': source,
                    'artifact_request': artifact_request,
                    'stage_direction': 'materialize_requested_json_after_artifact_evidence',
                    'phase_summary': f'materialize {extension} artifact {source_name}',
                }
            )
            if target_path:
                artifact_target['text_artifact_target_path'] = target_path
            remaining_requests = [
                candidate
                for index, candidate in enumerate(text_artifact_requests)
                if index != json_request_index
            ]
    next_phase_number = _next_phase_number(existing_branches)
    analysis = prompt_analysis if isinstance(prompt_analysis, Mapping) else {}
    local_visual_asset_binding_required = bool(analysis.get('local_visual_asset_requirement'))
    linked_image_phase_ids = [
        _clean_text(branch.get('phase_id'))
        for branch in existing_branches
        if normalize_capability(branch.get('capability')) == CAPABILITY_IMAGE_GENERATION
        and _clean_text(branch.get('phase_id'))
        and _clean_text(branch.get('contract_state')).lower() != 'reserved'
    ]
    for index, request in enumerate(remaining_requests, start=1):
        extension = _clean_text(request.get('extension')).lower() or 'txt'
        source_name = _clean_text(request.get('source_name')) or f'generated-{extension}'
        source = _clean_text(request.get('source')) or 'explicit_text_artifact_request'
        target_path = _clean_text(request.get('target_path'))
        phase_id = f'phase-{next_phase_number + index - 1}'
        depends_on = ['phase-1']
        image_link_text_artifact = extension in {'html', 'htm', 'css'}
        if local_visual_asset_binding_required and image_link_text_artifact and linked_image_phase_ids:
            depends_on = list(linked_image_phase_ids)
        branch_record = {
            'branch_id': f'branch-text_artifact-{index}',
            'phase_id': phase_id,
            'capability': CAPABILITY_CHAT,
            'output_type': 'text',
            'depends_on': depends_on,
            'queue_index': index,
            'source': 'text_artifact_request',
            'resolution': 'pending_dependency',
            'kind': 'materialize',
            'role': 'text_artifact_output',
            'required': True,
            'requires_artifact': True,
            'text_artifact_extension': extension,
            'text_artifact_source_name': source_name,
            'text_artifact_source': source,
            'artifact_request': {
                'extension': extension,
                'source_name': source_name,
                'source': source,
                **({'target_path': target_path} if target_path else {}),
            },
            'content_payload_source': 'current_phase_output',
            'stage_direction': 'materialize_requested_text_artifact',
            'phase_summary': f'materialize {extension} artifact {source_name}',
        }
        if target_path:
            branch_record['text_artifact_target_path'] = target_path
        if depends_on != ['phase-1']:
            branch_record.update(
                {
                    'dependency_contract': 'local_visual_asset_binding',
                    'image_asset_binding_required': True,
                    'required_image_phase_ids': list(linked_image_phase_ids),
                }
            )
        branches.append(
            branch_record
        )
    return branches


def _branch_has_executable_contract_authority(branch: Mapping[str, Any]) -> bool:
    """Accept only an active/promoted branch contract with well-formed state."""

    if _is_unpromoted_candidate_record(branch):
        return False
    if 'required' in branch and _explicit_required_flag(branch) is not True:
        return False

    authority_seen = False
    for field in (
        'contract_state',
        'contract_status',
        'obligation_state',
        'intent_state',
        'status',
    ):
        if field not in branch:
            continue
        value = branch.get(field)
        if not isinstance(value, str):
            return False
        state = value.strip().lower()
        if not state or state not in _EXECUTABLE_BRANCH_CONTRACT_STATES:
            return False
        authority_seen = True

    if 'resolution' in branch:
        value = branch.get('resolution')
        if not isinstance(value, str):
            return False
        resolution = value.strip().lower()
        if not resolution or resolution not in _EXECUTABLE_BRANCH_RESOLUTIONS:
            return False
        authority_seen = True

    return authority_seen


def _explicit_producer_order_index(
    branch: Mapping[str, Any],
) -> tuple[Optional[int], bool]:
    """Return one consistent positive queue/candidate index and its validity."""

    indexes: list[int] = []
    for field in ('candidate_selection_index', 'queue_index'):
        if field not in branch:
            continue
        raw_index = branch.get(field)
        if isinstance(raw_index, bool):
            return None, False
        if isinstance(raw_index, int):
            index = raw_index
        elif isinstance(raw_index, str) and re.fullmatch(r'[1-9]\d*', raw_index.strip()):
            index = int(raw_index.strip())
        else:
            return None, False
        if index <= 0:
            return None, False
        indexes.append(index)
    if not indexes:
        return None, True
    if len(set(indexes)) != 1:
        return None, False
    return indexes[0], True


def _structured_final_join_selected_phase_ids(
    branches: list[dict[str, Any]],
    final_contract: str,
) -> list[str]:
    """Resolve one unambiguous producer/evidence pair per media family."""

    selections: dict[str, int] = {}
    for match in _FINAL_JOIN_SELECTED_MEDIA_RE.finditer(final_contract):
        selection_prefix = final_contract[max(0, match.start() - 24):match.start()]
        if _FINAL_JOIN_SELECTOR_NEGATION_RE.search(selection_prefix):
            return []
        ordinal = _clean_text(match.group('ordinal')).lower()
        if ordinal.startswith('erst'):
            ordinal = 'erste'
        elif ordinal.startswith('zweit'):
            ordinal = 'zweite'
        selected_index = _ORDINAL_INDEXES.get(ordinal, 0)
        producer_capability = (
            CAPABILITY_IMAGE_GENERATION
            if _clean_text(match.group('media')).lower().startswith(('image', 'bild'))
            else CAPABILITY_TEXT_TO_SPEECH
        )
        if selected_index <= 0:
            return []
        previous_selection = selections.get(producer_capability)
        if previous_selection and previous_selection != selected_index:
            return []
        selections[producer_capability] = selected_index

    selected_phase_ids: set[str] = set()
    for producer_capability, evidence_capability in (
        (CAPABILITY_IMAGE_GENERATION, CAPABILITY_VISION_ANALYSIS),
        (CAPABILITY_TEXT_TO_SPEECH, CAPABILITY_SPEECH_TO_TEXT),
    ):
        producers = [
            branch
            for branch in branches
            if normalize_capability(branch.get('capability')) == producer_capability
        ]
        if (
            not producers
            or any(
                not _clean_text(branch.get('phase_id'))
                or not _branch_has_executable_contract_authority(branch)
                for branch in producers
            )
        ):
            return []
        if len(producers) > 1:
            indexed_producers: list[tuple[int, dict[str, Any]]] = []
            for producer in producers:
                producer_index, valid_index = _explicit_producer_order_index(producer)
                if not valid_index or producer_index is None:
                    return []
                indexed_producers.append((producer_index, producer))
            if len({index for index, _producer in indexed_producers}) != len(producers):
                return []
            producers = [
                producer
                for _index, producer in sorted(
                    indexed_producers,
                    key=lambda item: item[0],
                )
            ]
        selected_index = selections.get(producer_capability)
        if selected_index is None:
            if len(producers) != 1:
                return []
            selected_index = 1
        if selected_index > len(producers):
            return []
        producer = producers[selected_index - 1]
        producer_tokens = {
            _clean_text(producer.get('phase_id')),
            _clean_text(producer.get('branch_id')),
        }
        family_producer_tokens = {
            token
            for item in producers
            for token in (
                _clean_text(item.get('phase_id')),
                _clean_text(item.get('branch_id')),
            )
            if token
        }
        family_evidence_branches = [
            branch
            for branch in branches
            if normalize_capability(branch.get('capability')) == evidence_capability
            and {
                _clean_text(dependency)
                for dependency in (branch.get('depends_on') or [])
                if _clean_text(dependency)
            }.intersection(family_producer_tokens)
        ]
        if any(
            not _clean_text(branch.get('phase_id'))
            or not _branch_has_executable_contract_authority(branch)
            for branch in family_evidence_branches
        ):
            return []
        evidence_branches = [
            branch
            for branch in family_evidence_branches
            if _evidence_branch_binds_only_selected_producer(
                branch,
                selected_producer_tokens=producer_tokens,
                family_producer_tokens=family_producer_tokens,
            )
            and _clean_text(branch.get('phase_id'))
        ]
        if len(evidence_branches) != 1:
            return []
        selected_phase_ids.update(
            {
                _clean_text(producer.get('phase_id')),
                _clean_text(evidence_branches[0].get('phase_id')),
            }
        )

    return [
        _clean_text(branch.get('phase_id'))
        for branch in branches
        if _clean_text(branch.get('phase_id')) in selected_phase_ids
    ]


def _evidence_branch_binds_only_selected_producer(
    branch: Mapping[str, Any],
    *,
    selected_producer_tokens: set[str],
    family_producer_tokens: set[str],
) -> bool:
    """Require evidence to depend on the selection and no sibling producer."""

    dependencies = {
        _clean_text(dependency)
        for dependency in (branch.get('depends_on') or [])
        if _clean_text(dependency)
    }
    return bool(dependencies.intersection(selected_producer_tokens)) and not bool(
        dependencies.intersection(family_producer_tokens - selected_producer_tokens)
    )


def _structured_final_join_span_is_quoted(
    prompt_text: str,
    start: int,
    end: int,
) -> bool:
    """Return whether the exact matched join span sits inside one bounded quote."""

    return intent_span_is_literal_payload(prompt_text, start, end)


def _structured_final_join_binding_is_negated(
    prompt_text: str,
    binding_match: re.Match[str],
) -> bool:
    """Recognize only negation immediately governing the matched binding verb."""

    prefix = prompt_text[max(0, binding_match.start() - 48):binding_match.start()]
    suffix = prompt_text[binding_match.end():min(len(prompt_text), binding_match.end() + 64)]
    return bool(
        _FINAL_JOIN_BINDING_PREFIX_NEGATION_RE.search(prefix)
        or _FINAL_JOIN_BINDING_SUFFIX_NEGATION_RE.search(suffix)
    )


def _structured_final_join_requests_all_audio_pairs(final_contract: str) -> bool:
    """Return whether the final structure explicitly binds every audio pair."""

    audio_match = _AUDIO_ARTIFACT_REF_RE.search(final_contract)
    transcript_match = _TRANSCRIPT_JOIN_RE.search(final_contract)
    if not audio_match or not transcript_match:
        return False
    for match in (audio_match, transcript_match):
        nearby = final_contract[max(0, match.start() - 72):match.end()]
        if not _FINAL_JOIN_PLURAL_BINDING_RE.search(nearby):
            return False
    return True


def _structured_final_join_all_audio_pair_contract(
    branches: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Bind every unambiguous TTS producer to its single STT consumer."""

    producers = [
        branch
        for branch in branches
        if normalize_capability(branch.get('capability')) == CAPABILITY_TEXT_TO_SPEECH
        and not _is_unpromoted_candidate_record(branch)
    ]
    if len(producers) < 2:
        return [], []
    indexed: list[tuple[int, dict[str, Any]]] = []
    for producer in producers:
        producer_index, valid_index = _explicit_producer_order_index(producer)
        if not valid_index or producer_index is None:
            return [], []
        declared_count = _coerce_positive_int(
            producer.get('candidate_selection_count')
        )
        declared_variant_index = _coerce_positive_int(
            producer.get('audio_variant_index')
        )
        if (
            declared_count != len(producers)
            or declared_variant_index != producer_index
        ):
            return [], []
        if not _branch_has_executable_contract_authority(producer):
            return [], []
        indexed.append((producer_index, producer))
    if len({index for index, _producer in indexed}) != len(indexed):
        return [], []
    indexed.sort(key=lambda item: item[0])

    producer_index_by_token: dict[str, int] = {}
    for variant_index, producer in indexed:
        tokens = {
            _clean_text(producer.get('phase_id')),
            _clean_text(producer.get('branch_id')),
        } - {''}
        if len(tokens) < 2:
            return [], []
        for token in tokens:
            previous_index = producer_index_by_token.get(token)
            if previous_index is not None and previous_index != variant_index:
                return [], []
            producer_index_by_token[token] = variant_index

    consumers_by_variant: dict[int, list[dict[str, Any]]] = {
        variant_index: [] for variant_index, _producer in indexed
    }
    seen_consumer_phase_ids: set[str] = set()
    seen_consumer_branch_ids: set[str] = set()
    for consumer in branches:
        if (
            normalize_capability(consumer.get('capability'))
            != CAPABILITY_SPEECH_TO_TEXT
            or _is_unpromoted_candidate_record(consumer)
        ):
            continue
        dependencies = {
            _clean_text(dependency)
            for dependency in (consumer.get('depends_on') or [])
            if _clean_text(dependency)
        }
        matched_variants = {
            producer_index_by_token[dependency]
            for dependency in dependencies
            if dependency in producer_index_by_token
        }
        if not matched_variants:
            continue
        # One transcript branch may consume exactly one audio producer.  A shared
        # STT branch would otherwise be rebound as both real transcripts.
        if len(matched_variants) != 1:
            return [], []
        if not _branch_has_executable_contract_authority(consumer):
            return [], []
        consumer_phase_id = _clean_text(consumer.get('phase_id'))
        consumer_branch_id = _clean_text(consumer.get('branch_id'))
        if not consumer_phase_id or not consumer_branch_id:
            return [], []
        if (
            consumer_phase_id in seen_consumer_phase_ids
            or consumer_branch_id in seen_consumer_branch_ids
        ):
            return [], []
        seen_consumer_phase_ids.add(consumer_phase_id)
        seen_consumer_branch_ids.add(consumer_branch_id)
        consumers_by_variant[next(iter(matched_variants))].append(consumer)

    phase_ids: list[str] = []
    bindings: list[dict[str, Any]] = []
    for variant_index, producer in indexed:
        producer_phase_id = _clean_text(producer.get('phase_id'))
        producer_branch_id = _clean_text(producer.get('branch_id'))
        consumers = consumers_by_variant.get(variant_index) or []
        if len(consumers) != 1:
            return [], []
        consumer = consumers[0]
        consumer_phase_id = _clean_text(consumer.get('phase_id'))
        consumer_branch_id = _clean_text(consumer.get('branch_id'))
        if (
            not producer_phase_id
            or not producer_branch_id
            or not consumer_phase_id
            or not consumer_branch_id
        ):
            return [], []
        phase_ids.extend((producer_phase_id, consumer_phase_id))
        bindings.extend(
            (
                {
                    'field_name': f'audio_variant_{variant_index}_artifact_ref',
                    'field_role': 'audio_artifact_ref',
                    'source_phase_id': producer_phase_id,
                    'source_branch_id': producer_branch_id or None,
                    'variant_index': variant_index,
                },
                {
                    'field_name': f'audio_variant_{variant_index}_transcript',
                    'field_role': 'real_transcript',
                    'source_phase_id': consumer_phase_id,
                    'source_branch_id': consumer_branch_id,
                    'variant_index': variant_index,
                },
            )
        )
    return list(dict.fromkeys(phase_ids)), [
        {key: value for key, value in binding.items() if value not in (None, '')}
        for binding in bindings
    ]


def _bind_explicit_structured_multi_evidence_join(
    branches: list[dict[str, Any]],
    prompt_text: str,
    *,
    preserved_reference_input_refs: Optional[list[Mapping[str, Any]]] = None,
    preserved_reference_error: str = '',
) -> None:
    """Bind a terminal structured join to every explicitly named evidence source.

    Transitive dependencies are insufficient for a final branch that must emit
    both concrete media identities and derived evidence: branch-local payloads
    expose direct dependency results. Keep this guard deliberately narrow so an
    ordinary sequential summary continues to consume only its immediate source.
    """

    normalized_prompt = normalize_intent_text(prompt_text)
    preserved_refs = [
        dict(item)
        for item in (preserved_reference_input_refs or [])
        if isinstance(item, Mapping)
    ]
    preserved_reference_contract_required = bool(
        preserved_refs or _clean_text(preserved_reference_error)
    )
    final_markers = list(_FINAL_OUTPUT_MARKER_RE.finditer(normalized_prompt))
    if not final_markers and not preserved_reference_contract_required:
        return
    required_phase_ids: list[str] = []
    required_bindings: list[dict[str, Any]] = []
    structured_contract_error = ''
    for structured_output in _STRUCTURED_FINAL_OUTPUT_RE.finditer(normalized_prompt):
        preceding_markers = [
            marker
            for marker in final_markers
            if marker.start() <= structured_output.start()
        ]
        if not preceding_markers and not preserved_reference_contract_required:
            continue
        final_contract_start = (
            preceding_markers[-1].start()
            if preceding_markers
            else structured_output.start()
        )
        binding_match = _FINAL_OUTPUT_BINDING_RE.search(
            normalized_prompt,
            structured_output.end(),
            min(len(normalized_prompt), structured_output.end() + 1000),
        )
        if not binding_match:
            continue
        final_contract = normalized_prompt[final_contract_start:binding_match.end()]
        relative_structured_output = _STRUCTURED_FINAL_OUTPUT_RE.search(final_contract)
        if not (
            relative_structured_output
            and _VISUAL_EVIDENCE_JOIN_RE.search(final_contract)
            and (
                _IMAGE_ARTIFACT_REF_RE.search(final_contract)
                or preserved_reference_contract_required
            )
            and _AUDIO_ARTIFACT_REF_RE.search(final_contract)
            and _TRANSCRIPT_JOIN_RE.search(final_contract)
        ):
            continue
        if _text_artifact_format_match_is_negated(
            final_contract,
            relative_structured_output,
        ):
            continue
        if _structured_final_join_span_is_quoted(
            normalized_prompt,
            final_contract_start,
            binding_match.end(),
        ):
            continue
        if _structured_final_join_binding_is_negated(
            normalized_prompt,
            binding_match,
        ):
            continue
        if _ASSISTANT_OUTPUT_HYPOTHETICAL_RE.search(final_contract):
            continue

        if (
            preserved_reference_contract_required
            and _structured_final_join_requests_all_audio_pairs(final_contract)
        ):
            required_phase_ids, required_bindings = (
                _structured_final_join_all_audio_pair_contract(branches)
            )
            if not required_phase_ids:
                structured_contract_error = (
                    'structured_audio_pair_lineage_ambiguous'
                )
        else:
            required_phase_ids = _structured_final_join_selected_phase_ids(
                branches,
                final_contract,
            )
        if required_phase_ids or structured_contract_error:
            break
    if not required_phase_ids and not structured_contract_error:
        return

    depended_on_tokens = {
        _clean_text(dependency)
        for branch in branches
        for dependency in (branch.get('depends_on') or [])
        if _clean_text(dependency)
    }
    terminal_chat_branches = [
        branch
        for branch in branches
        if normalize_capability(branch.get('capability')) == CAPABILITY_CHAT
        and _clean_text(branch.get('role')).lower() == 'post_artifact_text_follow_up'
        and not {
            _clean_text(branch.get('phase_id')),
            _clean_text(branch.get('branch_id')),
        }.intersection(depended_on_tokens)
    ]
    if len(terminal_chat_branches) != 1:
        return
    terminal_branch = terminal_chat_branches[0]
    terminal_branch['dependency_contract'] = 'structured_multi_evidence_join'
    if structured_contract_error:
        terminal_branch['branch_contract_error'] = structured_contract_error
        terminal_branch['blocked_by_branch_contract'] = True
        terminal_branch['repair_action'] = 'repair_branch_contract'
        terminal_branch['resolution'] = 'blocked_branch_contract'
        return
    terminal_branch['depends_on'] = list(dict.fromkeys(required_phase_ids))
    if preserved_reference_error:
        terminal_branch['branch_contract_error'] = _clean_text(
            preserved_reference_error
        )
        terminal_branch['blocked_by_branch_contract'] = True
        terminal_branch['repair_action'] = 'repair_branch_contract'
        terminal_branch['resolution'] = 'blocked_branch_contract'
    if preserved_refs:
        phase_input_refs = [
            {
                'kind': 'phase_output',
                'phase_id': phase_id,
                'role': 'dependency',
            }
            for phase_id in required_phase_ids
        ]
        terminal_branch['input_refs'] = [*phase_input_refs, *preserved_refs]
        reference_bindings = [
            {
                key: value
                for key, value in {
                    'field_name': (
                        'preserved_visual_evidence'
                        if ref.get('role') == 'preserved_visual_evidence'
                        else None
                    ),
                    'field_role': ref.get('role'),
                    'source_kind': ref.get('kind'),
                    'artifact_ref': ref.get('artifact_ref'),
                    'message_id': ref.get('message_id'),
                    'source_response_id': ref.get('source_response_id'),
                }.items()
                if value not in (None, '')
            }
            for ref in preserved_refs
        ]
        terminal_branch['structured_output_contract'] = {
            'kind': 'ollmo.structured_output_contract',
            'format': 'json_object',
            'cardinality': 'exactly_one',
            'additional_prose_allowed': False,
            'required_bindings': [*reference_bindings, *required_bindings],
        }


def _image_branch_queue_indexes(branches: list[Mapping[str, Any]]) -> set[int]:
    indexes: set[int] = set()
    for branch in branches:
        if normalize_capability(branch.get('capability')) != CAPABILITY_IMAGE_GENERATION:
            continue
        queue_index = _coerce_positive_int(branch.get('queue_index'))
        if queue_index:
            indexes.add(queue_index)
            continue
        branch_id = _clean_text(branch.get('branch_id'))
        match = re.fullmatch(r'branch-image_generation-(\d+)', branch_id)
        if match:
            indexes.add(int(match.group(1)))
    return indexes


def _next_available_index(used_indexes: set[int]) -> int:
    index = 1
    while index in used_indexes:
        index += 1
    used_indexes.add(index)
    return index


def _enforce_requested_visual_output_branch_count(
    downstream_branches: list[dict[str, Any]],
    *,
    prompt_analysis: Mapping[str, Any],
    graph_refinements: list[dict[str, Any]],
    request_payload: Optional[Mapping[str, Any]] = None,
) -> None:
    requested_count = _coerce_positive_int(prompt_analysis.get('requested_visual_output_count'))
    if requested_count <= 0 or _visual_execution_is_preserved(
        prompt_analysis,
        CAPABILITY_IMAGE_GENERATION,
    ):
        return
    request = _mapping(request_payload)
    if len(_clean_string_list(request.get('batch_prompts'))) > 1:
        return
    if bool(prompt_analysis.get('explicit_visual_defer_materialization')):
        return
    if _prompt_negates_materialization_capability(prompt_analysis, CAPABILITY_IMAGE_GENERATION):
        return
    if _prompt_reserves_entire_materialization_capability(prompt_analysis, CAPABILITY_IMAGE_GENERATION):
        return

    promoted_image_branches = [
        branch
        for branch in _promoted_branch_records(downstream_branches)
        if normalize_capability(branch.get('capability')) == CAPABILITY_IMAGE_GENERATION
    ]
    existing_count = len(promoted_image_branches)
    if existing_count >= requested_count:
        return

    missing_count = requested_count - existing_count
    next_phase_number = _next_phase_number(downstream_branches)
    used_indexes = _image_branch_queue_indexes(downstream_branches)
    added_branch_ids: list[str] = []
    for offset in range(missing_count):
        queue_index = _next_available_index(used_indexes)
        phase_id = f'phase-{next_phase_number + offset}'
        branch_id = f'branch-image_generation-{queue_index}'
        downstream_branches.append(
            {
                'branch_id': branch_id,
                'phase_id': phase_id,
                'capability': CAPABILITY_IMAGE_GENERATION,
                'output_type': 'image',
                'depends_on': ['phase-1'],
                'queue_index': queue_index,
                'source': 'request_phase_graph_explicit_visual_obligation_guard',
                'resolution': 'pending_dependency',
                'kind': 'materialize',
                'role': 'image_generation_follow_up',
                'required': True,
                'requires_artifact': True,
                'stage_direction': 'materialize_requested_image_artifact',
            }
        )
        added_branch_ids.append(branch_id)

    graph_refinements.append(
        {
            'source': 'request_phase_graph_explicit_visual_obligation_guard',
            'refinement': 'explicit_visual_obligation_guard',
            'capability': CAPABILITY_IMAGE_GENERATION,
            'reason': 'explicit requested visual output count exceeded planned executable image branches',
            'requested_count': requested_count,
            'existing_count': existing_count,
            'added_count': missing_count,
            'added_branch_ids': added_branch_ids,
        }
    )


def _synthesized_mixed_media_final_text_branches(
    capabilities: list[str],
    *,
    prompt_analysis: Optional[Mapping[str, Any]] = None,
    request_payload: Optional[Mapping[str, Any]] = None,
    response_payload: Optional[Mapping[str, Any]] = None,
    refinement_capabilities: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    normalized_capabilities = [
        normalized
        for item in capabilities
        if (normalized := normalize_capability(item))
    ]
    if not normalized_capabilities or normalized_capabilities[-1] != CAPABILITY_CHAT:
        return []
    prefix = normalized_capabilities[:-1]
    if (
        CAPABILITY_TEXT_TO_SPEECH not in prefix
        or CAPABILITY_IMAGE_GENERATION not in prefix
        or CAPABILITY_CHAT in prefix
    ):
        return []
    supported_prefix = {
        CAPABILITY_TEXT_TO_SPEECH,
        CAPABILITY_IMAGE_GENERATION,
        CAPABILITY_VISION_ANALYSIS,
        CAPABILITY_SPEECH_TO_TEXT,
    }
    if any(capability not in supported_prefix for capability in prefix):
        return []

    analysis = prompt_analysis if isinstance(prompt_analysis, Mapping) else {}
    prompt_text = _clean_text(analysis.get('normalized_prompt'))
    request = request_payload if isinstance(request_payload, Mapping) else {}
    response = response_payload if isinstance(response_payload, Mapping) else {}
    response_phase_payload = (
        response.get('phase_payload')
        if isinstance(response.get('phase_payload'), Mapping)
        else {}
    )
    shared_batch_prompts = _clean_string_list(
        request.get('batch_prompts')
        or response.get('batch_prompts')
        or response_phase_payload.get('batch_prompts')
    )
    shared_batch_count = max(
        _coerce_positive_int(request.get('batch_count')),
        _coerce_positive_int(response.get('batch_count')),
        _coerce_positive_int(response_phase_payload.get('batch_count')),
    )
    requested_visual_output_count = _coerce_positive_int(analysis.get('requested_visual_output_count'))
    requested_audio_output_count = _requested_audio_output_count(analysis)
    selected_speakable_candidate_fields = _selected_speakable_candidate_fields(analysis)
    audio_generation_action_count = sum(
        1 for item in prefix if item == CAPABILITY_TEXT_TO_SPEECH
    )
    image_generation_action_count = sum(
        1 for item in prefix if item == CAPABILITY_IMAGE_GENERATION
    )
    needs_speech_to_text = (
        CAPABILITY_SPEECH_TO_TEXT in prefix
        or bool(_POST_AUDIO_TEXT_REQUIRES_STT_RE.search(prompt_text))
    )
    needs_visual_analysis = not _POST_GENERATED_ARTIFACT_INVENTORY_RE.search(prompt_text)

    branches: list[dict[str, Any]] = []
    occurrences: dict[str, int] = {}
    audio_phase_ids: list[str] = []
    image_phase_ids: list[str] = []

    def append_branch(
        capability: str,
        *,
        depends_on: list[str],
        kind: str,
        role: str,
        extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        occurrences[capability] = occurrences.get(capability, 0) + 1
        occurrence = occurrences[capability]
        phase_id = f'phase-{len(branches) + 2}'
        source = (
            'assistant_output_claim_refinement'
            if capability in (refinement_capabilities or set())
            else 'request_phase_graph_post_artifact_join'
        )
        record = {
            'branch_id': f'branch-{capability}-{occurrence}',
            'phase_id': phase_id,
            'capability': capability,
            'output_type': _output_type_for_capability(capability),
            'depends_on': list(depends_on),
            'queue_index': occurrence,
            'source': source,
            'resolution': 'pending_dependency',
            'kind': kind,
            'role': role,
        }
        if extra:
            record.update(extra)
        branches.append(record)
        return record

    for capability in prefix:
        if capability in {CAPABILITY_VISION_ANALYSIS, CAPABILITY_SPEECH_TO_TEXT}:
            continue
        if capability == CAPABILITY_TEXT_TO_SPEECH:
            branch_count = (
                max(requested_audio_output_count, 1)
                if audio_generation_action_count == 1
                else 1
            )
            for _ in range(branch_count):
                next_occurrence = occurrences.get(CAPABILITY_TEXT_TO_SPEECH, 0) + 1
                extra = (
                    dict(selected_speakable_candidate_fields)
                    if selected_speakable_candidate_fields
                    else _counted_audio_variant_selection_fields(
                        requested_audio_output_count,
                        next_occurrence,
                        analysis,
                    )
                )
                branch = append_branch(
                    CAPABILITY_TEXT_TO_SPEECH,
                    depends_on=['phase-1'],
                    kind='materialize',
                    role=f'{CAPABILITY_TEXT_TO_SPEECH}_follow_up',
                    extra=extra or None,
                )
                audio_phase_ids.append(branch['phase_id'])
            continue
        if capability != CAPABILITY_IMAGE_GENERATION:
            return []

        branch_count = 1
        branch_batch_prompts: list[str] = []
        if image_generation_action_count == 1:
            if len(shared_batch_prompts) > 1:
                branch_batch_prompts = list(shared_batch_prompts)
                branch_count = len(branch_batch_prompts)
            else:
                branch_count = max(shared_batch_count, requested_visual_output_count, 1)
        for _ in range(branch_count):
            extra: dict[str, Any] = {}
            next_occurrence = occurrences.get(CAPABILITY_IMAGE_GENERATION, 0) + 1
            if branch_batch_prompts and next_occurrence <= len(branch_batch_prompts):
                extra['artifact_prompt'] = branch_batch_prompts[next_occurrence - 1]
                extra['artifact_prompt_source'] = 'semantic_batch_prompts'
                extra['batch_prompts'] = list(branch_batch_prompts)
            branch = append_branch(
                CAPABILITY_IMAGE_GENERATION,
                depends_on=['phase-1'],
                kind='materialize',
                role=f'{CAPABILITY_IMAGE_GENERATION}_follow_up',
                extra=extra,
            )
            image_phase_ids.append(branch['phase_id'])

    vision_phase_ids: list[str] = []
    if needs_visual_analysis:
        for phase_id in image_phase_ids:
            branch = append_branch(
                CAPABILITY_VISION_ANALYSIS,
                depends_on=[phase_id],
                kind='evidence',
                role=f'{CAPABILITY_VISION_ANALYSIS}_follow_up',
            )
            vision_phase_ids.append(branch['phase_id'])

    speech_to_text_phase_ids: list[str] = []
    if needs_speech_to_text:
        for phase_id in audio_phase_ids:
            branch = append_branch(
                CAPABILITY_SPEECH_TO_TEXT,
                depends_on=[phase_id],
                kind='materialize',
                role=f'{CAPABILITY_SPEECH_TO_TEXT}_follow_up',
            )
            speech_to_text_phase_ids.append(branch['phase_id'])

    final_depends_on = (speech_to_text_phase_ids or audio_phase_ids) + (vision_phase_ids or image_phase_ids)
    if not final_depends_on:
        return []
    append_branch(
        CAPABILITY_CHAT,
        depends_on=final_depends_on,
        kind='postprocess',
        role='post_artifact_text_follow_up',
        extra={
            'content_payload_source': 'prior_artifact_result',
            'stage_direction': 'write_text_after_artifact_generation',
        },
    )
    _bind_explicit_structured_multi_evidence_join(branches, prompt_text)
    return branches


def _normalize_explicit_downstream_branch(
    raw_branch: Any,
    *,
    fallback_index: int,
) -> Optional[dict[str, Any]]:
    if isinstance(raw_branch, Mapping):
        capability = normalize_capability(raw_branch.get('capability'))
        branch_id = _clean_text(raw_branch.get('branch_id') or raw_branch.get('phase_id'))
        phase_id = _clean_text(raw_branch.get('phase_id') or branch_id)
        output_type = _clean_text(raw_branch.get('output_type')).lower() or _output_type_for_capability(capability)
        if not capability:
            return None
        if not branch_id:
            branch_id = f'branch-{capability}-{fallback_index}'
        if not phase_id:
            phase_id = f'phase-{fallback_index + 1}'
        record = {
            'branch_id': branch_id,
            'phase_id': phase_id,
            'capability': capability,
            'output_type': output_type,
            'depends_on': [
                _clean_text(item)
                for item in (raw_branch.get('depends_on') or [])
                if _clean_text(item)
            ] or ['phase-1'],
        }
        queue_index = _coerce_positive_int(raw_branch.get('queue_index'))
        if queue_index:
            record['queue_index'] = queue_index
        status = _clean_text(raw_branch.get('status')).lower()
        if status:
            record['status'] = status
        required = raw_branch.get('required')
        if isinstance(required, bool):
            record['required'] = required
        elif _clean_text(required):
            record['required'] = _clean_text(required)
        source = _clean_text(raw_branch.get('source'))
        if source:
            record['source'] = source
        for key in (
            'candidate_id',
            'contract_state',
            'contract_status',
            'obligation_state',
            'intent_state',
            'promotion_policy',
            'promotion_reason',
            'promotion_source',
            'promoted_from_candidate_id',
            'artifact_prompt',
            'artifact_prompt_source',
            'content_payload',
            'content_payload_source',
            'phase_summary',
            'stage_direction',
            'candidate_selection_index',
            'candidate_selection_count',
            'selection_policy',
            'selection_reason',
            'lang_code',
            'audio_variant_index',
            'audio_variant_role',
            'audio_variant_contract_source',
            'structured_output_contract',
            'branch_contract_error',
            'audio_variant_contract_conflicting_fields',
            'kind',
            'role',
            'semantic_intent',
            'objective',
            'deliverable',
            'rationale',
            'review_criteria',
            'input_refs',
            'execution_contract',
            'workload_task_ref',
            'output_obligation_ref',
            'output_contract',
            'accepted_proposals',
            'requires_artifact',
            'dependency_contract',
            'image_asset_binding_required',
            'required_image_phase_ids',
            'text_artifact_extension',
            'text_artifact_source_name',
            'text_artifact_source',
            'text_artifact_target_path',
            'artifact_request',
            'repair_action',
            'recovery_action',
            'repair_action_reason',
            'blocked_by_dependency_input',
            'blocked_by_branch_contract',
        ):
            value = raw_branch.get(key)
            if value not in (None, '', [], {}):
                record[key] = value
        batch_prompts = _clean_string_list(raw_branch.get('batch_prompts'))
        if batch_prompts:
            record['batch_prompts'] = batch_prompts
        promotion = raw_branch.get('promotion')
        if isinstance(promotion, Mapping):
            record['promotion'] = dict(promotion)
        return record
    capability = normalize_capability(raw_branch)
    if not capability:
        return None
    return {
        'branch_id': f'branch-{capability}-{fallback_index}',
        'phase_id': f'phase-{fallback_index + 1}',
        'capability': capability,
        'output_type': _output_type_for_capability(capability),
        'depends_on': ['phase-1'],
    }


def _explicit_downstream_branches(
    request_payload: Mapping[str, Any],
    route_payload: Mapping[str, Any],
    response_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    route_runtime = route_payload.get('route_runtime') if isinstance(route_payload.get('route_runtime'), Mapping) else {}
    response_runtime = response_payload.get('runtime') if isinstance(response_payload.get('runtime'), Mapping) else {}
    response_late_fill = response_payload.get('late_fill') if isinstance(response_payload.get('late_fill'), Mapping) else {}
    execution_planner = (
        route_runtime.get('execution_planner')
        if isinstance(route_runtime.get('execution_planner'), Mapping)
        else (
            response_runtime.get('execution_planner')
            if isinstance(response_runtime.get('execution_planner'), Mapping)
            else {}
        )
    )
    candidates = (
        request_payload.get('downstream_branches'),
        request_payload.get('downstream_phases'),
        request_payload.get('deferred_branches'),
        route_payload.get('downstream_branches'),
        route_payload.get('downstream_phases'),
        route_runtime.get('downstream_branches') if isinstance(route_runtime, Mapping) else None,
        route_runtime.get('downstream_phases') if isinstance(route_runtime, Mapping) else None,
        execution_planner.get('deferred_branches') if isinstance(execution_planner, Mapping) else None,
        response_runtime.get('downstream_branches') if isinstance(response_runtime, Mapping) else None,
        response_runtime.get('downstream_phases') if isinstance(response_runtime, Mapping) else None,
        response_late_fill.get('pending_branches') if isinstance(response_late_fill, Mapping) else None,
        response_late_fill.get('completed_branches') if isinstance(response_late_fill, Mapping) else None,
        response_late_fill.get('failed_branches') if isinstance(response_late_fill, Mapping) else None,
        response_late_fill.get('active_branches') if isinstance(response_late_fill, Mapping) else None,
        response_late_fill.get('fill_results') if isinstance(response_late_fill, Mapping) else None,
    )
    branches: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    fallback_index = 1
    for candidate in candidates:
        if not isinstance(candidate, list):
            continue
        for item in candidate:
            normalized = _normalize_explicit_downstream_branch(item, fallback_index=fallback_index)
            fallback_index += 1
            if not normalized:
                continue
            branch_id = _clean_text(normalized.get('branch_id'))
            if branch_id in seen_ids:
                continue
            seen_ids.add(branch_id)
            branches.append(normalized)
    return branches


def _explicit_workload_task_proposals(
    request_payload: Mapping[str, Any],
    route_payload: Mapping[str, Any],
    response_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    route_runtime = route_payload.get('route_runtime') if isinstance(route_payload.get('route_runtime'), Mapping) else {}
    response_runtime = response_payload.get('runtime') if isinstance(response_payload.get('runtime'), Mapping) else {}
    candidates = (
        request_payload.get('workload_task_proposals'),
        request_payload.get('workload_tasks'),
        route_payload.get('workload_task_proposals'),
        route_payload.get('workload_tasks'),
        route_runtime.get('workload_task_proposals') if isinstance(route_runtime, Mapping) else None,
        route_runtime.get('workload_tasks') if isinstance(route_runtime, Mapping) else None,
        response_runtime.get('workload_task_proposals') if isinstance(response_runtime, Mapping) else None,
        response_runtime.get('workload_tasks') if isinstance(response_runtime, Mapping) else None,
    )
    proposals: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str, str]] = set()
    for candidate in candidates:
        if not isinstance(candidate, list):
            continue
        for raw_item in candidate:
            if not isinstance(raw_item, Mapping):
                continue
            item = dict(raw_item)
            key = (
                _clean_text(item.get('proposal_id') or item.get('id')),
                _clean_text(item.get('phase_id')),
                _clean_text(item.get('task_id') or item.get('workload_task_id')),
                _clean_text(item.get('branch_id')),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            proposals.append(item)
    return proposals


def _synthesized_downstream_branches(
    capabilities: list[str],
    *,
    prompt_analysis: Mapping[str, Any],
    request_payload: Mapping[str, Any],
    response_payload: Mapping[str, Any],
    refinement_capabilities: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    branches: list[dict[str, Any]] = []
    occurrences: dict[str, int] = {}
    response_phase_payload = (
        response_payload.get('phase_payload')
        if isinstance(response_payload.get('phase_payload'), Mapping)
        else {}
    )
    shared_batch_prompts = _clean_string_list(
        request_payload.get('batch_prompts')
        or response_payload.get('batch_prompts')
        or response_phase_payload.get('batch_prompts')
    )
    shared_batch_count = max(
        _coerce_positive_int(request_payload.get('batch_count')),
        _coerce_positive_int(response_payload.get('batch_count')),
        _coerce_positive_int(response_phase_payload.get('batch_count')),
    )
    requested_visual_output_count = _coerce_positive_int(prompt_analysis.get('requested_visual_output_count'))
    requested_audio_output_count = _requested_audio_output_count(prompt_analysis)
    selected_speakable_candidate_fields = _selected_speakable_candidate_fields(prompt_analysis)
    text_to_speech_phase_ids: list[str] = []
    speech_to_text_phase_ids: list[str] = []
    text_to_speech_action_count = sum(
        1 for item in capabilities if item == CAPABILITY_TEXT_TO_SPEECH
    )
    speech_to_text_action_count = sum(
        1 for item in capabilities if item == CAPABILITY_SPEECH_TO_TEXT
    )
    for capability in capabilities:
        branch_count = 1
        branch_batch_prompts: list[str] = []
        if capability == CAPABILITY_IMAGE_GENERATION:
            if len(shared_batch_prompts) > 1:
                branch_batch_prompts = list(shared_batch_prompts)
                branch_count = len(branch_batch_prompts)
            else:
                branch_count = max(shared_batch_count, requested_visual_output_count, 1)
        elif capability == CAPABILITY_TEXT_TO_SPEECH and text_to_speech_action_count == 1:
            branch_count = max(requested_audio_output_count, 1)
        elif (
            capability == CAPABILITY_SPEECH_TO_TEXT
            and speech_to_text_action_count == 1
            and text_to_speech_phase_ids
            and not _has_audio_input_artifact(request_payload)
        ):
            branch_count = len(text_to_speech_phase_ids)
        for branch_offset in range(branch_count):
            occurrences[capability] = occurrences.get(capability, 0) + 1
            occurrence = occurrences[capability]
            source = (
                'assistant_output_claim_refinement'
                if capability in (refinement_capabilities or set())
                else 'request_phase_graph'
            )
            branch_record = {
                'branch_id': f'branch-{capability}-{occurrence}',
                'phase_id': f'phase-{len(branches) + 2}',
                'capability': capability,
                'output_type': _output_type_for_capability(capability),
                'depends_on': ['phase-1'],
                'queue_index': occurrence,
                'source': source,
            }
            if capability == CAPABILITY_TEXT_TO_SPEECH:
                branch_record.update(
                    selected_speakable_candidate_fields
                    or _counted_audio_variant_selection_fields(
                        requested_audio_output_count,
                        occurrence,
                        prompt_analysis,
                    )
                )
            if (
                capability == CAPABILITY_SPEECH_TO_TEXT
                and text_to_speech_phase_ids
                and not _has_audio_input_artifact(request_payload)
            ):
                branch_record['depends_on'] = [
                    text_to_speech_phase_ids[
                        min(branch_offset, len(text_to_speech_phase_ids) - 1)
                    ]
                ]
            if (
                speech_to_text_phase_ids
                and capability == CAPABILITY_CHAT
            ):
                branch_record['depends_on'] = list(speech_to_text_phase_ids)
                branch_record['content_payload_source'] = 'speech_to_text_branch_result'
            elif (
                speech_to_text_phase_ids
                and capability != CAPABILITY_SPEECH_TO_TEXT
                and _has_audio_input_artifact(request_payload)
                and _speech_to_text_result_feeds_capability(prompt_analysis, capability)
            ):
                branch_record['depends_on'] = list(speech_to_text_phase_ids)
                branch_record['content_payload_source'] = 'speech_to_text_branch_result'
            if branch_batch_prompts and occurrence <= len(branch_batch_prompts):
                branch_record['artifact_prompt'] = branch_batch_prompts[occurrence - 1]
                branch_record['artifact_prompt_source'] = 'semantic_batch_prompts'
                branch_record['batch_prompts'] = list(branch_batch_prompts)
            branches.append(branch_record)
            if capability == CAPABILITY_TEXT_TO_SPEECH:
                text_to_speech_phase_ids.append(branch_record['phase_id'])
            if capability == CAPABILITY_SPEECH_TO_TEXT:
                speech_to_text_phase_ids.append(branch_record['phase_id'])
    return branches


def _synthesized_sequential_downstream_branches(
    capabilities: list[str],
    *,
    prompt_analysis: Optional[Mapping[str, Any]] = None,
    request_payload: Optional[Mapping[str, Any]] = None,
    response_payload: Optional[Mapping[str, Any]] = None,
    refinement_capabilities: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    branches: list[dict[str, Any]] = []
    occurrences: dict[str, int] = {}
    previous_phase_ids = ['phase-1']
    analysis = prompt_analysis if isinstance(prompt_analysis, Mapping) else {}
    request = request_payload if isinstance(request_payload, Mapping) else {}
    response = response_payload if isinstance(response_payload, Mapping) else {}
    response_phase_payload = (
        response.get('phase_payload')
        if isinstance(response.get('phase_payload'), Mapping)
        else {}
    )
    shared_batch_prompts = _clean_string_list(
        request.get('batch_prompts')
        or response.get('batch_prompts')
        or response_phase_payload.get('batch_prompts')
    )
    shared_batch_count = max(
        _coerce_positive_int(request.get('batch_count')),
        _coerce_positive_int(response.get('batch_count')),
        _coerce_positive_int(response_phase_payload.get('batch_count')),
    )
    requested_visual_output_count = _coerce_positive_int(analysis.get('requested_visual_output_count'))
    requested_audio_output_count = _requested_audio_output_count(analysis)
    normalized_capabilities = [
        normalized
        for item in capabilities
        if (normalized := normalize_capability(item))
    ]
    image_generation_action_count = sum(
        1 for item in normalized_capabilities if item == CAPABILITY_IMAGE_GENERATION
    )
    text_to_speech_action_count = sum(
        1 for item in normalized_capabilities if item == CAPABILITY_TEXT_TO_SPEECH
    )
    selected_image_candidate_index = _selected_image_candidate_index(analysis)
    selected_speakable_candidate_fields = _selected_speakable_candidate_fields(analysis)
    root_materialization_siblings = bool(normalized_capabilities) and set(normalized_capabilities).issubset(
        {
            CAPABILITY_TEXT_TO_SPEECH,
            CAPABILITY_IMAGE_GENERATION,
        }
    )
    previous_capability = ''
    for normalized in normalized_capabilities:
        branch_count = 1
        branch_batch_prompts: list[str] = []
        if normalized == CAPABILITY_IMAGE_GENERATION and image_generation_action_count == 1:
            if len(shared_batch_prompts) > 1:
                branch_batch_prompts = list(shared_batch_prompts)
                branch_count = len(branch_batch_prompts)
            else:
                branch_count = max(shared_batch_count, requested_visual_output_count, 1)
        elif normalized == CAPABILITY_TEXT_TO_SPEECH and text_to_speech_action_count == 1:
            branch_count = max(requested_audio_output_count, 1)
        elif normalized == CAPABILITY_VISION_ANALYSIS and len(previous_phase_ids) > 1:
            branch_count = len(previous_phase_ids)
        elif (
            normalized == CAPABILITY_SPEECH_TO_TEXT
            and previous_capability == CAPABILITY_TEXT_TO_SPEECH
            and len(previous_phase_ids) > 1
        ):
            branch_count = len(previous_phase_ids)
        created_phase_ids: list[str] = []
        for branch_offset in range(branch_count):
            occurrences[normalized] = occurrences.get(normalized, 0) + 1
            occurrence = occurrences[normalized]
            phase_id = f'phase-{len(branches) + 2}'
            source = (
                'assistant_output_claim_refinement'
                if normalized in (refinement_capabilities or set())
                else 'request_phase_graph_post_artifact_continuation'
            )
            depends_on = (
                ['phase-1']
                if root_materialization_siblings
                and normalized in {CAPABILITY_TEXT_TO_SPEECH, CAPABILITY_IMAGE_GENERATION}
                else list(previous_phase_ids)
            )
            if (
                normalized in {CAPABILITY_VISION_ANALYSIS, CAPABILITY_SPEECH_TO_TEXT}
                and branch_count == len(previous_phase_ids)
            ):
                depends_on = [previous_phase_ids[branch_offset]]
            branch_record = {
                'branch_id': f'branch-{normalized}-{occurrence}',
                'phase_id': phase_id,
                'capability': normalized,
                'output_type': _output_type_for_capability(normalized),
                'depends_on': depends_on,
                'queue_index': (
                    selected_image_candidate_index
                    if normalized == CAPABILITY_IMAGE_GENERATION
                    and branch_count == 1
                    and selected_image_candidate_index > 0
                    else occurrence
                ),
                'source': source,
                'resolution': 'pending_dependency',
            }
            if normalized == CAPABILITY_TEXT_TO_SPEECH:
                branch_record.update(
                    selected_speakable_candidate_fields
                    or _counted_audio_variant_selection_fields(
                        requested_audio_output_count,
                        occurrence,
                        analysis,
                    )
                )
            if (
                normalized == CAPABILITY_IMAGE_GENERATION
                and branch_count == 1
                and selected_image_candidate_index > 0
            ):
                branch_record['candidate_selection_index'] = selected_image_candidate_index
                branch_record['selection_policy'] = 'selected_candidate_only'
            if branch_batch_prompts and occurrence <= len(branch_batch_prompts):
                branch_record['artifact_prompt'] = branch_batch_prompts[occurrence - 1]
                branch_record['artifact_prompt_source'] = 'semantic_batch_prompts'
                branch_record['batch_prompts'] = list(branch_batch_prompts)
            if normalized == CAPABILITY_CHAT:
                if (
                    CAPABILITY_SPEECH_TO_TEXT in normalized_capabilities
                    and _COMPARE_WITH_ORIGINAL_TEXT_RE.search(_clean_text(analysis.get('normalized_prompt')))
                    and 'phase-1' not in depends_on
                ):
                    depends_on = ['phase-1', *depends_on]
                    branch_record['depends_on'] = depends_on
                branch_record.update(
                    {
                        'kind': 'postprocess',
                        'role': 'post_artifact_text_follow_up',
                        'content_payload_source': 'prior_artifact_result',
                        'stage_direction': 'write_text_after_artifact_generation',
                    }
                )
            else:
                branch_record['kind'] = 'materialize'
                branch_record['role'] = f'{normalized}_follow_up'
            branches.append(branch_record)
            created_phase_ids.append(phase_id)
        if created_phase_ids and not root_materialization_siblings:
            previous_phase_ids = created_phase_ids
        previous_capability = normalized
    _bind_explicit_structured_multi_evidence_join(
        branches,
        _clean_text(analysis.get('normalized_prompt')),
    )
    return branches


def _reserved_materialization_candidates(
    prompt_analysis: Mapping[str, Any],
    *,
    route_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Represent explicitly negated materialization as reserved possibilities."""

    prompt_text = _clean_text(prompt_analysis.get('normalized_prompt'))
    if not prompt_text:
        return []
    route_capability = normalize_capability(route_payload.get('capability'))
    candidates: list[dict[str, Any]] = []
    rejects_image_candidate = bool(
        _REJECTED_IMAGE_CANDIDATE_PLANNING_RE.search(prompt_text)
    )
    preserves_image = _visual_execution_is_preserved(
        prompt_analysis,
        CAPABILITY_IMAGE_GENERATION,
    )
    reserves_image = bool(
        not rejects_image_candidate
        and not preserves_image
        and _prompt_reserves_materialization_capability(
            prompt_analysis,
            CAPABILITY_IMAGE_GENERATION,
        )
    )
    negates_image = bool(
        not rejects_image_candidate
        and not preserves_image
        and _prompt_negates_materialization_capability(
            prompt_analysis,
            CAPABILITY_IMAGE_GENERATION,
        )
    )
    if (
        (negates_image or reserves_image)
        and (
            bool(prompt_analysis.get('requests_visual_output'))
            or normalize_capability(prompt_analysis.get('primary_capability')) == CAPABILITY_IMAGE_GENERATION
            or route_capability == CAPABILITY_IMAGE_GENERATION
            or negates_image
            or reserves_image
        )
    ):
        promotion_reason = (
            'current prompt asked to keep image materialization as a reserved option'
            if reserves_image and not negates_image
            else 'current prompt explicitly asked not to generate the image yet'
        )
        candidates.append(
            {
                'branch_id': 'branch-image_generation-reserved-1',
                'phase_id': 'phase-image_generation-reserved-1',
                'candidate_id': 'candidate-image_generation-reserved-1',
                'capability': CAPABILITY_IMAGE_GENERATION,
                'output_type': 'image',
                'depends_on': ['phase-1'],
                'queue_index': 1,
                'source': 'request_phase_graph_reserved_materialization'
                if reserves_image and not negates_image
                else 'request_phase_graph_negated_materialization',
                'required': False,
                'contract_state': 'reserved',
                'promotion_policy': 'requires_user_confirmation',
                'promotion_reason': promotion_reason,
                'kind': 'candidate',
                'role': 'reserved_materialization_candidate',
                'resolution': 'reserved_until_promoted',
            }
        )
    if bool(prompt_analysis.get('audio_output_count_exceeds_bound')):
        raw_count = _coerce_positive_int(prompt_analysis.get('requested_audio_output_count_raw'))
        maximum = _coerce_positive_int(prompt_analysis.get('requested_audio_output_count_max')) or 6
        candidates.append(
            {
                'branch_id': 'branch-text_to_speech-cardinality-blocked-1',
                'phase_id': 'phase-text_to_speech-cardinality-blocked-1',
                'candidate_id': 'candidate-text_to_speech-cardinality-blocked-1',
                'capability': CAPABILITY_TEXT_TO_SPEECH,
                'output_type': 'audio',
                'depends_on': ['phase-1'],
                'queue_index': 1,
                'source': 'request_phase_graph_audio_cardinality_guard',
                'required': False,
                'contract_state': 'reserved',
                'promotion_policy': 'requires_valid_bounded_audio_count',
                'promotion_reason': (
                    f'requested audio output count {raw_count} exceeds bounded maximum {maximum}'
                ),
                'kind': 'candidate',
                'role': 'blocked_materialization_candidate',
                'resolution': 'blocked_invalid_cardinality',
                'phase_summary': (
                    f'blocked audio materialization count {raw_count}; maximum is {maximum}'
                ),
                'repair_action': 'clarify_audio_output_count',
                'repair_action_reason': 'requested audio count exceeds the bounded synthesis contract',
                'blocked_by_branch_contract': True,
            }
        )
    if (
        _prompt_negates_materialization_capability(prompt_analysis, CAPABILITY_TEXT_TO_SPEECH)
        and not bool(prompt_analysis.get('audio_output_count_exceeds_bound'))
        and (
            bool(prompt_analysis.get('requests_audio_output'))
            or normalize_capability(prompt_analysis.get('primary_capability')) == CAPABILITY_TEXT_TO_SPEECH
            or route_capability == CAPABILITY_TEXT_TO_SPEECH
        )
    ):
        candidates.append(
            {
                'branch_id': 'branch-text_to_speech-reserved-1',
                'phase_id': 'phase-text_to_speech-reserved-1',
                'candidate_id': 'candidate-text_to_speech-reserved-1',
                'capability': CAPABILITY_TEXT_TO_SPEECH,
                'output_type': 'audio',
                'depends_on': ['phase-1'],
                'queue_index': 1,
                'source': 'request_phase_graph_negated_materialization',
                'required': False,
                'contract_state': 'reserved',
                'promotion_policy': 'requires_user_confirmation',
                'promotion_reason': 'current prompt explicitly asked not to generate audio yet',
                'kind': 'candidate',
                'role': 'reserved_materialization_candidate',
                'resolution': 'reserved_until_promoted',
            }
        )
    return candidates


def _audio_cardinality_blocking_obligation(
    prompt_analysis: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    if not bool(prompt_analysis.get('audio_output_count_exceeds_bound')):
        return None
    raw_count = _coerce_positive_int(prompt_analysis.get('requested_audio_output_count_raw'))
    maximum = _coerce_positive_int(prompt_analysis.get('requested_audio_output_count_max')) or 6
    return {
        'obligation_id': 'intent-obligation-audio-cardinality-guard-1',
        'kind': 'intent_cardinality_guard',
        'source': 'current_user_intent',
        'evidence': 'audio_output_count_exceeds_bound',
        'required': True,
        'capability': CAPABILITY_TEXT_TO_SPEECH,
        'output_type': 'audio',
        'count': 1,
        'requested_count': raw_count,
        'maximum_allowed_count': maximum,
        'relationship': 'blocks_materialization_until_clarified',
        'status': 'blocked',
        'resolution': 'needs_clarification',
        'promotion_policy': 'requires_valid_bounded_audio_count',
        'repair_action': 'clarify_audio_output_count',
        'recovery_action': 'clarify_audio_output_count',
        'repair_action_reason': 'requested audio count exceeds the bounded synthesis contract',
        'blocked_by_branch_contract': True,
        'depends_on_obligation_ids': [],
    }


def _downstream_branches(
    prompt_analysis: Mapping[str, Any],
    *,
    request_payload: Mapping[str, Any],
    request_meta: Mapping[str, Any],
    route_payload: Mapping[str, Any],
    response_payload: Mapping[str, Any],
    refinement_capabilities: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    explicit = _explicit_downstream_branches(request_payload, route_payload, response_payload)
    if explicit:
        return explicit
    reserved = _reserved_materialization_candidates(prompt_analysis, route_payload=route_payload)
    post_artifact_sequence = _post_artifact_continuation_sequence(prompt_analysis)
    has_pre_tts_stt, post_tts_stt_sources = _stt_source_contract_around_tts(
        prompt_analysis
    )
    post_tts_direct_input_stt, post_tts_generated_audio_stt = _post_tts_stt_source_targets(
        prompt_analysis
    )
    if (
        post_artifact_sequence
        and _has_audio_input_artifact(request_payload)
        and _prompt_references_current_input_audio(prompt_analysis)
        and post_tts_direct_input_stt
        and post_tts_generated_audio_stt
        and normalize_capability(post_artifact_sequence[0])
        != CAPABILITY_SPEECH_TO_TEXT
    ):
        # The direct input transcription is an independent current phase. Keep
        # the separately requested TTS -> STT pair as its generated-audio
        # continuation instead of clearing the whole ordered sequence.
        post_artifact_sequence = [
            CAPABILITY_SPEECH_TO_TEXT,
            *post_artifact_sequence,
        ]
    if (
        post_artifact_sequence
        and normalize_capability(post_artifact_sequence[0]) == CAPABILITY_SPEECH_TO_TEXT
        and _has_audio_input_artifact(request_payload)
        and post_tts_stt_sources
    ):
        represented_post_tts_sources = (
            len(post_tts_stt_sources)
            if has_pre_tts_stt
            else sum(source == 'generated_audio' for source in post_tts_stt_sources)
        )
        desired_stt_count = 1 + represented_post_tts_sources
        existing_stt_count = sum(
            normalize_capability(capability) == CAPABILITY_SPEECH_TO_TEXT
            for capability in post_artifact_sequence
        )
        missing_stt_count = max(desired_stt_count - existing_stt_count, 0)
        if missing_stt_count:
            last_stt_index = max(
                index
                for index, capability in enumerate(post_artifact_sequence)
                if normalize_capability(capability) == CAPABILITY_SPEECH_TO_TEXT
            )
            post_artifact_sequence = [
                *post_artifact_sequence[:last_stt_index + 1],
                *([CAPABILITY_SPEECH_TO_TEXT] * missing_stt_count),
                *post_artifact_sequence[last_stt_index + 1:],
            ]
    if (
        post_artifact_sequence
        and CAPABILITY_SPEECH_TO_TEXT in post_artifact_sequence
        and _has_audio_input_artifact(request_payload)
        and _prompt_references_current_input_audio(prompt_analysis)
        and (
            not _sequence_binds_stt_to_generated_audio(post_artifact_sequence)
            or (
                _prompt_targets_direct_input_audio_for_stt(prompt_analysis)
                and normalize_capability(post_artifact_sequence[0])
                != CAPABILITY_SPEECH_TO_TEXT
            )
        )
    ):
        # A named input-audio transcription is direct evidence work, not a
        # downstream transcription of a separately requested generated audio.
        # Keep a leading direct STT in a longer STT -> TTS -> STT chain so the
        # normal promotion pass can preserve the generated-audio consumer.
        post_artifact_sequence = []
    if post_artifact_sequence:
        mixed_media_final_text_branches = _synthesized_mixed_media_final_text_branches(
            post_artifact_sequence,
            prompt_analysis=prompt_analysis,
            request_payload=request_payload,
            response_payload=response_payload,
            refinement_capabilities=refinement_capabilities,
        )
        if mixed_media_final_text_branches:
            return [*mixed_media_final_text_branches, *reserved]
        sequential_branches = _synthesized_sequential_downstream_branches(
            post_artifact_sequence,
            prompt_analysis=prompt_analysis,
            request_payload=request_payload,
            response_payload=response_payload,
            refinement_capabilities=refinement_capabilities,
        )
        if _has_audio_input_artifact(request_payload):
            sequential_branches = _bind_post_tts_direct_input_stt_to_request_audio(
                sequential_branches,
                prompt_analysis=prompt_analysis,
                request_payload=request_payload,
            )
        return [
            *sequential_branches,
            *reserved,
        ]
    capabilities = _downstream_capabilities(
        prompt_analysis,
        request_meta=request_meta,
        request_payload=request_payload,
        route_payload=route_payload,
    )
    promoted = _synthesized_downstream_branches(
        capabilities,
        prompt_analysis=prompt_analysis,
        request_payload=request_payload,
        response_payload=response_payload,
        refinement_capabilities=refinement_capabilities,
    )
    if promoted:
        return [*promoted, *reserved]
    return reserved


def _current_phase_reason(
    downstream_capabilities: list[str],
    *,
    prompt_analysis: Mapping[str, Any],
    route_payload: Mapping[str, Any],
) -> Optional[str]:
    if not downstream_capabilities:
        return None
    route_source = _clean_text(route_payload.get('route_source')).lower()
    if route_source == 'ghost_carried':
        return 'current phase stays on Ghost chat while deferred follow-up phases remain unresolved'
    if bool(prompt_analysis.get('text_preparation_before_audio_output')) and bool(
        prompt_analysis.get('text_preparation_before_visual_output')
    ):
        return 'text preparation is required before downstream audio and image materialization'
    if bool(prompt_analysis.get('text_preparation_before_audio_output')):
        return 'text preparation is required before downstream audio materialization'
    if bool(prompt_analysis.get('text_preparation_before_visual_output')):
        return 'text preparation is required before downstream image materialization'
    return 'current phase remains text-capable while downstream materialization phases depend on its output'


def _direct_stt_is_generated_audio_follow_up(
    capability: Any,
    downstream_branches: list[dict[str, Any]],
) -> bool:
    """Return whether the graph already binds STT to a generated-audio producer."""

    if normalize_capability(capability) != CAPABILITY_SPEECH_TO_TEXT:
        return False
    audio_producer_tokens: set[str] = set()
    for branch in downstream_branches:
        if normalize_capability(branch.get('capability')) != CAPABILITY_TEXT_TO_SPEECH:
            continue
        audio_producer_tokens.update(
            token
            for token in (
                _clean_text(branch.get('phase_id')),
                _clean_text(branch.get('branch_id')),
            )
            if token
        )
    if not audio_producer_tokens:
        return False
    for branch in downstream_branches:
        if normalize_capability(branch.get('capability')) != CAPABILITY_SPEECH_TO_TEXT:
            continue
        dependencies = {
            _clean_text(item)
            for item in (branch.get('depends_on') or [])
            if _clean_text(item)
        }
        if dependencies.intersection(audio_producer_tokens):
            return True
    return False


def _current_phase_status(
    response_payload: Mapping[str, Any],
    *,
    current_capability: Optional[str],
) -> str:
    status = _clean_text(response_payload.get('status')).lower()
    if status in {'failed', 'error', 'cancelled'}:
        return 'blocked'
    if current_capability == CAPABILITY_CHAT:
        if _clean_text(response_payload.get('output_text')) or _artifact_matches_output_type(
            response_payload.get('artifacts'),
            'text',
        ):
            return 'completed'
        if response_payload:
            return 'active'
        return 'planned'
    output_type = _output_type_for_capability(current_capability)
    if output_type and _artifact_matches_output_type(response_payload.get('artifacts'), output_type):
        return 'completed'
    if response_payload:
        return 'active'
    return 'planned'


def _downstream_phase_status(
    branch: Mapping[str, Any],
    *,
    response_payload: Mapping[str, Any],
    branch_capability_counts: Optional[Mapping[str, int]] = None,
    branch_output_counts: Optional[Mapping[str, int]] = None,
) -> str:
    capability = normalize_capability(branch.get('capability'))
    branch_id = _clean_text(branch.get('branch_id') or branch.get('phase_id'))
    output_type = _output_type_for_capability(capability)
    capability_count = int((branch_capability_counts or {}).get(capability or '', 0))
    output_count = int((branch_output_counts or {}).get(output_type or '', 0))
    is_unique_capability_branch = capability_count <= 1
    is_unique_output_branch = output_count <= 1
    if (
        output_type
        and is_unique_capability_branch
        and is_unique_output_branch
        and _artifact_matches_output_type(response_payload.get('artifacts'), output_type)
    ):
        return 'completed'
    late_fill = response_payload.get('late_fill') if isinstance(response_payload.get('late_fill'), Mapping) else {}
    for key, status in (
        ('completed_branches', 'completed'),
        ('failed_branches', 'blocked'),
        ('pending_branches', 'pending'),
        ('active_branches', 'pending'),
    ):
        values = late_fill.get(key)
        if not isinstance(values, list):
            continue
        for raw_branch in values:
            normalized = _normalize_explicit_downstream_branch(raw_branch, fallback_index=1)
            if not normalized:
                continue
            if _clean_text(normalized.get('branch_id')) == branch_id:
                return status
    completed_capabilities = _unique_capabilities(
        late_fill.get('completed_capabilities') if isinstance(late_fill.get('completed_capabilities'), list) else []
    )
    if is_unique_capability_branch and capability in completed_capabilities:
        return 'completed'
    failed_capabilities = _unique_capabilities(
        late_fill.get('failed_capabilities') if isinstance(late_fill.get('failed_capabilities'), list) else []
    )
    if is_unique_capability_branch and capability in failed_capabilities:
        return 'blocked'
    expected_capability = normalize_capability(late_fill.get('expected_capability'))
    if is_unique_capability_branch and expected_capability == capability:
        return _phase_status_from_late_fill(_clean_text(late_fill.get('status')))
    return 'pending' if response_payload else 'planned'


def build_request_phase_graph(
    prompt: str,
    *,
    intent_prompt: Optional[str] = None,
    request_payload: Optional[Mapping[str, Any]] = None,
    route_payload: Optional[Mapping[str, Any]] = None,
    response_payload: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build a compact phase graph for the request currently under execution."""

    request = _mapping(request_payload)
    route = _mapping(route_payload)
    response = _mapping(response_payload)
    current_turn_prompt = _current_turn_prompt_from_request_payload(request)
    prompt_candidate = prompt
    if current_turn_prompt and _looks_like_serialized_responses_history(prompt_candidate):
        prompt_candidate = current_turn_prompt
    normalized_prompt = _clean_text(
        intent_prompt
        or prompt_candidate
        or current_turn_prompt
        or request.get('prompt')
        or request.get('input')
        or request.get('instructions')
        or response.get('output_text')
    )
    prompt_analysis = analyze_prompt_intent(normalized_prompt)
    text_artifact_requests = detect_text_artifact_requests(
        normalized_prompt,
        source_available=_has_source_artifact_available(request, route, response),
    )
    text_artifact_requests = _merge_text_artifact_requests(
        text_artifact_requests,
        _text_artifact_source_requests_from_history(
            normalized_prompt,
            request,
            route,
            response,
        ),
    )
    prompt_analysis, graph_refinements = _refine_prompt_analysis_from_response_output(
        prompt_analysis,
        response,
    )
    prompt_analysis = _apply_implicit_web_page_visual_binding(
        prompt_analysis,
        text_artifact_requests,
    )
    prompt_analysis, current_predecessor_image_prompts = (
        _apply_current_predecessor_image_prompt_contract(
            prompt_analysis,
            request,
            text_artifact_requests,
        )
    )
    refinement_capabilities = {
        normalize_capability(item.get('capability'))
        for item in graph_refinements
        if isinstance(item, Mapping) and normalize_capability(item.get('capability'))
    }
    request_meta = extract_request_meta(request)
    workload_task_proposals = _explicit_workload_task_proposals(request, route, response)
    ghost_owned_request = bool(request.get('ghost_route')) or (
        _clean_text(route.get('route_source')).lower() == 'ghost_carried'
    )
    wants_materialization_follow_up = bool(
        prompt_analysis.get('requests_audio_output')
        or prompt_analysis.get('has_audio_follow_up_request')
        or prompt_analysis.get('text_preparation_before_audio_output')
        or prompt_analysis.get('requests_visual_output')
        or prompt_analysis.get('has_visual_follow_up_request')
        or prompt_analysis.get('text_preparation_before_visual_output')
        or prompt_analysis.get('requests_speech_to_text_output')
        or _coerce_positive_int(prompt_analysis.get('requested_visual_output_count')) > 0
        or bool(text_artifact_requests)
    )
    downstream_branches = _downstream_branches(
        prompt_analysis,
        request_payload=request,
        request_meta=request_meta,
        route_payload=route,
        response_payload=response,
        refinement_capabilities=refinement_capabilities,
    )
    _retain_counted_audio_variant_contracts(
        downstream_branches,
        prompt_analysis=prompt_analysis,
    )
    _enforce_requested_visual_output_branch_count(
        downstream_branches,
        prompt_analysis=prompt_analysis,
        graph_refinements=graph_refinements,
        request_payload=request,
    )
    _bind_current_predecessor_image_prompts(
        downstream_branches,
        current_predecessor_image_prompts,
        graph_refinements,
    )
    _bind_direct_clause_local_media_payloads(
        downstream_branches,
        prompt_text=normalized_prompt,
        prompt_analysis=prompt_analysis,
        graph_refinements=graph_refinements,
    )
    downstream_branches.extend(
        _text_artifact_request_branches(
            text_artifact_requests,
            existing_branches=downstream_branches,
            prompt_analysis=prompt_analysis,
        )
    )
    preserved_reference_input_refs, preserved_reference_error = (
        _preserved_visual_reference_input_refs(
            prompt_analysis,
            request,
        )
    )
    _bind_explicit_structured_multi_evidence_join(
        downstream_branches,
        normalized_prompt,
        preserved_reference_input_refs=preserved_reference_input_refs,
        preserved_reference_error=preserved_reference_error,
    )
    intent_obligations = build_intent_obligation_ledger(
        prompt=normalized_prompt,
        prompt_analysis=prompt_analysis,
        text_artifact_requests=text_artifact_requests,
        downstream_branches=downstream_branches,
    )
    audio_cardinality_block = _audio_cardinality_blocking_obligation(prompt_analysis)
    if audio_cardinality_block:
        intent_obligations.append(audio_cardinality_block)
    apply_intent_obligation_dependency_edges(
        downstream_branches,
        intent_obligations,
    )
    promoted_downstream_branches = _promoted_branch_records(downstream_branches)
    promoted_current_branch: Optional[dict[str, Any]] = None
    if (
        promoted_downstream_branches
        and _has_audio_input_artifact(request)
        and normalize_capability(promoted_downstream_branches[0].get('capability')) == CAPABILITY_SPEECH_TO_TEXT
    ):
        promoted_current_branch = dict(promoted_downstream_branches[0])
        promoted_phase_id = _clean_text(promoted_current_branch.get('phase_id'))
        promoted_branch_id = _clean_text(promoted_current_branch.get('branch_id'))
        promoted_tokens = {token for token in (promoted_phase_id, promoted_branch_id) if token}
        downstream_branches = [
            dict(item)
            for item in downstream_branches
            if _clean_text(item.get('phase_id')) not in promoted_tokens
            and _clean_text(item.get('branch_id')) not in promoted_tokens
        ]
        promoted_downstream_branches = _promoted_branch_records(downstream_branches)
        if promoted_tokens:
            for branch in downstream_branches:
                depends_on = [
                    _clean_text(item)
                    for item in (branch.get('depends_on') or [])
                    if _clean_text(item)
                ]
                if any(item in promoted_tokens for item in depends_on):
                    branch['depends_on'] = [
                        'phase-1' if item in promoted_tokens else item
                        for item in depends_on
                    ]
                    if (
                        normalize_capability(promoted_current_branch.get('capability'))
                        == CAPABILITY_SPEECH_TO_TEXT
                        and _clean_text(branch.get('content_payload_source'))
                        not in {
                            'current_input_audio_artifact',
                            'selected_reference_audio_artifact',
                        }
                    ):
                        branch['content_payload_source'] = 'speech_to_text_branch_result'
                elif not depends_on:
                    branch['depends_on'] = ['phase-1']
    downstream_capabilities = _unique_capabilities(
        [item.get('capability') for item in promoted_downstream_branches]
    )
    direct_capability_candidates = [
        route.get('capability'),
        response.get('capability'),
        request.get('capability'),
        request_meta.get('capability_hint'),
        prompt_analysis.get('primary_capability'),
    ]
    direct_capability = normalize_capability(promoted_current_branch.get('capability')) if promoted_current_branch else None
    for candidate in direct_capability_candidates:
        if direct_capability:
            break
        normalized_candidate = normalize_capability(candidate)
        if not normalized_candidate:
            continue
        if (
            normalized_candidate == CAPABILITY_CHAT
            and normalize_capability(prompt_analysis.get('primary_capability')) == CAPABILITY_VISION_ANALYSIS
            and bool(prompt_analysis.get('separate_visual_analysis_request'))
        ):
            continue
        if _prompt_negates_materialization_capability(prompt_analysis, normalized_candidate):
            continue
        if _prompt_reserves_entire_materialization_capability(prompt_analysis, normalized_candidate):
            continue
        if _visual_execution_is_preserved(prompt_analysis, normalized_candidate):
            continue
        if (
            normalized_candidate == CAPABILITY_TEXT_TO_SPEECH
            and bool(prompt_analysis.get('audio_output_count_exceeds_bound'))
        ):
            continue
        if (
            ghost_owned_request
            and not wants_materialization_follow_up
            and normalized_candidate in _GHOST_PREPARE_FIRST_MATERIALIZATION_CAPABILITIES
        ):
            continue
        direct_capability = normalized_candidate
        break
    current_phase_capability = direct_capability or CAPABILITY_CHAT
    current_phase_kind = 'materialize'
    current_phase_role = 'single_phase_response'
    current_phase_resolution = 'router_required'
    current_phase_reason = None
    if promoted_downstream_branches:
        if promoted_current_branch:
            current_phase_capability = normalize_capability(promoted_current_branch.get('capability')) or current_phase_capability
            current_phase_kind = 'materialize'
            current_phase_role = f'{current_phase_capability}_evidence'
        elif (
            direct_capability in {CAPABILITY_VISION_ANALYSIS, CAPABILITY_SPEECH_TO_TEXT}
            and not _direct_stt_is_generated_audio_follow_up(
                direct_capability,
                promoted_downstream_branches,
            )
        ):
            current_phase_capability = direct_capability
            current_phase_kind = 'evidence'
            current_phase_role = f'{current_phase_capability}_evidence'
        else:
            current_phase_capability = CAPABILITY_CHAT
            current_phase_kind = 'prepare'
            current_phase_role = 'text_preparation'
        current_phase_resolution = 'graph_resolved'
        current_phase_reason = (
            'current phase materializes input audio evidence before dependent follow-up phases'
            if promoted_current_branch
            else _current_phase_reason(
                downstream_capabilities,
                prompt_analysis=prompt_analysis,
                route_payload=route,
            )
        )
    current_phase = {
        'phase_id': 'phase-1',
        'obligation_id': 'obligation-phase-1',
        'kind': current_phase_kind,
        'role': current_phase_role,
        'capability': current_phase_capability,
        'output_type': _output_type_for_capability(current_phase_capability),
        'status': _current_phase_status(
            response,
            current_capability=current_phase_capability,
        ),
        'depends_on': [],
        'resolution': current_phase_resolution,
    }
    if current_phase_reason:
        current_phase['reason'] = current_phase_reason

    phases: list[dict[str, Any]] = [current_phase]
    downstream_phase_ids: list[str] = []
    branch_capability_counts: dict[str, int] = {}
    branch_output_counts: dict[str, int] = {}
    for branch in promoted_downstream_branches:
        capability = normalize_capability(branch.get('capability'))
        output_type = _clean_text(branch.get('output_type')).lower() or _output_type_for_capability(capability)
        if capability:
            branch_capability_counts[capability] = branch_capability_counts.get(capability, 0) + 1
        if output_type:
            branch_output_counts[output_type] = branch_output_counts.get(output_type, 0) + 1
    for index, branch in enumerate(downstream_branches, start=2):
        capability = normalize_capability(branch.get('capability'))
        phase_id = _clean_text(branch.get('phase_id')) or f'phase-{index}'
        branch_id = _clean_text(branch.get('branch_id')) or phase_id
        if not _is_unpromoted_candidate_record(branch):
            downstream_phase_ids.append(phase_id)
        phase_record = {
            'phase_id': phase_id,
            'branch_id': branch_id,
            'obligation_id': f'obligation-{phase_id}',
            'kind': _clean_text(branch.get('kind')).lower() or 'materialize',
            'role': _clean_text(branch.get('role')) or f'{capability}_follow_up',
            'capability': capability,
            'output_type': _clean_text(branch.get('output_type')).lower() or _output_type_for_capability(capability),
            'status': _downstream_phase_status(
                branch,
                response_payload=response,
                branch_capability_counts=branch_capability_counts,
                branch_output_counts=branch_output_counts,
            ),
            'depends_on': [
                _clean_text(item)
                for item in (branch.get('depends_on') or [])
                if _clean_text(item)
            ] or ['phase-1'],
            'resolution': _clean_text(branch.get('resolution')).lower() or 'pending_dependency',
        }
        queue_index = _coerce_positive_int(branch.get('queue_index'))
        if queue_index:
            phase_record['queue_index'] = queue_index
        source = _clean_text(branch.get('source'))
        if source:
            phase_record['source'] = source
        required = branch.get('required')
        if isinstance(required, bool) or _clean_text(required):
            phase_record['required'] = required
        if capability in refinement_capabilities:
            phase_record['refinement_source'] = 'assistant_output_claim'
        for key in (
            'candidate_id',
            'contract_state',
            'contract_status',
            'obligation_state',
            'intent_state',
            'promotion_policy',
            'promotion_reason',
            'promotion_source',
            'promoted_from_candidate_id',
            'artifact_prompt',
            'artifact_prompt_source',
            'content_payload',
            'content_payload_source',
            'phase_summary',
            'stage_direction',
            'candidate_selection_index',
            'candidate_selection_count',
            'selection_policy',
            'selection_reason',
            'lang_code',
            'audio_variant_index',
            'audio_variant_role',
            'audio_variant_contract_source',
            'structured_output_contract',
            'branch_contract_error',
            'audio_variant_contract_conflicting_fields',
            'requires_artifact',
            'dependency_contract',
            'image_asset_binding_required',
            'required_image_phase_ids',
            'text_artifact_extension',
            'text_artifact_source_name',
            'text_artifact_source',
            'text_artifact_target_path',
            'artifact_request',
            'kind',
            'role',
            'semantic_intent',
            'objective',
            'deliverable',
            'rationale',
            'review_criteria',
            'input_refs',
            'execution_contract',
            'workload_task_ref',
            'output_obligation_ref',
            'output_contract',
            'accepted_proposals',
            'repair_action',
            'recovery_action',
            'repair_action_reason',
            'blocked_by_dependency_input',
            'blocked_by_branch_contract',
        ):
            value = branch.get(key)
            if value not in (None, '', [], {}):
                phase_record[key] = value
        batch_prompts = _clean_string_list(branch.get('batch_prompts'))
        if batch_prompts:
            phase_record['batch_prompts'] = batch_prompts
        if isinstance(branch.get('promotion'), Mapping):
            phase_record['promotion'] = dict(branch.get('promotion') or {})
        phases.append(phase_record)

    if downstream_phase_ids:
        current_phase['downstream_phase_ids'] = downstream_phase_ids

    graph_mode = 'single_phase'
    if promoted_downstream_branches:
        graph_mode = 'phase_chain'
    if _clean_text(route.get('route_source')).lower() == 'ghost_carried' and promoted_downstream_branches:
        graph_mode = 'carried_phase_chain'

    downstream_branch_records: list[dict[str, Any]] = []
    for item in downstream_branches:
        branch_id = _clean_text(item.get('branch_id') or item.get('phase_id'))
        phase_id = _clean_text(item.get('phase_id') or item.get('branch_id'))
        capability = normalize_capability(item.get('capability'))
        if not branch_id or not capability:
            continue
        payload = {
            'branch_id': branch_id,
            'phase_id': phase_id or branch_id,
            'obligation_id': f'obligation-{phase_id or branch_id}',
            'capability': capability,
            'output_type': _clean_text(item.get('output_type')).lower() or _output_type_for_capability(item.get('capability')),
            'depends_on': [
                _clean_text(dep)
                for dep in (item.get('depends_on') or [])
                if _clean_text(dep)
            ] or ['phase-1'],
        }
        queue_index = _coerce_positive_int(item.get('queue_index'))
        if queue_index:
            payload['queue_index'] = queue_index
        source = _clean_text(item.get('source'))
        if source:
            payload['source'] = source
        required = item.get('required')
        if isinstance(required, bool) or _clean_text(required):
            payload['required'] = required
        if capability in refinement_capabilities:
            payload['refinement_source'] = 'assistant_output_claim'
        for key in (
            'candidate_id',
            'contract_state',
            'contract_status',
            'obligation_state',
            'intent_state',
            'promotion_policy',
            'promotion_reason',
            'promotion_source',
            'promoted_from_candidate_id',
            'artifact_prompt',
            'artifact_prompt_source',
            'content_payload',
            'content_payload_source',
            'phase_summary',
            'stage_direction',
            'candidate_selection_index',
            'candidate_selection_count',
            'selection_policy',
            'selection_reason',
            'lang_code',
            'audio_variant_index',
            'audio_variant_role',
            'audio_variant_contract_source',
            'structured_output_contract',
            'branch_contract_error',
            'audio_variant_contract_conflicting_fields',
            'requires_artifact',
            'dependency_contract',
            'image_asset_binding_required',
            'required_image_phase_ids',
            'text_artifact_extension',
            'text_artifact_source_name',
            'text_artifact_source',
            'text_artifact_target_path',
            'artifact_request',
            'kind',
            'role',
            'semantic_intent',
            'objective',
            'deliverable',
            'rationale',
            'review_criteria',
            'input_refs',
            'execution_contract',
            'workload_task_ref',
            'output_obligation_ref',
            'output_contract',
            'accepted_proposals',
            'repair_action',
            'recovery_action',
            'repair_action_reason',
            'blocked_by_dependency_input',
            'blocked_by_branch_contract',
        ):
            value = item.get(key)
            if value not in (None, '', [], {}):
                payload[key] = value
        batch_prompts = _clean_string_list(item.get('batch_prompts'))
        if batch_prompts:
            payload['batch_prompts'] = batch_prompts
        if isinstance(item.get('promotion'), Mapping):
            payload['promotion'] = dict(item.get('promotion') or {})
        downstream_branch_records.append(payload)

    required_obligations = required_intent_obligations(intent_obligations)
    required_intent_summary = summarize_required_intent_obligations(required_obligations)
    required_capabilities = list(required_intent_summary.get('capabilities') or [])
    required_capability_counts = dict(required_intent_summary.get('capability_counts') or {})
    required_output_counts = dict(required_intent_summary.get('material_output_counts') or {})
    required_phase_ids_by_capability: dict[str, set[str]] = {}
    for obligation in required_obligations:
        capability = normalize_capability(obligation.get('capability'))
        phase_id = _clean_text(obligation.get('phase_id') or obligation.get('branch_id'))
        if capability and phase_id:
            required_phase_ids_by_capability.setdefault(capability, set()).add(phase_id)
    preparation_capabilities: set[str] = set()
    if current_phase_capability == CAPABILITY_CHAT:
        for branch in downstream_branch_records:
            capability = normalize_capability(branch.get('capability'))
            phase_id = _clean_text(branch.get('phase_id') or branch.get('branch_id'))
            depends_on = {
                _clean_text(item)
                for item in (branch.get('depends_on') or [])
                if _clean_text(item)
            }
            if (
                capability
                and phase_id in required_phase_ids_by_capability.get(capability, set())
                and 'phase-1' in depends_on
            ):
                preparation_capabilities.add(capability)
    direct_spoken_payload_bound = any(
        normalize_capability(branch.get('capability')) == CAPABILITY_TEXT_TO_SPEECH
        and _clean_text(branch.get('content_payload'))
        and _clean_text(branch.get('content_payload_source'))
        == 'current_turn_direct_spoken_clause'
        for branch in downstream_branch_records
    )

    prompt_intent = {
        'primary_capability': normalize_capability(prompt_analysis.get('primary_capability')),
        'direct_audio_materialization_request': bool(prompt_analysis.get('direct_audio_materialization_request')),
        'explicit_defer_materialization': bool(prompt_analysis.get('explicit_defer_materialization')),
        'explicit_visual_defer_materialization': bool(prompt_analysis.get('explicit_visual_defer_materialization')),
        'explicit_audio_defer_materialization': bool(prompt_analysis.get('explicit_audio_defer_materialization')),
        'visual_artifact_preservation_without_regeneration': bool(
            prompt_analysis.get('visual_artifact_preservation_without_regeneration')
        ),
        'visual_analysis_preservation_without_reanalysis': bool(
            prompt_analysis.get('visual_analysis_preservation_without_reanalysis')
        ),
        'visual_preservation_cues': _clean_string_list(
            prompt_analysis.get('visual_preservation_cues')
        ),
        'separate_visual_generation_request': bool(
            prompt_analysis.get('separate_visual_generation_request')
        ),
        'separate_visual_analysis_request': bool(
            prompt_analysis.get('separate_visual_analysis_request')
        ),
        'separate_visual_work_cues': _clean_string_list(
            prompt_analysis.get('separate_visual_work_cues')
        ),
        'visual_artifact_execution_suppressed_by_preservation': bool(
            prompt_analysis.get('visual_artifact_execution_suppressed_by_preservation')
        ),
        'visual_analysis_execution_suppressed_by_preservation': bool(
            prompt_analysis.get('visual_analysis_execution_suppressed_by_preservation')
        ),
        'requests_audio_output': bool(
            prompt_analysis.get('requests_audio_output')
            or required_output_counts.get('audio')
        ),
        'requests_visual_output': bool(
            prompt_analysis.get('requests_visual_output')
            or required_output_counts.get('image')
        ),
        'counted_visual_output_obligation': bool(prompt_analysis.get('counted_visual_output_obligation')),
        'local_visual_asset_requirement': bool(prompt_analysis.get('local_visual_asset_requirement')),
        'local_visual_asset_cues': _clean_string_list(prompt_analysis.get('local_visual_asset_cues')),
        'inferred_visual_output_count_source': _clean_text(
            prompt_analysis.get('inferred_visual_output_count_source')
        ),
        'requests_speech_to_text_output': bool(
            prompt_analysis.get('requests_speech_to_text_output')
            or required_capability_counts.get(CAPABILITY_SPEECH_TO_TEXT)
        ),
        'input_audio_artifact_promoted_to_stt': bool(
            _has_audio_input_artifact(request)
            and _prompt_references_current_input_audio(prompt_analysis)
            and current_phase_capability == CAPABILITY_SPEECH_TO_TEXT
        ),
        'requested_visual_output_count': max(
            _coerce_positive_int(prompt_analysis.get('requested_visual_output_count')),
            _coerce_positive_int(required_output_counts.get('image')),
        ),
        'counted_audio_output_obligation': bool(
            prompt_analysis.get('counted_audio_output_obligation')
        ),
        'requested_audio_output_count': (
            0
            if bool(prompt_analysis.get('audio_output_count_exceeds_bound'))
            else max(
                _coerce_positive_int(prompt_analysis.get('requested_audio_output_count')),
                _coerce_positive_int(required_output_counts.get('audio')),
            )
        ),
        'requested_audio_output_count_raw': _coerce_positive_int(
            prompt_analysis.get('requested_audio_output_count_raw')
        ),
        'audio_output_count_exceeds_bound': bool(
            prompt_analysis.get('audio_output_count_exceeds_bound')
        ),
        'requested_audio_output_count_max': _coerce_positive_int(
            prompt_analysis.get('requested_audio_output_count_max')
        ),
        'blocked_audio_output_cardinality': bool(audio_cardinality_block),
        'needs_clarification': bool(audio_cardinality_block),
        'blocking_intent_obligation_ids': (
            [audio_cardinality_block['obligation_id']]
            if audio_cardinality_block
            else []
        ),
        'text_preparation_before_audio_output': bool(
            prompt_analysis.get('text_preparation_before_audio_output')
            or (
                CAPABILITY_TEXT_TO_SPEECH in preparation_capabilities
                and not direct_spoken_payload_bound
            )
        ),
        'text_preparation_before_visual_output': bool(
            prompt_analysis.get('text_preparation_before_visual_output')
            or CAPABILITY_IMAGE_GENERATION in preparation_capabilities
        ),
        'downstream_follow_up_capabilities': required_capabilities,
        'requests_text_artifact_output': bool(text_artifact_requests),
        'text_artifact_output_count': len(text_artifact_requests),
        'text_artifact_extensions': [
            _clean_text(item.get('extension')).lower()
            for item in text_artifact_requests
            if _clean_text(item.get('extension'))
        ],
        'intent_obligation_count': len(intent_obligations),
        'required_intent_obligation_count': int(required_intent_summary.get('required_count') or 0),
        'intent_obligation_kinds': list(
            dict.fromkeys(
                _clean_text(item.get('kind'))
                for item in intent_obligations
                if _clean_text(item.get('kind'))
            )
        ),
        'required_intent_obligation_kinds': list(required_intent_summary.get('kinds') or []),
        'required_intent_capabilities': required_capabilities,
        'required_intent_capability_counts': required_capability_counts,
        'required_intent_output_counts': required_output_counts,
        'text_revision_turn': bool(prompt_analysis.get('text_revision_turn')),
        'refined_from_output_claim': bool(graph_refinements),
    }
    request_ir = build_request_ir(
        intent_prompt=normalized_prompt,
        prompt_intent=prompt_intent,
        phases=phases,
        current_phase_id='phase-1',
        graph_mode=graph_mode,
        workload_task_proposals=workload_task_proposals,
        accepted_learning_hints=_accepted_learning_hints_from_payloads(request, route, response),
        semantic_role_profile=_semantic_role_profile_from_payloads(request, route, response),
    )
    output_obligations = request_ir.get('output_obligations') if isinstance(request_ir.get('output_obligations'), list) else []
    output_candidates = request_ir.get('output_candidates') if isinstance(request_ir.get('output_candidates'), list) else []
    promotions = request_ir.get('promotions') if isinstance(request_ir.get('promotions'), list) else []
    workload_graph = request_ir.get('workload_graph') if isinstance(request_ir.get('workload_graph'), Mapping) else {}
    workload_task_ids = request_ir.get('workload_task_ids') if isinstance(request_ir.get('workload_task_ids'), list) else []
    workload_task_lookup = _workload_task_lookup(workload_graph)
    if workload_task_lookup:
        phases = [
            _project_workload_task_fields(phase, workload_task_lookup)
            for phase in phases
        ]
        downstream_branch_records = [
            _project_workload_task_fields(branch, workload_task_lookup)
            for branch in downstream_branch_records
        ]
    workload_proposal_review = (
        request_ir.get('workload_proposal_review')
        if isinstance(request_ir.get('workload_proposal_review'), Mapping)
        else {}
    )
    candidate_graph = (
        request_ir.get('candidate_graph')
        if isinstance(request_ir.get('candidate_graph'), Mapping)
        else {}
    )
    promotion_review = (
        request_ir.get('promotion_review')
        if isinstance(request_ir.get('promotion_review'), Mapping)
        else {}
    )
    decision_contract = (
        request_ir.get('decision_contract')
        if isinstance(request_ir.get('decision_contract'), Mapping)
        else {}
    )
    graph_payload = {
        'graph_version': REQUEST_PHASE_GRAPH_VERSION,
        'kind': 'ollmo.request_phase_graph',
        'mode': graph_mode,
        'request_ir': request_ir,
        'output_obligations': output_obligations,
        'output_candidates': output_candidates,
        'promotions': promotions,
        'candidate_graph': candidate_graph,
        'promotion_review': promotion_review,
        'decision_contract': decision_contract,
        'workload_graph': workload_graph,
        'workload_task_ids': workload_task_ids,
        'workload_proposal_review': workload_proposal_review,
        'prompt': normalized_prompt or None,
        'current_phase_id': 'phase-1',
        'current_phase_capability': current_phase_capability,
        'current_phase_resolution': current_phase_resolution,
        'downstream_phase_ids': downstream_phase_ids,
        'downstream_branch_ids': [
            _clean_text(item.get('branch_id') or item.get('phase_id'))
            for item in promoted_downstream_branches
            if _clean_text(item.get('branch_id') or item.get('phase_id'))
        ],
        'downstream_branches': downstream_branch_records,
        'downstream_capabilities': downstream_capabilities,
        'is_multi_phase': bool(promoted_downstream_branches),
        'continuation_required': bool(promoted_downstream_branches),
        'phases': phases,
        'prompt_intent': prompt_intent,
        'intent_obligations': intent_obligations,
    }
    if audio_cardinality_block:
        graph_payload.update(
            {
                'blocked_by_intent_contract': True,
                'needs_clarification': True,
                'blocking_intent_obligation_ids': [
                    audio_cardinality_block['obligation_id']
                ],
            }
        )
    if graph_refinements:
        graph_payload['graph_refinements'] = graph_refinements
    return graph_payload


def current_phase_capability(phase_graph: Optional[Mapping[str, Any]]) -> Optional[str]:
    if not isinstance(phase_graph, Mapping):
        return None
    return normalize_capability(phase_graph.get('current_phase_capability'))


def downstream_phase_capabilities(phase_graph: Optional[Mapping[str, Any]]) -> list[str]:
    if not isinstance(phase_graph, Mapping):
        return []
    return _unique_capabilities([item.get('capability') for item in downstream_phase_records(phase_graph)])


def downstream_phase_records(phase_graph: Optional[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(item) for item in _downstream_phase_records(phase_graph)]


def next_executable_downstream_branches(
    phase_graph: Optional[Mapping[str, Any]],
    *,
    pending_branches: Optional[list[Mapping[str, Any]]] = None,
    completed_branches: Optional[list[Mapping[str, Any]]] = None,
    failed_branches: Optional[list[Mapping[str, Any]]] = None,
    pending_capabilities: Optional[list[Any]] = None,
    completed_capabilities: Optional[list[Any]] = None,
    failed_capabilities: Optional[list[Any]] = None,
) -> list[dict[str, Any]]:
    phases = _phase_records(phase_graph)
    pending = [
        _normalize_explicit_downstream_branch(item, fallback_index=index)
        for index, item in enumerate(
            pending_branches if isinstance(pending_branches, list) and pending_branches else downstream_phase_records(phase_graph),
            start=1,
        )
    ]
    pending = [item for item in pending if item and not _is_unpromoted_candidate_record(item)]
    if not phases or not pending:
        if pending:
            return [dict(item) for item in pending[:1]]
        return []

    completed_phase_ids = _completed_phase_ids(phase_graph)
    failed_phase_ids = _failed_phase_ids(phase_graph)
    for branch_list, target in ((completed_branches, completed_phase_ids), (failed_branches, failed_phase_ids)):
        if not isinstance(branch_list, list):
            continue
        for index, item in enumerate(branch_list, start=1):
            normalized = _normalize_explicit_downstream_branch(item, fallback_index=index)
            phase_id = _clean_text((normalized or {}).get('phase_id'))
            if not phase_id and isinstance(item, Mapping):
                phase_id = _clean_text(item.get('phase_id') or item.get('branch_id'))
            if phase_id:
                target.add(phase_id)
    for capability in _unique_capabilities(completed_capabilities or []):
        phase_ids = _phase_ids_for_capability(phase_graph, capability)
        if len(phase_ids) == 1:
            completed_phase_ids.add(phase_ids[0])
    for capability in _unique_capabilities(failed_capabilities or []):
        phase_ids = _phase_ids_for_capability(phase_graph, capability)
        if len(phase_ids) == 1:
            failed_phase_ids.add(phase_ids[0])

    ready: list[dict[str, Any]] = []
    for branch in pending:
        phase_id = _clean_text(branch.get('phase_id'))
        if not phase_id or phase_id in completed_phase_ids or phase_id in failed_phase_ids:
            continue
        phase = next((item for item in phases if _clean_text(item.get('phase_id')) == phase_id), None)
        if phase is None:
            phase = dict(branch)
        depends_on = [
            _clean_text(item)
            for item in (phase.get('depends_on') or branch.get('depends_on') or [])
            if _clean_text(item)
        ]
        if any(dep_id in failed_phase_ids for dep_id in depends_on):
            continue
        if all(dep_id in completed_phase_ids for dep_id in depends_on):
            ready.append(
                {
                    **dict(phase),
                    **{
                        key: branch.get(key)
                        for key in (
                            'queue_index',
                            'artifact_prompt',
                            'artifact_prompt_source',
                            'content_payload',
                            'content_payload_source',
                            'phase_summary',
                            'stage_direction',
                            'repair_scope',
                            'resource_class',
                            'dependency_policy',
                            'runtime_scheduling_context',
                            'allow_gpu_heavy_concurrency',
                            'candidate_selection_index',
                            'candidate_selection_count',
                            'selection_policy',
                            'lang_code',
                            'audio_variant_index',
                            'audio_variant_role',
                            'audio_variant_contract_source',
                            'structured_output_contract',
                            'branch_contract_error',
                            'audio_variant_contract_conflicting_fields',
                            'batch_prompts',
                            'semantic_intent',
                            'objective',
                            'deliverable',
                            'rationale',
                            'review_criteria',
                            'input_refs',
                            'execution_contract',
                            'workload_task_ref',
                            'output_obligation_ref',
                            'output_contract',
                            'accepted_proposals',
                            'requires_artifact',
                            'text_artifact_extension',
                            'text_artifact_source_name',
                            'text_artifact_source',
                            'text_artifact_target_path',
                            'artifact_request',
                            'repair_action',
                            'recovery_action',
                            'repair_action_reason',
                            'repair_contract',
                            'repair_contract_id',
                            'repair_contract_status',
                            'repair_execution_policy',
                            'repair_promotion_source',
                            'contract_state',
                            'promotion_source',
                            'reconsideration_rebuild',
                            'decision_contract_block_resolution_signals',
                            'decision_contract_active_reconsideration_decisions',
                            'decision_contract_semantic_quality_contracts',
                            'block_resolution_reflex',
                            'block_resolution_signal',
                            'block_resolution_action',
                            'block_resolution_policy',
                            'reconsideration_reflex',
                            'active_reconsideration_review',
                            'active_reconsideration_decision',
                            'active_reconsideration_action',
                            'active_reconsideration_review_type',
                            'semantic_quality_review',
                            'semantic_quality_contract',
                            'semantic_quality_status',
                            'semantic_quality_review_id',
                            'recursive_cycle_review',
                            'recursive_cycle_state',
                            'semantic_decision_review',
                            'semantic_decision_proposal',
                            'semantic_decision_action',
                            'semantic_decision_confidence',
                            'semantic_decision_reason',
                            'semantic_review_verdict',
                            'semantic_review_verdict_status',
                            'semantic_review_recommended_transition',
                            'branch_semantic_review',
                            'branch_semantic_review_branch_id',
                            'branch_semantic_review_phase_id',
                            'branch_semantic_review_status',
                            'branch_semantic_review_reason',
                            'branch_semantic_review_source_branch_id',
                            'branch_semantic_review_source_phase_id',
                            'decision_contract_semantic_decision_proposals',
                            'controlled_attention_review',
                            'controlled_attention_frame',
                            'controlled_attention_frame_id',
                            'controlled_attention_scope',
                            'controlled_attention_priority',
                            'controlled_attention_question',
                            'controlled_attention_allowed_transitions',
                            'decision_contract_controlled_attention_frames',
                            'global_semantic_closure_review',
                            'global_semantic_closure_proposal',
                            'global_semantic_closure_status',
                            'global_semantic_closure_reason',
                            'global_semantic_closure_confidence',
                            'surface_state',
                            'blocked_by_dependency_input',
                            'blocked_by_branch_contract',
                        )
                        if branch.get(key) not in (None, '', [], {})
                    },
                    'branch_id': _clean_text(branch.get('branch_id') or phase.get('branch_id') or phase_id),
                    'phase_id': phase_id,
                    'capability': normalize_capability(branch.get('capability') or phase.get('capability')),
                    'output_type': _clean_text(branch.get('output_type') or phase.get('output_type')).lower()
                    or _output_type_for_capability(branch.get('capability') or phase.get('capability')),
                }
            )
    if ready:
        return ready
    return []


def downstream_phase_branch_batches(
    phase_graph: Optional[Mapping[str, Any]],
    *,
    pending_branches: Optional[list[Mapping[str, Any]]] = None,
    completed_branches: Optional[list[Mapping[str, Any]]] = None,
    failed_branches: Optional[list[Mapping[str, Any]]] = None,
    pending_capabilities: Optional[list[Any]] = None,
    completed_capabilities: Optional[list[Any]] = None,
    failed_capabilities: Optional[list[Any]] = None,
) -> list[list[dict[str, Any]]]:
    remaining = [
        _normalize_explicit_downstream_branch(item, fallback_index=index)
        for index, item in enumerate(
            pending_branches if isinstance(pending_branches, list) and pending_branches else downstream_phase_records(phase_graph),
            start=1,
        )
    ]
    remaining = [item for item in remaining if item and not _is_unpromoted_candidate_record(item)]
    completed = [
        _normalize_explicit_downstream_branch(item, fallback_index=index)
        for index, item in enumerate(completed_branches or [], start=1)
    ]
    completed = [item for item in completed if item]
    failed = [
        _normalize_explicit_downstream_branch(item, fallback_index=index)
        for index, item in enumerate(failed_branches or [], start=1)
    ]
    failed = [item for item in failed if item]
    batches: list[list[dict[str, Any]]] = []
    while remaining:
        batch = next_executable_downstream_branches(
            phase_graph,
            pending_branches=remaining,
            completed_branches=completed,
            failed_branches=failed,
            pending_capabilities=pending_capabilities,
            completed_capabilities=completed_capabilities,
            failed_capabilities=failed_capabilities,
        )
        if not batch:
            break
        batches.append(batch)
        completed.extend(item for item in batch if item not in completed)
        completed_ids = {_clean_text(item.get('branch_id') or item.get('phase_id')) for item in completed}
        remaining = [
            item
            for item in remaining
            if _clean_text(item.get('branch_id') or item.get('phase_id')) not in completed_ids
        ]
    return batches


def next_executable_downstream_capabilities(
    phase_graph: Optional[Mapping[str, Any]],
    *,
    pending_capabilities: Optional[list[Any]] = None,
    completed_capabilities: Optional[list[Any]] = None,
    failed_capabilities: Optional[list[Any]] = None,
) -> list[str]:
    batch = next_executable_downstream_branches(
        phase_graph,
        pending_capabilities=pending_capabilities,
        completed_capabilities=completed_capabilities,
        failed_capabilities=failed_capabilities,
    )
    return _unique_capabilities([item.get('capability') for item in batch])


def downstream_phase_batches(
    phase_graph: Optional[Mapping[str, Any]],
    *,
    pending_capabilities: Optional[list[Any]] = None,
    completed_capabilities: Optional[list[Any]] = None,
    failed_capabilities: Optional[list[Any]] = None,
) -> list[list[str]]:
    batches = downstream_phase_branch_batches(
        phase_graph,
        pending_capabilities=pending_capabilities,
        completed_capabilities=completed_capabilities,
        failed_capabilities=failed_capabilities,
    )
    return [_unique_capabilities([item.get('capability') for item in batch]) for batch in batches if batch]


def current_phase_is_graph_resolved(phase_graph: Optional[Mapping[str, Any]]) -> bool:
    if not isinstance(phase_graph, Mapping):
        return False
    return _clean_text(phase_graph.get('current_phase_resolution')).lower() == 'graph_resolved'


def current_phase_reason(phase_graph: Optional[Mapping[str, Any]]) -> Optional[str]:
    if not isinstance(phase_graph, Mapping):
        return None
    phases = phase_graph.get('phases')
    if not isinstance(phases, list):
        return None
    for raw_phase in phases:
        if not isinstance(raw_phase, Mapping):
            continue
        if _clean_text(raw_phase.get('phase_id')) != _clean_text(phase_graph.get('current_phase_id')):
            continue
        reason = _clean_text(raw_phase.get('reason'))
        return reason or None
    return None
