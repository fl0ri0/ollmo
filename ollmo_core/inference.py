"""Shared inference service layer for Ollmo capability dispatch."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from helpers.model_capabilities import (
    CAPABILITY_IMAGE_GENERATION,
    CAPABILITY_SPEECH_TO_TEXT,
    CAPABILITY_TEXT_TO_SPEECH,
    CAPABILITY_VISION_ANALYSIS,
)
from helpers.ocr_modes import GENERIC_OCR_FALLBACK_PROMPT, resolve_ocr_prompt
from ollmo_core.transports import (
    join_pcm_wav_bytes,
    text_artifact_content_is_materializer_instruction_echo,
)
from ollmo_services.tts_audio_integrity import (
    TTS_QWEN_SENTENCE_CHUNK_INTEGRITY_PROFILE,
    build_tts_audio_integrity_evidence,
    build_tts_semantic_source,
    tts_audio_has_qwen_generation_limit_exhaustion,
)
from ollmo_services.tts_source import extract_legacy_tts_wrapper_text

TEXT_ARTIFACT_EXTENSIONS = {
    'txt',
    'md',
    'markdown',
    'html',
    'htm',
    'css',
    'js',
    'mjs',
    'cjs',
    'ts',
    'tsx',
    'jsx',
    'json',
    'yaml',
    'yml',
    'xml',
    'csv',
    'svg',
    'py',
    'sh',
    'sql',
}
_TEXT_ARTIFACT_EXTENSION_RE = re.compile(
    r'(?P<name>[A-Za-z0-9][A-Za-z0-9._-]{0,80})\.(?P<ext>'
    + '|'.join(sorted(TEXT_ARTIFACT_EXTENSIONS, key=len, reverse=True))
    + r')\b',
    re.IGNORECASE,
)
_TEXT_ARTIFACT_FILE_CUE_RE = re.compile(
    r'\b('
    r'file|files|artifact|artifacts|artefact|artefacts|artefakt|artefakte|datei|dateien|'
    r'document|documents|dokument|dokumente|export|download|downloadable|'
    r'save|saved|speicher|speichere|persist|persistiere|'
    r'materialize|materialized|materialise|materialised|materialisiere|materialisieren|materialisiert|'
    r'write\s+to|as\s+a\s+file|als\s+datei'
    r')\b',
    re.IGNORECASE,
)
_TEXT_ARTIFACT_ACTION_CUE_RE = re.compile(
    r'\b('
    r'create|generate|make|write|produce|return|provide|build|emit|design|draft|'
    r'change|modify|update|edit|revise|alter|save|materialize|materialise|'
    r'erzeuge|erstelle|generiere|schreibe|gib|liefere|baue|'
    r'entwirf|entwerfe|entwerfen|gestalte|gestalten|'
    r'aendere|ändere|veraendere|verändere|anpassen|passe|speichere|'
    r'materialisiere|materialisieren'
    r')\b',
    re.IGNORECASE,
)
_TEXT_ARTIFACT_SOURCE_EDIT_CUE_RE = re.compile(
    r'\b('
    r'change|modify|update|edit|revise|alter|replace|'
    r'add|append|insert|include|remove|delete|section|'
    r'style|restyle|color|colour|font|fix|adjust|tweak|'
    r'aendere|ändere|veraendere|verändere|anpassen|passe|ersetze|'
    r'fuege|füge|hinzufuegen|hinzufügen|entferne|loesche|lösche|abschnitt|'
    r'farbe|schrift|rot|red'
    r')\b',
    re.IGNORECASE,
)
_TEXT_ARTIFACT_SOURCE_EDIT_ACTION_RE = re.compile(
    r'\b('
    r'change|modify|update|edit|revise|alter|replace|'
    r'add|append|insert|include|remove|delete|style|restyle|color|colour|fix|adjust|tweak|'
    r'aendere|ändere|veraendere|verändere|aktualisiere|anpassen|passe|ersetze|'
    r'fuege|füge|hinzufuegen|hinzufügen|entferne|loesche|lösche|'
    r'gestalte|korrigiere|repariere'
    r')\b',
    re.IGNORECASE,
)
_TEXT_ARTIFACT_SOURCE_EDIT_NEXT_DIRECTIVE_RE = re.compile(
    r'\b('
    r'change|modify|update|edit|revise|alter|replace|add|append|insert|include|remove|delete|'
    r'style|restyle|color|colour|fix|adjust|tweak|'
    r'create|generate|make|write|produce|return|provide|build|emit|save|materialize|materialise|'
    r'transcribe|analyse|analyze|preserve|'
    r'aendere|ändere|veraendere|verändere|aktualisiere|anpassen|passe|ersetze|'
    r'fuege|füge|hinzufuegen|hinzufügen|entferne|loesche|lösche|gestalte|korrigiere|repariere|'
    r'erzeuge|erstelle|generiere|schreibe|gib|liefere|baue|speichere|materialisiere|'
    r'transkribiere|analysiere|bewahre'
    r')\b',
    re.IGNORECASE,
)
_TEXT_ARTIFACT_SOURCE_EDIT_CLAUSE_RE = re.compile(r'[^.;:!?\n]+')
_TEXT_ARTIFACT_SOURCE_EDIT_EXPLICIT_SOURCE_RE = re.compile(
    r'\b(?:'
    r'(?:(?:this|that|the|selected|current|existing|attached|uploaded|provided|opened|updated)\s+){1,3}'
    r'(?:file|document|artifact|artefact|source|text|page|checklist|readme)|'
    r'(?:(?:diese|dieses|diesen|dieser|die|das|den|ausgewahlte|ausgewählte|aktuelle|'
    r'bestehende|angehangte|angehängte|hochgeladene|bereitgestellte|geoffnete|geöffnete)\s+){1,3}'
    r'(?:datei|dokument|artefakt|quelle|text|seite|checkliste)'
    r')\b',
    re.IGNORECASE,
)
_TEXT_ARTIFACT_SOURCE_EDIT_DIRECT_PRONOUN_RE = re.compile(
    r'^\s*(?:(?:please|bitte)\s+)?(?:it|this|that|these|those|dies|diese|dieses|diesen|ihn|sie|es)\b',
    re.IGNORECASE,
)
_TEXT_ARTIFACT_SOURCE_EDIT_COMPETING_TARGET_RE = re.compile(
    r'\b(?:'
    r'audio[\w-]*|recording|voice|sound|speech|clip|track|'
    r'image[\w-]*|picture|photo|video|visual|'
    r'aufnahme|stimme|tonspur|sprachfassung|'
    r'bild[\w-]*|foto[\w-]*|video[\w-]*|'
    r'graph|subgraph|branch|route|model|tool|workflow|pipeline|node|'
    r'graph[\w-]*|teilgraph|zweig|route|modell|werkzeug|arbeitsablauf|knoten'
    r')\b',
    re.IGNORECASE,
)
_TEXT_ARTIFACT_SOURCE_EDIT_JSON_TARGET_RE = re.compile(
    r'\bjson\s*(?:[-–—]\s*)?(?:object|objekt|response|antwort|output|ausgabe)\b',
    re.IGNORECASE,
)
_TEXT_ARTIFACT_SOURCE_EDIT_QUALIFIED_JSON_RESPONSE_TARGET_RE = re.compile(
    r'\b(?:new|final|resulting|response|output|neu(?:e|en|em|er|es)?|'
    r'final(?:e|en|em|er|es)?|abschliessend(?:e|en|em|er|es)?|'
    r'abschließend(?:e|en|em|er|es)?)\s+'
    r'json\s*(?:[-–—]\s*)?(?:object|objekt|response|antwort|output|ausgabe)?\b|'
    r'\bjson\s*(?:[-–—]\s*)?(?:response|antwort|output|ausgabe)\b',
    re.IGNORECASE,
)
_TEXT_ARTIFACT_SOURCE_EDIT_COMMON_CONTENT_RE = re.compile(
    r'\b(?:'
    r'file|document|artifact|artefact|source|text|content|copy|wording|transcript|'
    r'typo|spelling|grammar|sentence|paragraph|heading|headline|title|section|note|list|table|'
    r'checklist|readme|line|link|url|citation|footer|button|introduction|attribute|caption|label|'
    r'datei|dokument|artefakt|quelle|text|inhalt|wortlaut|transkript|'
    r'tippfehler|rechtschreibung|grammatik|satz|absatz|uberschrift|überschrift|titel|abschnitt|'
    r'notiz|liste|tabelle|checkliste|zeile|link|url|zitat|fusszeile|fußzeile|knopf|'
    r'einleitung|attribut|bildunterschrift|beschriftung'
    r')\b',
    re.IGNORECASE,
)
_TEXT_ARTIFACT_SOURCE_EDIT_STRONG_CONTENT_RE = re.compile(
    r'\b(?:'
    r'copy|wording|typo|spelling|grammar|sentence|paragraph|heading|headline|title|section|note|'
    r'list|table|checklist|readme|line|link|url|citation|footer|button|introduction|attribute|'
    r'caption|label|wortlaut|tippfehler|rechtschreibung|grammatik|satz|absatz|uberschrift|'
    r'überschrift|titel|abschnitt|notiz|liste|tabelle|checkliste|zeile|zitat|fusszeile|fußzeile|'
    r'knopf|einleitung|attribut|bildunterschrift|beschriftung'
    r')\b',
    re.IGNORECASE,
)
_TEXT_ARTIFACT_SOURCE_EDIT_COMPOUND_CONTENT_RE = re.compile(
    r'\b(?:audio[\w-]*|image[\w-]*|video[\w-]*|model|graph|route|tool|'
    r'bild[\w-]*|modell|route|werkzeug)\s+'
    r'(?:url|link|field|key|value|property|attribute|reference|ref|caption|label|name|alt\s+text|'
    r'feld|schlussel|schlüssel|wert|eigenschaft|attribut|referenz|beschriftung|name)\b',
    re.IGNORECASE,
)
_TEXT_ARTIFACT_SOURCE_EDIT_WEB_CONTENT_RE = re.compile(
    r'\b(?:font|color|colour|style|layout|theme|selector|class|stylesheet|'
    r'schrift|farbe|stil|farbschema|selektor|klasse)\b',
    re.IGNORECASE,
)
_TEXT_ARTIFACT_SOURCE_EDIT_DATA_CONTENT_RE = re.compile(
    r'\b(?:field|key|value|status|property|entry|row|column|schema|'
    r'feld|schlussel|schlüssel|wert|eigenschaft|eintrag|zeile|spalte|schema)\b',
    re.IGNORECASE,
)
_TEXT_ARTIFACT_SOURCE_EDIT_CODE_CONTENT_RE = re.compile(
    r'\b(?:code|function|class|method|module|import|variable|constant|script|query|bug|'
    r'funktion|klasse|methode|modul|variable|konstante|skript|abfrage|fehler)\b',
    re.IGNORECASE,
)
_TEXT_ARTIFACT_FORMAT_RE = re.compile(
    r'\b(?P<format>'
    r'html|htm|css|stylesheet|style\s*sheet|javascript|java\s*script|js|markdown|md|plain\s+text|txt|'
    r'readme|json|yaml|yml|xml|csv|svg|typescript|ts|tsx|jsx|python|py|shell|bash|sh|sql'
    r')\b',
    re.IGNORECASE,
)
_TEXT_ARTIFACT_DISTINCT_FORMAT_CUE_RE = re.compile(
    r'\b(?:'
    r'dedicated|separate|stand[\s-]?alone|external|own|individual|'
    r'dediziert(?:e|en|em|er|es)?|separat(?:e|en|em|er|es)?|'
    r'getrennt(?:e|en|em|er|es)?|eigen(?:e|en|em|er|es)?|'
    r'eigenst(?:a|ä|ae)ndig(?:e|en|em|er|es)?|extern(?:e|en|em|er|es)?|'
    r'individuell(?:e|en|em|er|es)?'
    r')\b',
    re.IGNORECASE,
)
_TEXT_ARTIFACT_RESPONSE_FORMAT_ONLY_EXTENSIONS = {'json'}
_JSON_TEXT_ARTIFACT_QUOTED_SPAN_RE = re.compile(
    r'"[^"\n]*"|“[^”\n]*”|`[^`\n]*`|(?<!\w)\'[^\'\n]*\'(?!\w)'
)
# Closed JSON grammar: quoted filenames require ``create/save ... file``;
# field/reference/path/url compounds remain response shapes, not file requests.
_JSON_TEXT_ARTIFACT_QUOTED_FILENAME_RE = re.compile(
    r'(?P<quote>["`])(?P<filename>[A-Za-z0-9][A-Za-z0-9._-]{0,80}\.json)(?P=quote)',
    re.IGNORECASE,
)
_JSON_TEXT_ARTIFACT_QUOTED_FILENAME_ACTION_RE = re.compile(
    r'\b(?:create|save)\s+(?:(?:the|a)\s+)?file\s*$',
    re.IGNORECASE,
)
_JSON_TEXT_ARTIFACT_DIRECT_FILE_RE = re.compile(
    r'\bjson\b\s*(?:[-–—]\s*)?'
    r'(?:file|artifact|artefact|artefakt|datei|document|dokument)\b'
    r'(?![\s_-]+(?:path|paths|reference|references|ref|refs|field|fields|url|urls|'
    r'pfad|pfade|referenz|referenzen|feld|felder)\b)|'
    r'\bjson\b[^,;.!?\n]{0,24}\b(?:as\s+(?:a\s+)?file|als\s+datei)\b',
    re.IGNORECASE,
)
_JSON_TEXT_ARTIFACT_FILE_FORMAT_RE = re.compile(
    r'\bfile\s+in\s+json\s+format\b|'
    r'\bfile\s+formatted\s+as\s+json\b|'
    r'\bjson[-_]formatted\s+file\b',
    re.IGNORECASE,
)
_JSON_TEXT_ARTIFACT_EXTRA_ACTION_RE = re.compile(
    r'\b(?:output|export|download|persist|persistiere)\b',
    re.IGNORECASE,
)
_JSON_TEXT_ARTIFACT_FORMAT_ACTION_RE = re.compile(
    r'\b(?:save|materialize|materialise|download|export|persist|'
    r'speichere|materialisiere|persistiere)\b'
    r'(?:\s+(?:a\s+|an\s+|ein(?:e|en|em|er|es)?\s+)?json\b|'
    r'[^.;!?\n]{0,120}\b(?:as|als)\s+'
    r'(?:a\s+|an\s+|ein(?:e|en|em|er|es)?\s+)?json\b)',
    re.IGNORECASE,
)
_JSON_TEXT_ARTIFACT_NEGATION_PREFIX_RE = re.compile(
    r'\b(?:do\s+not|don[\'’]?t|dont|must\s+not|should\s+not|never|no|not|without|'
    r'kein(?:e|en|em|er|es)?|nicht|ohne)\b[^,;.!?\n]{0,96}$',
    re.IGNORECASE,
)
_JSON_TEXT_ARTIFACT_NEGATION_SUFFIX_RE = re.compile(
    r'^\s*(?:(?:must|should|is|be)\s+)?(?:not|never|nicht)\b',
    re.IGNORECASE,
)
_JSON_TEXT_ARTIFACT_NEGATED_CONTEXT_RE = re.compile(
    r'\b(?:do\s+not|don[\'’]?t|dont|must\s+not|should\s+not|never|no|not|without|'
    r'kein(?:e|en|em|er|es)?|nicht|ohne)\b[^;!?\n]{0,96}\bjson\b|'
    r'\bjson\b[^\n]{0,120}\b(?:no|not|never|without|kein(?:e|en|em|er|es)?|nicht|ohne)\b'
    r'[^\n]{0,64}\b(?:file|artifact|artefact|artefakt|datei)\b',
    re.IGNORECASE,
)
_JSON_TEXT_ARTIFACT_RESPONSE_FORMAT_RE = re.compile(
    r'\b(?:return|provide|emit|output|gib|liefere)\b[^.;!?\n]{0,120}\bjson\b|'
    r'\bjson\b\s*(?:[-–—]\s*)?(?:object|objekt|response|antwort|output|ausgabe)\b',
    re.IGNORECASE,
)
_TEXT_ARTIFACT_NEGATION_RE = re.compile(
    r'\b(?:do\s+not|don[\'’]?t|without|kein|keine|keinen|nicht|ohne)\b[^.;!?\n]{0,80}\b'
    r'(?:file|artifact|artefact|artefakt|datei|download|save|speicher|persist)',
    re.IGNORECASE,
)
_TEXT_ARTIFACT_CARDINALITY_CONSTRAINT_RE = re.compile(
    r'\b(?:do\s+not|don[\'’]?t|dont|must\s+not|should\s+not|never|no|nicht|kein(?:e|en|er|es)?)\b'
    r'(?=[^.;!?\n]{0,220}\b(?:extra|additional|duplicate|duplicated|more|further|'
    r'zusaetzlich(?:e|en|er|es)?|zusatzlich(?:e|en|er|es)?|zusätzlich(?:e|en|er|es)?|'
    r'weitere?|doppelte?)\b)'
    r'(?=[^.;!?\n]{0,240}\b(?:beyond|except|other\s+than|apart\s+from|besides|duplicate|'
    r'duplicates|instead\s+of\s+creating\s+duplicate|ueber|uber|über|ausser|außer)\b)'
    r'(?=[^.;!?\n]{0,240}\b(?:artifact(?:s)?|artefact(?:s)?|artefakt(?:e|en)?|file(?:s)?|'
    r'datei(?:en)?|asset(?:s)?|html|css|image(?:s)?|picture(?:s)?|photo(?:s)?|bild(?:er)?|'
    r'page(?:\s+file)?(?:s)?)\b)',
    re.IGNORECASE,
)
_TEXT_ARTIFACT_GENERIC_FALLBACK_CUE_RE = re.compile(
    r'\b('
    r'file|files|datei|dateien|document|documents|dokument|dokumente|'
    r'download|downloadable|save|saved|speicher|speichere|persist|persistiere|'
    r'write\s+to|as\s+a\s+file|als\s+datei|'
    r'(?:text|code|markdown|html|css|json)\s+(?:artifact|artefact|artefakt)'
    r')\b',
    re.IGNORECASE,
)
_TEXT_ARTIFACT_COUNT_WORDS: dict[str, int] = {
    'two': 2,
    'zwei': 2,
    'three': 3,
    'drei': 3,
    'four': 4,
    'vier': 4,
    'five': 5,
    'funf': 5,
    'fuenf': 5,
    'six': 6,
    'sechs': 6,
}
_TEXT_ARTIFACT_EACH_PART_CUE_RE = re.compile(
    r'\b(?:each|every|per|je|jedes|jeden|jede|jeder|jedem)\b'
    r'[\s\S]{0,96}\b(?:part|section|chapter|teil|abschnitt|kapitel)\b'
    r'[\s\S]{0,96}\b(?:own|separate|individual|eigen(?:e|es|en|er)?|separat(?:e|es|en|er)?)\b|'
    r'\b(?:own|separate|individual|eigen(?:e|es|en|er)?|separat(?:e|es|en|er)?)\b'
    r'[\s\S]{0,96}\b(?:for\s+each|je|jedes|jeden|jede|jeder|jedem)\b'
    r'[\s\S]{0,96}\b(?:part|section|chapter|teil|abschnitt|kapitel)\b|'
    r'\b(?:part|section|chapter|teil|abschnitt|kapitel)\b'
    r'[\s\S]{0,96}\b(?:own|separate|individual|eigen(?:e|es|en|er)?|separat(?:e|es|en|er)?)\b',
    re.IGNORECASE,
)
_TEXT_ARTIFACT_PART_COUNT_RE = re.compile(
    r'\b(?P<count>\d+|two|three|four|five|six|zwei|drei|vier|funf|fuenf|sechs)\b'
    r'[\s-]*(?:part|parts|section|sections|chapter|chapters|teil|teile|teilig(?:e|es|er|en)?|'
    r'abschnitt|abschnitte|kapitel)\b|'
    r'\b(?P<count_prefix>\d+|two|three|four|five|six|zwei|drei|vier|funf|fuenf|sechs)'
    r'teilig(?:e|es|er|en)?\b',
    re.IGNORECASE,
)
_TEXT_ARTIFACT_FORMAT_NEGATION_RE = re.compile(
    r'\b(?:do\s+not|don[\'’]?t|dont|no|not|without|kein(?:e|en|er|es)?|nicht|ohne)\b'
    r'[^.;!?\n]{0,48}$',
    re.IGNORECASE,
)
_TEXT_ARTIFACT_VALIDATION_ONLY_RE = re.compile(
    r'\b(?:treat|consider|count|mark|fail|fails|failing|closure|criteria|criterion|'
    r'complete|completion|incomplete|required|required\s+artifact|validation|validate|'
    r'werte|betrachte|gilt|gelte|abschluss|vollstaendig|vollständig|unvollstaendig|unvollständig|'
    r'erforderlich|pflicht|kriterium|kriterien)\b',
    re.IGNORECASE,
)
_TEXT_ARTIFACT_MISSING_RE = re.compile(
    r'\b(?:missing|absent|not\s+present|not\s+created|not\s+generated|not\s+saved|'
    r'fehlt|fehlen|fehlend|nicht\s+vorhanden|nicht\s+erstellt|nicht\s+generiert|'
    r'nicht\s+gespeichert|unvollstaendig|unvollständig|incomplete)\b',
    re.IGNORECASE,
)
_TEXT_ARTIFACT_DEMONSTRATIVE_REFERENCE_RE = re.compile(
    r'\b(?:this|that|it|these|those|dies|diese|dieser|dieses|das|den|dem|ihn|sie|es|daraus|davon)\b'
    r'.{0,96}\b(?:file|artifact|artefact|artefakt|datei|document|dokument|html|css|js|markdown|text)\b|'
    r'\b(?:file|artifact|artefact|artefakt|datei|document|dokument|html|css|js|markdown|text)\b'
    r'.{0,96}\b(?:this|that|it|these|those|dies|diese|dieser|dieses|das|den|dem|ihn|sie|es|daraus|davon)\b',
    re.IGNORECASE,
)
_RELATIVE_ARTIFACT_CONTENT_THAT_RE = re.compile(
    r'\b(?:audio|speech|voice|recording|image|picture|photo|illustration|'
    r'file|artifact|artefact|document|text|aufnahme|bild|datei|artefakt|dokument)\b'
    r'(?:\s+[\w-]+){0,3}\s+(?P<relative>that)\b'
    r'(?=\s+(?:says?|reads?|contains?|shows?|depicts?|features?|states?|speaks?|narrates?)\b)',
    re.IGNORECASE,
)
_TEXT_ARTIFACT_SELF_CONTAINED_SOURCE_RE = re.compile(
    r'\b(?:create|generate|write|produce|build|make|'
    r'erzeuge|erstelle|generiere|schreibe|baue|verfasse)\b'
    r'.{0,120}\b(?:text|copy|note|notice|warning|protocol|report|caption|claim|slogan|script|'
    r'text|hinweis|warnhinweis|notiz|protokoll|bericht|caption|claim|slogan|skript|zeile)\b'
    r'.{0,180}\b(?:save|persist|speichere|persistiere|artifact|artefact|artefakt|file|datei)\b',
    re.IGNORECASE,
)
_TEXT_ARTIFACT_SAME_PROMPT_GENERATED_SOURCE_RE = re.compile(
    r'\b(?:create|generate|write|produce|build|make|'
    r'erzeuge|erstelle|generiere|schreibe|baue|verfasse)\b'
    r'(?:\s+(?:me|mir|uns))?[\s,]*(?:an?\b|one\b|the\b|'
    r'einen?\b|eine\b|ein\b|den\b|die\b|das\b|'
    r'kurzen?\b|kurzes?\b|short\b|brief\b|concise\b)'
    r'[\s\S]{0,220}\b(?:it|ihn|sie|es|das|den|diese|diesen|daraus|davon)\b'
    r'[\s\S]{0,140}\b(?:save|saved|speichere|persist|persistiere|'
    r'artifact|artefact|artefakt|file|datei)\b',
    re.IGNORECASE,
)
_TEXT_ARTIFACT_SELF_CONTAINED_CODE_BUNDLE_RE = re.compile(
    r'\b(?:create|generate|write|produce|build|make|'
    r'erzeuge|erstelle|generiere|schreibe|baue|verfasse)\b'
    r'[\s\S]{0,240}\bhtml\b[\s\S]{0,96}\bcss\b'
    r'[\s\S]{0,160}\b(?:artifact(?:s)?|artefact(?:s)?|artefakt(?:e|en)?|file(?:s)?|datei(?:en)?)\b|'
    r'\b(?:create|generate|write|produce|build|make|'
    r'erzeuge|erstelle|generiere|schreibe|baue|verfasse)\b'
    r'[\s\S]{0,180}\b(?:artifact(?:s)?|artefact(?:s)?|artefakt(?:e|en)?|file(?:s)?|datei(?:en)?)\b'
    r'[\s\S]{0,160}\bhtml\b[\s\S]{0,96}\bcss\b',
    re.IGNORECASE,
)
_TEXT_ARTIFACT_IMPLICIT_WEB_PAGE_RE = re.compile(
    r'\b(?:'
    r'landing\s*page|landingpage|product\s*page|sales\s*page|web\s?page|website|webseite|homepage|startseite|'
    r'(?:produkt|verkaufs|angebots|promo|marketing|shop|portfolio|profil|event|projekt|kampagnen|marken|'
    r'reise|tourismus|touren|ferien|urlaubs|hotel|destinations?)seite'
    r')\b',
    re.IGNORECASE,
)
_TEXT_ARTIFACT_BARE_GERMAN_PAGE_RE = re.compile(r'\bseite\b', re.IGNORECASE)
_TEXT_ARTIFACT_WEB_PAGE_NEED_CUE_RE = re.compile(
    r'\b(?:need|want|require|brauch(?:e|en|st|t)?|ben[oö]tig(?:e|en|st|t)?|'
    r'h[aä]tte(?:st|n)?\s+gern(?:e)?|w[uü]nsch(?:e|en|st|t)?)\b',
    re.IGNORECASE,
)
_TEXT_ARTIFACT_GENERATED_VISUAL_CONTEXT_RE = re.compile(
    r'\b(?:generate|create|render|make|show|generier(?:e|en)?|erzeug(?:e|en)?|erstelle|erstellen|mach(?:e|en)?|zeige)\w*\b'
    r'[^.;!?\n]{0,180}\b(?:image|images|picture|pictures|photo|photos|bild|bilder|foto|fotos)\b|'
    r'\b(?:image|images|picture|pictures|photo|photos|bild|bilder|foto|fotos)\b'
    r'[^.;!?\n]{0,180}\b(?:generate|create|render|make|show|generier(?:e|en)?|erzeug(?:e|en)?|erstelle|erstellen|mach(?:e|en)?|zeige)\w*\b|'
    r'\b(?:exactly|genau)?\s*(?:\d+|one|two|three|four|five|six|ein(?:e|s|en)?|zwei|drei|vier|fuenf|fünf|sechs)\s+'
    r'(?:image|images|picture|pictures|photo|photos|bild|bilder|foto|fotos)\b',
    re.IGNORECASE,
)
_TEXT_ARTIFACT_INLINE_SOURCE_RE = re.compile(
    r'(?is)('
    r'(?:content|code|source|html|css|javascript|json|markdown|text|following|below|inhalt|quelle|code)\s*:\s*\S.{20,}|'
    r':\s*(?:<!doctype\s+html\b|<html\b|<[a-z][\w:-]*(?:\s|>|/)|\{|\[|function\b|const\b|let\b|var\b|body\s*\{)'
    r')'
)
_TEXT_ARTIFACT_SELF_CLAIM_RE = re.compile(
    r'(?is)('
    r'\[artifact:\s*[^\]]+\]|'
    r'artifact created|'
    r'artifact generated|'
    r'saved locally|'
    r'saved as a local artifact|'
    r'downloadable artifact|'
    r'ready-to-run artifact|'
    r'ready to be downloaded'
    r')'
)
_TEXT_ARTIFACT_FENCED_BLOCK_RE = re.compile(
    r'```(?P<lang>[A-Za-z0-9_+.-]*)(?:[^\n`]*)?\n(?P<body>.*?)(?:\n```|```)',
    re.DOTALL,
)
_TEXT_ARTIFACT_TAG_RE = re.compile(
    r'(?is)<artifact\b(?P<attrs>[^>]*)>(?P<body>.*?)</artifact>'
)
_TEXT_ARTIFACT_TAG_ATTR_RE = re.compile(
    r'(?P<name>[A-Za-z_:][-A-Za-z0-9_:.]*)\s*=\s*(?P<quote>["\'])(?P<value>.*?)(?P=quote)',
    re.DOTALL,
)
_TEXT_ARTIFACT_BLOCKED_CONTENT_RE = re.compile(
    r'(?is)\b('
    r'i\s+(?:cannot|can[\'’]?t)\s+(?:proceed|create|generate|make|write)|'
    r'cannot\s+(?:proceed|create|generate|make|write)|'
    r'awaiting\s+(?:clarification|input)|'
    r'needs?\s+(?:clarification|content|input|specification)|'
    r'missing\s+(?:content|input|specification)|'
    r'not\s+included\s+in\s+your\s+request|'
    r'please\s+provide\s+(?:the\s+)?(?:html|code|content|description|specification)|'
    r'provide\s+(?:the\s+)?(?:html|code|content|description|specification)'
    r')\b',
)
_TEXT_ARTIFACT_NON_PAYLOAD_LANGS = {
    'analysis',
    'reasoning',
    'thought',
    'thinking',
    'plan',
}
_TEXT_ARTIFACT_LANGUAGE_EXTENSIONS = {
    'md': 'md',
    'markdown': 'md',
    'html': 'html',
    'htm': 'html',
    'css': 'css',
    'js': 'js',
    'javascript': 'js',
    'mjs': 'js',
    'cjs': 'js',
    'ts': 'ts',
    'typescript': 'ts',
    'tsx': 'tsx',
    'jsx': 'jsx',
    'json': 'json',
    'yaml': 'yaml',
    'yml': 'yaml',
    'xml': 'xml',
    'csv': 'csv',
    'svg': 'svg',
    'py': 'py',
    'python': 'py',
    'sh': 'sh',
    'bash': 'sh',
    'shell': 'sh',
    'sql': 'sql',
    'txt': 'txt',
    'text': 'txt',
}
_TEXT_ARTIFACT_CONTROL_JSON_KEYS = {
    'candidate_graph',
    'output_candidates',
    'output_obligations',
    'promotion_review',
    'request_ir',
    'workload_graph',
}

QWEN3_CUSTOMVOICE_SPEAKERS = [
    'serena',
    'vivian',
    'uncle_fu',
    'ryan',
    'aiden',
    'ono_anna',
    'sohee',
    'eric',
    'dylan',
]


def _normalize_text_artifact_extension(value: str) -> Optional[str]:
    token = re.sub(r'[^a-z0-9]+', '', str(value or '').strip().lower().lstrip('.'))
    aliases = {
        'plaintext': 'txt',
        'text': 'txt',
        'markdown': 'md',
        'readme': 'md',
        'stylesheet': 'css',
        'javascript': 'js',
        'java script': 'js',
        'typescript': 'ts',
        'python': 'py',
        'shell': 'sh',
        'bash': 'sh',
    }
    token = aliases.get(token, token)
    if token == 'markdown':
        token = 'md'
    if token in TEXT_ARTIFACT_EXTENSIONS:
        return 'md' if token == 'markdown' else token
    return None


def _text_artifact_source_name_is_local_target(scope: str, source_name: Optional[str]) -> bool:
    name = str(source_name or '').strip()
    if not name:
        return False
    tokens = [token for token in re.split(r'[\s._-]+', name) if token]
    if not tokens:
        return False
    pattern = r'[\s._-]+'.join(re.escape(token) for token in tokens)
    return bool(re.search(rf'(?<!\w){pattern}(?!\w)', scope, flags=re.IGNORECASE))


def _text_artifact_source_format_is_local_target(scope: str, source_extension: str) -> bool:
    normalized_source = _normalize_text_artifact_extension(source_extension or '')
    if not normalized_source:
        return False
    return any(
        _normalize_text_artifact_extension(match.group('format') or '') == normalized_source
        for match in _TEXT_ARTIFACT_FORMAT_RE.finditer(scope)
    )


def _text_artifact_source_content_is_local_target(scope: str, source_extension: str) -> bool:
    normalized_source = _normalize_text_artifact_extension(source_extension or '') or ''
    if _TEXT_ARTIFACT_SOURCE_EDIT_COMMON_CONTENT_RE.search(scope):
        return True
    if normalized_source in {'html', 'htm', 'css', 'svg'}:
        return bool(_TEXT_ARTIFACT_SOURCE_EDIT_WEB_CONTENT_RE.search(scope))
    if normalized_source in {'json', 'yaml', 'yml', 'xml', 'csv'}:
        return bool(_TEXT_ARTIFACT_SOURCE_EDIT_DATA_CONTENT_RE.search(scope))
    if normalized_source in {'js', 'mjs', 'cjs', 'ts', 'tsx', 'jsx', 'py', 'sh', 'sql'}:
        return bool(_TEXT_ARTIFACT_SOURCE_EDIT_CODE_CONTENT_RE.search(scope))
    return False


def _text_artifact_source_strong_content_is_local_target(scope: str, source_extension: str) -> bool:
    normalized_source = _normalize_text_artifact_extension(source_extension or '') or ''
    if _TEXT_ARTIFACT_SOURCE_EDIT_STRONG_CONTENT_RE.search(scope):
        return True
    if normalized_source in {'html', 'htm', 'css', 'svg'}:
        return bool(_TEXT_ARTIFACT_SOURCE_EDIT_WEB_CONTENT_RE.search(scope))
    if normalized_source in {'json', 'yaml', 'yml', 'xml', 'csv'}:
        return bool(_TEXT_ARTIFACT_SOURCE_EDIT_DATA_CONTENT_RE.search(scope))
    if normalized_source in {'js', 'mjs', 'cjs', 'ts', 'tsx', 'jsx', 'py', 'sh', 'sql'}:
        return bool(_TEXT_ARTIFACT_SOURCE_EDIT_CODE_CONTENT_RE.search(scope))
    return False


def _text_artifact_selected_source_edit_is_bound(
    text: str,
    *,
    source_extension: str,
    source_name: Optional[str],
    response_json_only: bool,
) -> bool:
    """Return true when an edit action locally targets the selected text source."""

    prompt = str(text or '').strip()
    if not prompt or not _TEXT_ARTIFACT_SOURCE_EDIT_CUE_RE.search(prompt):
        return False
    for clause_match in _TEXT_ARTIFACT_SOURCE_EDIT_CLAUSE_RE.finditer(prompt):
        clause = clause_match.group(0)
        actions = list(_TEXT_ARTIFACT_SOURCE_EDIT_ACTION_RE.finditer(clause))
        for action in actions:
            action_tail = clause[action.end():]
            next_directive = _TEXT_ARTIFACT_SOURCE_EDIT_NEXT_DIRECTIVE_RE.search(action_tail)
            if next_directive:
                action_tail = action_tail[:next_directive.start()]
            local_scope = action_tail[:160]
            if not local_scope.strip():
                continue
            normalized_source = _normalize_text_artifact_extension(source_extension or '') or ''
            json_response_target = None
            if response_json_only:
                json_response_target = (
                    _TEXT_ARTIFACT_SOURCE_EDIT_JSON_TARGET_RE.search(local_scope)
                    if normalized_source != 'json'
                    else _TEXT_ARTIFACT_SOURCE_EDIT_QUALIFIED_JSON_RESPONSE_TARGET_RE.search(local_scope)
                )
            if json_response_target:
                binding_scope = local_scope[:json_response_target.start()]
                if _TEXT_ARTIFACT_SOURCE_EDIT_EXPLICIT_SOURCE_RE.search(binding_scope):
                    return True
                if _text_artifact_source_name_is_local_target(binding_scope, source_name):
                    return True
                continue
            competing_target = _TEXT_ARTIFACT_SOURCE_EDIT_COMPETING_TARGET_RE.search(local_scope)
            prior_binding_scope = (
                local_scope[:competing_target.start()]
                if competing_target
                else local_scope
            )
            prior_scope = clause[:action.start()]
            prior_competing_target = _TEXT_ARTIFACT_SOURCE_EDIT_COMPETING_TARGET_RE.search(prior_scope)
            prior_json_response_target = bool(
                response_json_only
                and _JSON_TEXT_ARTIFACT_RESPONSE_FORMAT_RE.search(prior_scope)
            )
            if prior_json_response_target:
                if _TEXT_ARTIFACT_SOURCE_EDIT_EXPLICIT_SOURCE_RE.search(prior_binding_scope):
                    return True
                if _text_artifact_source_name_is_local_target(prior_binding_scope, source_name):
                    return True
                if _text_artifact_source_format_is_local_target(prior_binding_scope, source_extension):
                    return True
                continue
            compound_content_target = bool(
                _TEXT_ARTIFACT_SOURCE_EDIT_COMPOUND_CONTENT_RE.search(local_scope)
            )
            if prior_competing_target:
                if _TEXT_ARTIFACT_SOURCE_EDIT_EXPLICIT_SOURCE_RE.search(prior_binding_scope):
                    return True
                if _text_artifact_source_name_is_local_target(prior_binding_scope, source_name):
                    return True
                if _text_artifact_source_format_is_local_target(prior_binding_scope, source_extension):
                    return True
                if compound_content_target or _text_artifact_source_strong_content_is_local_target(
                    local_scope,
                    source_extension,
                ):
                    return True
                continue
            if not competing_target:
                # With one selected source and no other explicit target, the
                # action's implicit object remains that source (for example,
                # "Add a footer" or "Replace the introduction").
                return True
            if compound_content_target:
                return True
            binding_scope = (
                local_scope[:competing_target.start()]
            )
            if _TEXT_ARTIFACT_SOURCE_EDIT_EXPLICIT_SOURCE_RE.search(binding_scope):
                return True
            if _TEXT_ARTIFACT_SOURCE_EDIT_DIRECT_PRONOUN_RE.search(binding_scope):
                return True
            if _text_artifact_source_format_is_local_target(binding_scope, source_extension):
                return True
            if _text_artifact_source_name_is_local_target(binding_scope, source_name):
                return True
            if _text_artifact_source_content_is_local_target(binding_scope, source_extension):
                return True
    return False


def _text_artifact_count_from_token(value: str) -> int:
    token = re.sub(r'[^a-z0-9]+', '', str(value or '').strip().lower())
    if not token:
        return 0
    if token.isdigit():
        parsed = int(token)
        return parsed if parsed > 0 else 0
    return int(_TEXT_ARTIFACT_COUNT_WORDS.get(token, 0))


def _text_artifact_each_part_count(text: str) -> int:
    prompt = str(text or '').strip()
    if not prompt or not _TEXT_ARTIFACT_EACH_PART_CUE_RE.search(prompt):
        return 0
    counts = [
        _text_artifact_count_from_token(match.group('count') or match.group('count_prefix') or '')
        for match in _TEXT_ARTIFACT_PART_COUNT_RE.finditer(prompt)
    ]
    counts = [count for count in counts if count > 1]
    return max(counts) if counts else 0


def _normalize_text_artifact_block_language(value: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(value or '').strip().lower())


def _text_artifact_block_language_is_payload(language: str) -> bool:
    lang = _normalize_text_artifact_block_language(language)
    return lang not in _TEXT_ARTIFACT_NON_PAYLOAD_LANGS


def _text_artifact_block_extension(language: str) -> Optional[str]:
    lang = _normalize_text_artifact_block_language(language)
    if not lang or lang in _TEXT_ARTIFACT_NON_PAYLOAD_LANGS:
        return None
    return _TEXT_ARTIFACT_LANGUAGE_EXTENSIONS.get(lang)


def normalize_text_artifact_extension(value: str) -> Optional[str]:
    return _normalize_text_artifact_extension(value)


def prompt_has_inline_text_artifact_source(prompt: str) -> bool:
    text = str(prompt or '').strip()
    return bool(
        _TEXT_ARTIFACT_FENCED_BLOCK_RE.search(text)
        or _TEXT_ARTIFACT_INLINE_SOURCE_RE.search(text)
    )


def _text_artifact_format_match_is_negated(text: str, match: re.Match[str]) -> bool:
    prefix = str(text or '')[max(0, match.start() - 64):match.start()]
    return bool(_TEXT_ARTIFACT_FORMAT_NEGATION_RE.search(prefix))


def _text_artifact_match_scope(text: str, start: int, end: int) -> tuple[str, int, int]:
    prompt = str(text or '')
    start_index = max(0, int(start or 0))
    end_index = min(len(prompt), max(start_index, int(end or start_index)))
    lower = max(
        prompt.rfind('.', 0, start_index),
        prompt.rfind('!', 0, start_index),
        prompt.rfind('?', 0, start_index),
        prompt.rfind('\n', 0, start_index),
    )
    end_candidates = [
        index
        for index in (
            prompt.find('.', end_index),
            prompt.find('!', end_index),
            prompt.find('?', end_index),
            prompt.find('\n', end_index),
        )
        if index >= 0
    ]
    upper = min(end_candidates) if end_candidates else len(prompt)
    return prompt[lower + 1:upper].strip(), lower + 1, upper


@dataclass(frozen=True)
class _JsonTextArtifactIntent:
    """Bounded JSON-only materialization decision for one prompt."""

    format_spans: frozenset[tuple[int, int]] = frozenset()
    extension_spans: frozenset[tuple[int, int]] = frozenset()
    suppress_generic_fallback: bool = False

    @property
    def has_materialization(self) -> bool:
        return bool(self.format_spans or self.extension_spans)


def _mask_json_text_artifact_quotes(text: str) -> tuple[str, bool]:
    prompt = str(text or '')
    masked = list(prompt)
    quoted_json = False
    for match in _JSON_TEXT_ARTIFACT_QUOTED_SPAN_RE.finditer(prompt):
        quoted_json = quoted_json or bool(re.search(r'\bjson\b', match.group(0), re.IGNORECASE))
        for index in range(match.start(), match.end()):
            masked[index] = ' '
    return ''.join(masked), quoted_json


def _json_text_artifact_candidate_state(text: str, start: int, end: int) -> tuple[bool, bool]:
    scope, scope_start, _ = _text_artifact_match_scope(text, start, end)
    if not scope:
        return False, False
    relative_start = max(0, start - scope_start)
    relative_end = min(len(scope), max(relative_start, end - scope_start))
    negated = bool(
        _JSON_TEXT_ARTIFACT_NEGATION_PREFIX_RE.search(scope[:relative_start])
        or _JSON_TEXT_ARTIFACT_NEGATION_SUFFIX_RE.search(scope[relative_end:])
    )
    has_action = bool(
        _TEXT_ARTIFACT_ACTION_CUE_RE.search(scope)
        or _JSON_TEXT_ARTIFACT_EXTRA_ACTION_RE.search(scope)
    )
    return negated, has_action


def _json_text_artifact_intent(text: str) -> _JsonTextArtifactIntent:
    """Separate JSON response formatting from explicit JSON file materialization."""

    prompt = str(text or '')
    masked, quoted_json = _mask_json_text_artifact_quotes(prompt)
    format_matches = [
        match
        for match in _TEXT_ARTIFACT_FORMAT_RE.finditer(masked)
        if _normalize_text_artifact_extension(match.group('format') or '') == 'json'
    ]
    extension_matches = [
        match
        for match in _TEXT_ARTIFACT_EXTENSION_RE.finditer(masked)
        if _normalize_text_artifact_extension(match.group('ext') or '') == 'json'
    ]
    format_spans: set[tuple[int, int]] = set()
    extension_spans: set[tuple[int, int]] = set()
    negated_materialization = False

    def add_format_spans(candidate_start: int, candidate_end: int) -> None:
        for format_match in format_matches:
            if candidate_start <= format_match.start() and format_match.end() <= candidate_end:
                format_spans.add((format_match.start(), format_match.end()))

    for pattern, action_is_in_candidate in (
        (_JSON_TEXT_ARTIFACT_DIRECT_FILE_RE, False),
        (_JSON_TEXT_ARTIFACT_FILE_FORMAT_RE, False),
        (_JSON_TEXT_ARTIFACT_FORMAT_ACTION_RE, True),
    ):
        for candidate in pattern.finditer(masked):
            negated, has_action = _json_text_artifact_candidate_state(
                masked,
                candidate.start(),
                candidate.end(),
            )
            if negated:
                negated_materialization = True
            elif action_is_in_candidate or has_action:
                add_format_spans(candidate.start(), candidate.end())

    for extension_match in extension_matches:
        negated, has_action = _json_text_artifact_candidate_state(
            masked,
            extension_match.start(),
            extension_match.end(),
        )
        if negated:
            negated_materialization = True
        elif has_action:
            extension_spans.add((extension_match.start(), extension_match.end()))

    quoted_spans = list(_JSON_TEXT_ARTIFACT_QUOTED_SPAN_RE.finditer(prompt))
    for quoted_filename in _JSON_TEXT_ARTIFACT_QUOTED_FILENAME_RE.finditer(prompt):
        if any(
            quote_span.start() < quoted_filename.start()
            and quoted_filename.end() < quote_span.end()
            for quote_span in quoted_spans
        ):
            continue
        _, scope_start, _ = _text_artifact_match_scope(
            prompt,
            quoted_filename.start(),
            quoted_filename.end(),
        )
        if not _JSON_TEXT_ARTIFACT_QUOTED_FILENAME_ACTION_RE.search(
            prompt[scope_start:quoted_filename.start()]
        ):
            continue
        negated, _ = _json_text_artifact_candidate_state(
            prompt,
            quoted_filename.start(),
            quoted_filename.end(),
        )
        if negated:
            negated_materialization = True
            continue
        filename_match = _TEXT_ARTIFACT_EXTENSION_RE.fullmatch(
            quoted_filename.group('filename') or ''
        )
        if not filename_match:
            continue
        filename_start = quoted_filename.start('filename')
        extension_spans.add(
            (
                filename_start + filename_match.start(),
                filename_start + filename_match.end(),
            )
        )

    has_positive = bool(format_spans or extension_spans)
    first_json_start = min((match.start() for match in format_matches), default=-1)
    file_cue_before_json = bool(
        first_json_start > 0
        and _TEXT_ARTIFACT_FILE_CUE_RE.search(masked[:first_json_start])
    )
    response_format = bool(_JSON_TEXT_ARTIFACT_RESPONSE_FORMAT_RE.search(masked))
    negated_context = bool(_JSON_TEXT_ARTIFACT_NEGATED_CONTEXT_RE.search(masked))
    suppress_generic_fallback = bool(
        not has_positive
        and (
            negated_materialization
            or negated_context
            or quoted_json
            or (response_format and not file_cue_before_json)
        )
    )
    return _JsonTextArtifactIntent(
        format_spans=frozenset(format_spans),
        extension_spans=frozenset(extension_spans),
        suppress_generic_fallback=suppress_generic_fallback,
    )


def _text_artifact_format_match_has_distinct_artifact_cue(
    text: str,
    match: re.Match[str],
    *,
    json_intent: Optional[_JsonTextArtifactIntent] = None,
) -> bool:
    scope, _, _ = _text_artifact_match_scope(text, match.start(), match.end())
    raw_format = str(match.group('format') or '').strip().lower()
    extension = _normalize_text_artifact_extension(raw_format) or 'txt'
    if (
        extension in _TEXT_ARTIFACT_RESPONSE_FORMAT_ONLY_EXTENSIONS
        and (match.start(), match.end())
        not in (json_intent or _json_text_artifact_intent(text)).format_spans
    ):
        return False
    return bool(
        scope
        and _TEXT_ARTIFACT_ACTION_CUE_RE.search(scope)
        and _TEXT_ARTIFACT_DISTINCT_FORMAT_CUE_RE.search(scope)
    )


def _text_artifact_negation_match_is_cardinality_constraint(
    text: str,
    match: re.Match[str],
) -> bool:
    scope, _, _ = _text_artifact_match_scope(text, match.start(), match.end())
    if not scope:
        return False
    return bool(_TEXT_ARTIFACT_CARDINALITY_CONSTRAINT_RE.search(scope))


def _text_artifact_prompt_has_positive_request_outside_negation(
    text: str,
    matches: list[re.Match[str]],
) -> bool:
    masked = list(str(text or ''))
    for match in matches:
        if _text_artifact_negation_match_is_cardinality_constraint(text, match):
            _, start, end = _text_artifact_match_scope(text, match.start(), match.end())
        else:
            start, end = match.start(), match.end()
        for index in range(max(0, start), min(len(masked), end)):
            masked[index] = ' '
    remainder = ''.join(masked)
    return bool(
        _TEXT_ARTIFACT_FILE_CUE_RE.search(remainder)
        and _TEXT_ARTIFACT_ACTION_CUE_RE.search(remainder)
    )


def _text_artifact_prompt_is_negated(text: str) -> bool:
    matches = list(_TEXT_ARTIFACT_NEGATION_RE.finditer(str(text or '')))
    if not matches:
        return False
    return not _text_artifact_prompt_has_positive_request_outside_negation(text, matches)


def _text_artifact_filename_match_is_validation_only(text: str, match: re.Match[str]) -> bool:
    prompt = str(text or '')
    clause, _, _ = _text_artifact_match_scope(prompt, match.start(), match.end())
    if not clause:
        return False
    if not _TEXT_ARTIFACT_MISSING_RE.search(clause):
        return False
    if _TEXT_ARTIFACT_ACTION_CUE_RE.search(clause):
        return False
    return bool(_TEXT_ARTIFACT_VALIDATION_ONLY_RE.search(clause))


def text_artifact_request_is_ungrounded_reference(prompt: str, *, source_available: bool = False) -> bool:
    text = str(prompt or '').strip()
    if not text or source_available or prompt_has_inline_text_artifact_source(text):
        return False
    if _json_text_artifact_intent(text).has_materialization:
        return False
    if _TEXT_ARTIFACT_SELF_CONTAINED_SOURCE_RE.search(text):
        return False
    if _TEXT_ARTIFACT_SAME_PROMPT_GENERATED_SOURCE_RE.search(text):
        return False
    if _TEXT_ARTIFACT_SELF_CONTAINED_CODE_BUNDLE_RE.search(text):
        return False
    extension_match = _TEXT_ARTIFACT_EXTENSION_RE.search(text)
    if extension_match and (
        _TEXT_ARTIFACT_FILE_CUE_RE.search(text)
        or _TEXT_ARTIFACT_ACTION_CUE_RE.search(text)
    ):
        return False
    reference_text = list(text)
    for match in _RELATIVE_ARTIFACT_CONTENT_THAT_RE.finditer(text):
        for index in range(match.start('relative'), match.end('relative')):
            reference_text[index] = ' '
    return bool(
        _TEXT_ARTIFACT_DEMONSTRATIVE_REFERENCE_RE.search(
            ''.join(reference_text)
        )
    )


def detect_text_artifact_request(
    prompt: str,
    *,
    source_available: bool = False,
    source_extension: Optional[str] = None,
    source_name: Optional[str] = None,
    source_path: Optional[str] = None,
) -> Optional[dict[str, str]]:
    """Detect explicit requests to materialize chat text as a file artifact."""

    requests = detect_text_artifact_requests(
        prompt,
        source_available=source_available,
        source_extension=source_extension,
        source_name=source_name,
        source_path=source_path,
    )
    return requests[0] if requests else None


def detect_text_artifact_requests(
    prompt: str,
    *,
    source_available: bool = False,
    source_extension: Optional[str] = None,
    source_name: Optional[str] = None,
    source_path: Optional[str] = None,
) -> list[dict[str, str]]:
    """Detect explicit requests to materialize one or more chat text file artifacts."""

    text = str(prompt or '').strip()
    if not text or _text_artifact_prompt_is_negated(text):
        return []
    json_intent = _json_text_artifact_intent(text)
    if text_artifact_request_is_ungrounded_reference(text, source_available=source_available):
        return []

    requests: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    def append_request(request: dict[str, str]) -> None:
        extension = _normalize_text_artifact_extension(request.get('extension') or '')
        if not extension:
            return
        source_name = str(request.get('source_name') or '').strip() or f'generated-{extension}'
        source = str(request.get('source') or '').strip() or 'explicit_file_cue'
        key = (extension, source_name, source)
        if key in seen or any(
            item.get('extension') == extension
            and item.get('source_name') == source_name
            and item.get('source') == source
            for item in requests
        ):
            return
        if source == 'explicit_extension':
            for existing in requests:
                if existing.get('extension') == extension and existing.get('source') != 'explicit_extension':
                    existing['source'] = source
                    existing['source_name'] = source_name
                    seen.add(key)
                    return
        elif (
            source != 'each_part_file_cue'
            and any(item.get('extension') == extension for item in requests)
        ):
            return
        seen.add(key)
        normalized_request = {
            'extension': extension,
            'source': source,
            'source_name': source_name,
        }
        target_path = str(request.get('target_path') or '').strip()
        if target_path:
            normalized_request['target_path'] = target_path
        requests.append(normalized_request)

    def has_request_extension(extension: str) -> bool:
        normalized_extension = _normalize_text_artifact_extension(extension or '')
        return bool(
            normalized_extension
            and any(item.get('extension') == normalized_extension for item in requests)
        )

    has_file_cue = bool(_TEXT_ARTIFACT_FILE_CUE_RE.search(text))
    has_action_cue = bool(_TEXT_ARTIFACT_ACTION_CUE_RE.search(text))
    has_web_page_need_cue = bool(_TEXT_ARTIFACT_WEB_PAGE_NEED_CUE_RE.search(text))
    has_implicit_web_page_cue = bool(_TEXT_ARTIFACT_IMPLICIT_WEB_PAGE_RE.search(text))
    has_bare_german_page_with_generated_visuals = bool(
        _TEXT_ARTIFACT_BARE_GERMAN_PAGE_RE.search(text)
        and _TEXT_ARTIFACT_GENERATED_VISUAL_CONTEXT_RE.search(text)
    )
    format_matches = list(_TEXT_ARTIFACT_FORMAT_RE.finditer(text))
    distinct_format_match_spans = {
        (format_match.start(), format_match.end())
        for format_match in format_matches
        if _text_artifact_format_match_has_distinct_artifact_cue(
            text,
            format_match,
            json_intent=json_intent,
        )
    }
    has_explicit_format_file_cue = has_file_cue and has_action_cue
    explicit_format_file_match_spans = {
        (format_match.start(), format_match.end())
        for format_match in format_matches
        if (
            (
                (_normalize_text_artifact_extension(format_match.group('format') or '') or 'txt')
                in _TEXT_ARTIFACT_RESPONSE_FORMAT_ONLY_EXTENSIONS
                and (format_match.start(), format_match.end()) in json_intent.format_spans
            )
            or (
                has_explicit_format_file_cue
                and (
                    (_normalize_text_artifact_extension(format_match.group('format') or '') or 'txt')
                    not in _TEXT_ARTIFACT_RESPONSE_FORMAT_ONLY_EXTENSIONS
                )
            )
        )
    }

    if explicit_format_file_match_spans or distinct_format_match_spans:
        each_part_count = _text_artifact_each_part_count(text)
        if explicit_format_file_match_spans and each_part_count > 1:
            for format_match in format_matches:
                if _text_artifact_format_match_is_negated(text, format_match):
                    continue
                if (
                    (format_match.start(), format_match.end())
                    not in explicit_format_file_match_spans
                ):
                    continue
                raw_format = str(format_match.group('format') or '').strip().lower()
                extension = _normalize_text_artifact_extension(raw_format) or 'txt'
                for index in range(1, each_part_count + 1):
                    append_request(
                        {
                            'extension': extension,
                            'source': 'each_part_file_cue',
                            'source_name': f'generated-{extension}-part-{index}',
                        }
                    )
                if requests:
                    return requests
        for format_match in format_matches:
            if _text_artifact_format_match_is_negated(text, format_match):
                continue
            match_span = (format_match.start(), format_match.end())
            has_explicit_match_cue = match_span in explicit_format_file_match_spans
            if not has_explicit_match_cue and match_span not in distinct_format_match_spans:
                continue
            raw_format = str(format_match.group('format') or '').strip().lower()
            extension = _normalize_text_artifact_extension(raw_format) or 'txt'
            if has_request_extension(extension):
                continue
            append_request(
                {
                    'extension': extension,
                    'source': (
                        'explicit_format_file_cue'
                        if has_explicit_match_cue
                        else 'distinct_format_artifact_cue'
                    ),
                    'source_name': 'README' if raw_format == 'readme' else f'generated-{extension}',
                }
            )

    if has_file_cue or has_action_cue:
        for extension_match in _TEXT_ARTIFACT_EXTENSION_RE.finditer(text):
            if _text_artifact_filename_match_is_validation_only(text, extension_match):
                continue
            extension = _normalize_text_artifact_extension(extension_match.group('ext') or '')
            if (
                extension == 'json'
                and (extension_match.start(), extension_match.end())
                not in json_intent.extension_spans
            ):
                continue
            if extension:
                append_request(
                    {
                        'extension': extension,
                        'source': 'explicit_extension',
                        'source_name': extension_match.group('name') or f'generated-{extension}',
                    }
                )

    if (
        (has_action_cue or has_web_page_need_cue)
        and not has_request_extension('html')
        and (has_implicit_web_page_cue or has_bare_german_page_with_generated_visuals)
    ):
        append_request(
            {
                'extension': 'html',
                'source': 'implicit_web_page_cue',
                'source_name': 'generated-html',
            }
        )

    source_artifact_extension = _normalize_text_artifact_extension(source_extension or '')
    if (
        source_available
        and source_artifact_extension
        and _text_artifact_selected_source_edit_is_bound(
            text,
            source_extension=source_artifact_extension,
            source_name=source_name,
            response_json_only=json_intent.suppress_generic_fallback,
        )
    ):
        append_request(
            {
                'extension': source_artifact_extension,
                'source': 'selected_source_edit',
                'source_name': str(source_name or '').strip() or f'updated-{source_artifact_extension}',
                **(
                    {'target_path': str(source_path or '').strip()}
                    if str(source_path or '').strip()
                    else {}
                ),
            }
        )
        return requests

    if not (has_file_cue and has_action_cue):
        return requests
    if requests:
        return requests
    if json_intent.suppress_generic_fallback:
        return requests
    if not _TEXT_ARTIFACT_GENERIC_FALLBACK_CUE_RE.search(text):
        return requests

    append_request({'extension': 'txt', 'source': 'explicit_file_cue', 'source_name': 'generated-text'})
    return requests


def generated_text_is_artifact_self_claim(content: str) -> bool:
    """Return true when text only claims an artifact instead of being the artifact content."""

    return bool(_TEXT_ARTIFACT_SELF_CLAIM_RE.search(str(content or '').strip()))


def generated_text_blocks_artifact_persistence(content: str) -> bool:
    return bool(_TEXT_ARTIFACT_BLOCKED_CONTENT_RE.search(str(content or '').strip()))


def _code_block_language_matches_extension(language: str, extension: str) -> bool:
    if not _text_artifact_block_language_is_payload(language):
        return False
    lang = _normalize_text_artifact_block_language(language)
    ext = _normalize_text_artifact_extension(extension or '') or 'txt'
    aliases = {
        'md': {'md', 'markdown'},
        'html': {'html', 'htm'},
        'css': {'css'},
        'js': {'js', 'javascript', 'mjs', 'cjs'},
        'ts': {'ts', 'typescript'},
        'tsx': {'tsx'},
        'jsx': {'jsx'},
        'json': {'json'},
        'yaml': {'yaml', 'yml'},
        'yml': {'yaml', 'yml'},
        'xml': {'xml'},
        'csv': {'csv'},
        'svg': {'svg', 'xml'},
        'py': {'py', 'python'},
        'sh': {'sh', 'bash', 'shell'},
        'sql': {'sql'},
        'txt': {'txt', 'text', ''},
    }
    return lang in aliases.get(ext, {ext, ''})


def _parse_text_artifact_tag_attrs(raw_attrs: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for match in _TEXT_ARTIFACT_TAG_ATTR_RE.finditer(str(raw_attrs or '')):
        name = str(match.group('name') or '').strip().lower()
        value = str(match.group('value') or '').strip()
        if name and value:
            attrs[name] = value
    return attrs


def _text_artifact_tag_extension(attrs: dict[str, str]) -> Optional[str]:
    identifier = str(
        attrs.get('identifier')
        or attrs.get('filename')
        or attrs.get('file')
        or attrs.get('name')
        or ''
    ).strip()
    if identifier:
        extension = _normalize_text_artifact_extension(Path(identifier).suffix)
        if extension:
            return extension
    raw_type = str(attrs.get('type') or attrs.get('mime') or attrs.get('mime_type') or '').strip().lower()
    if raw_type:
        if '/' in raw_type:
            extension = _normalize_text_artifact_extension(raw_type.rsplit('/', 1)[-1])
            if extension:
                return extension
        extension = _normalize_text_artifact_extension(raw_type)
        if extension:
            return extension
    return None


def _text_artifact_tag_source_name(attrs: dict[str, str], extension: str) -> str:
    identifier = str(
        attrs.get('identifier')
        or attrs.get('filename')
        or attrs.get('file')
        or attrs.get('name')
        or attrs.get('title')
        or ''
    ).strip()
    if identifier:
        stem = Path(identifier).stem
        if stem:
            return stem
    return f'generated-{extension or "text"}'


def _artifact_request_with_tag_metadata(
    artifact_request: dict[str, str],
    *,
    tag_extension: Optional[str],
    tag_source_name: Optional[str],
) -> dict[str, str]:
    request = dict(artifact_request or {})
    normalized_tag_extension = _normalize_text_artifact_extension(tag_extension or '')
    request_extension = _normalize_text_artifact_extension(request.get('extension') or '') or 'txt'
    if normalized_tag_extension and request_extension == 'txt':
        request['extension'] = normalized_tag_extension
        request['source'] = 'artifact_tag'
    elif normalized_tag_extension:
        request['extension'] = request_extension
    else:
        request['extension'] = request_extension
    if normalized_tag_extension and (
        not str(request.get('source_name') or '').strip()
        or str(request.get('source_name') or '').strip() == 'generated-text'
    ):
        request['source_name'] = str(tag_source_name or '').strip() or f'generated-{normalized_tag_extension}'
    return request


def _artifact_request_with_code_block_metadata(
    artifact_request: dict[str, str],
    *,
    block_extension: Optional[str],
) -> dict[str, str]:
    request = dict(artifact_request or {})
    normalized_block_extension = _normalize_text_artifact_extension(block_extension or '')
    request_extension = _normalize_text_artifact_extension(request.get('extension') or '') or 'txt'
    if normalized_block_extension and request_extension == 'txt':
        request['extension'] = normalized_block_extension
        request['source'] = 'code_block'
        if (
            not str(request.get('source_name') or '').strip()
            or str(request.get('source_name') or '').strip() == 'generated-text'
        ):
            request['source_name'] = f'generated-{normalized_block_extension}'
    else:
        request['extension'] = request_extension
    return request


def _looks_like_raw_artifact_payload(content: str, extension: str) -> bool:
    text = str(content or '').strip()
    ext = _normalize_text_artifact_extension(extension or '') or 'txt'
    if not text:
        return False
    if ext in {'html', 'htm'}:
        return bool(re.search(r'(?is)^\s*(?:<!doctype\s+html\b|<html\b|<[a-z][\w:-]*(?:\s|>|/))', text))
    if ext == 'svg':
        return bool(re.search(r'(?is)^\s*<svg\b', text))
    if ext == 'css':
        return bool(re.search(r'(?s)[{};]', text))
    if ext in {'js', 'mjs', 'cjs', 'ts', 'tsx', 'jsx'}:
        return bool(re.search(r'(?m)^\s*(?:import|export|const|let|var|function|class|interface|type)\b', text))
    if ext == 'json':
        return text.startswith(('{', '['))
    if ext in {'yaml', 'yml'}:
        return bool(re.search(r'(?m)^\s*[-\w]+\s*:', text))
    if ext == 'xml':
        return text.startswith('<?xml') or bool(re.search(r'(?s)^\s*<[A-Za-z_][\w:.-]*(?:\s|>|/)', text))
    if ext == 'csv':
        return ',' in text and '\n' in text
    if ext in {'py', 'sh', 'sql'}:
        return bool(re.search(r'(?m)^\s*(?:def |class |import |from |#!|SELECT |CREATE |INSERT |UPDATE |DELETE )', text, re.IGNORECASE))
    return True


def _typed_text_artifact_payload_is_plausible(content: str, extension: str) -> bool:
    ext = _normalize_text_artifact_extension(extension or '') or 'txt'
    if text_artifact_content_is_materializer_instruction_echo(content):
        return False
    if ext in {'txt', 'md', 'markdown'}:
        return True
    return _looks_like_raw_artifact_payload(content, ext)


def _json_text_artifact_value_has_control_shape(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if any(key in value for key in _TEXT_ARTIFACT_CONTROL_JSON_KEYS):
        return True
    payload = value.get('payload')
    if isinstance(payload, dict) and any(key in payload for key in _TEXT_ARTIFACT_CONTROL_JSON_KEYS):
        return True
    return False


def _json_text_artifact_wrapper_candidates(text: str) -> list[Any]:
    candidates: list[Any] = []
    raw_text = str(text or '').strip()
    if raw_text:
        candidates.append(raw_text)
    for match in _TEXT_ARTIFACT_FENCED_BLOCK_RE.finditer(raw_text):
        language = _normalize_text_artifact_block_language(match.group('lang') or '')
        if language == 'json':
            body = (match.group('body') or '').strip()
            if body:
                candidates.append(body)

    parsed: list[Any] = []
    for candidate in candidates:
        try:
            parsed.append(json.loads(candidate))
        except (TypeError, ValueError):
            continue
    return parsed


def _json_text_artifact_obligations(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    raw_obligations = value.get('output_obligations')
    if raw_obligations is None and isinstance(value.get('request_ir'), dict):
        raw_obligations = value['request_ir'].get('output_obligations')
    if raw_obligations is None and isinstance(value.get('payload'), dict):
        raw_obligations = value['payload'].get('output_obligations')
    if not isinstance(raw_obligations, list):
        return []
    return [item for item in raw_obligations if isinstance(item, dict)]


def _json_text_artifact_obligation_extension(obligation: dict[str, Any]) -> Optional[str]:
    name = str(
        obligation.get('name')
        or obligation.get('filename')
        or obligation.get('file')
        or obligation.get('identifier')
        or ''
    ).strip()
    if name:
        extension = _normalize_text_artifact_extension(Path(name).suffix)
        if extension:
            return extension
    raw_type = str(
        obligation.get('mime_type')
        or obligation.get('mime')
        or obligation.get('type')
        or ''
    ).strip().lower()
    if raw_type:
        if raw_type.startswith('text/'):
            extension = _normalize_text_artifact_extension(raw_type.rsplit('/', 1)[-1])
            if extension:
                return extension
        extension = _normalize_text_artifact_extension(raw_type)
        if extension:
            return extension
    return None


def _json_text_artifact_obligation_source_name(
    obligation: dict[str, Any],
    extension: str,
) -> str:
    name = str(
        obligation.get('name')
        or obligation.get('filename')
        or obligation.get('file')
        or obligation.get('identifier')
        or obligation.get('title')
        or ''
    ).strip()
    if name:
        stem = Path(name).stem
        if stem:
            return stem
    return f'generated-{extension or "text"}'


def _json_text_artifact_content_payload_is_safe(content: str, extension: str) -> bool:
    text = str(content or '').strip()
    ext = _normalize_text_artifact_extension(extension or '') or 'txt'
    if not text or text_artifact_content_is_materializer_instruction_echo(text) or ext == 'txt':
        return False
    if ext in {'md', 'markdown'}:
        return bool(re.search(r'(?m)^\s*(?:#|[-*]\s+|```|\|)', text))
    if ext == 'json' and _json_text_artifact_control_wrapper_without_payload(text):
        return False
    return _looks_like_raw_artifact_payload(text, ext)


def _json_text_artifact_obligation_content(
    obligation: dict[str, Any],
    *,
    request_extension: str,
) -> Optional[str]:
    extension = _json_text_artifact_obligation_extension(obligation) or request_extension
    normalized_extension = _normalize_text_artifact_extension(extension or '') or 'txt'

    def safe_payload(value: Any) -> Optional[str]:
        if not isinstance(value, str) or not value.strip():
            return None
        text = value.strip()
        if text_artifact_content_is_materializer_instruction_echo(text):
            return None
        if normalized_extension == 'txt':
            return text
        if not _json_text_artifact_content_payload_is_safe(text, normalized_extension):
            return None
        return text

    content = safe_payload(obligation.get('content'))
    if content:
        return content
    content_payload = safe_payload(obligation.get('content_payload'))
    if not content_payload:
        return None
    return content_payload.strip()


def _json_text_artifact_control_wrapper_without_payload(text: str) -> bool:
    parsed_candidates = _json_text_artifact_wrapper_candidates(text)
    for candidate in parsed_candidates:
        if not _json_text_artifact_value_has_control_shape(candidate):
            continue
        obligations = _json_text_artifact_obligations(candidate)
        for obligation in obligations:
            extension = _json_text_artifact_obligation_extension(obligation) or 'txt'
            if _json_text_artifact_obligation_content(
                obligation,
                request_extension=extension,
            ):
                return False
        return True
    return False


def _artifact_request_with_json_obligation_metadata(
    artifact_request: dict[str, str],
    *,
    obligation: dict[str, Any],
) -> dict[str, str]:
    request = dict(artifact_request or {})
    request_extension = _normalize_text_artifact_extension(request.get('extension') or '') or 'txt'
    obligation_extension = _json_text_artifact_obligation_extension(obligation)
    if obligation_extension and request_extension == 'txt':
        request['extension'] = obligation_extension
    else:
        request['extension'] = request_extension
    if (
        not str(request.get('source_name') or '').strip()
        or str(request.get('source_name') or '').strip() == 'generated-text'
    ):
        request['source_name'] = _json_text_artifact_obligation_source_name(
            obligation,
            request.get('extension') or request_extension,
        )
    request['source'] = 'json_output_obligation'
    return request


def _json_text_artifact_payloads(
    text: str,
    requests: list[dict[str, str]],
) -> list[dict[str, Any]]:
    if not requests:
        return []
    parsed_candidates = _json_text_artifact_wrapper_candidates(text)
    if not parsed_candidates:
        return []

    obligations: list[dict[str, Any]] = []
    for candidate in parsed_candidates:
        obligations.extend(_json_text_artifact_obligations(candidate))
    if not obligations:
        return []

    used_obligations: set[int] = set()
    payloads: list[dict[str, Any]] = []
    for request in requests:
        request_extension = _normalize_text_artifact_extension(request.get('extension') or '') or 'txt'
        matching: list[tuple[int, dict[str, Any]]] = []
        generic: list[tuple[int, dict[str, Any]]] = []
        for index, obligation in enumerate(obligations):
            if index in used_obligations:
                continue
            content = _json_text_artifact_obligation_content(
                obligation,
                request_extension=request_extension,
            )
            if not content:
                continue
            obligation_extension = _json_text_artifact_obligation_extension(obligation)
            if (
                obligation_extension
                and request_extension != 'txt'
                and obligation_extension == request_extension
            ):
                matching.append((index, obligation))
            elif request_extension == 'txt' or not obligation_extension:
                generic.append((index, obligation))
        selected = matching[0] if matching else (generic[0] if generic else None)
        if not selected:
            continue
        index, obligation = selected
        used_obligations.add(index)
        payloads.append(
            {
                'artifact_request': _artifact_request_with_json_obligation_metadata(
                    request,
                    obligation=obligation,
                ),
                'content': str(
                    _json_text_artifact_obligation_content(
                        obligation,
                        request_extension=request_extension,
                    )
                    or ''
                ).strip(),
            }
        )
    return payloads


def extract_text_artifact_payload(content: str, artifact_request: Optional[dict[str, str]]) -> Optional[str]:
    """Return the exact file payload to persist, or None when no truthful payload exists."""

    payloads = extract_text_artifact_payloads(content, [artifact_request] if artifact_request else [])
    return str(payloads[0].get('content') or '') if payloads else None


def extract_text_artifact_payloads(
    content: str,
    artifact_requests: Optional[list[dict[str, str]]],
) -> list[dict[str, Any]]:
    """Return exact file payloads to persist for each requested text artifact."""

    text = str(content or '').strip()
    requests = [item for item in (artifact_requests or []) if isinstance(item, dict)]
    if not text or not requests:
        return []
    blocks = [
        {
            'index': index,
            'language': match.group('lang') or '',
            'extension': _text_artifact_block_extension(match.group('lang') or ''),
            'is_payload': _text_artifact_block_language_is_payload(match.group('lang') or ''),
            'is_control_json': (
                _normalize_text_artifact_block_language(match.group('lang') or '') == 'json'
                and _json_text_artifact_control_wrapper_without_payload((match.group('body') or '').strip())
            ),
            'body': (match.group('body') or '').strip(),
        }
        for index, match in enumerate(_TEXT_ARTIFACT_FENCED_BLOCK_RE.finditer(text))
        if (match.group('body') or '').strip()
    ]
    artifact_tags = []
    for index, match in enumerate(_TEXT_ARTIFACT_TAG_RE.finditer(text)):
        body = (match.group('body') or '').strip()
        if not body:
            continue
        attrs = _parse_text_artifact_tag_attrs(match.group('attrs') or '')
        tag_extension = _text_artifact_tag_extension(attrs)
        artifact_tags.append(
            {
                'index': index,
                'attrs': attrs,
                'extension': tag_extension,
                'source_name': _text_artifact_tag_source_name(attrs, tag_extension or 'txt'),
                'body': body,
            }
        )
    used_blocks: set[int] = set()
    used_tags: set[int] = set()
    fulfilled_request_indexes: set[int] = set()
    payloads: list[dict[str, Any]] = []

    for request_index, request in enumerate(requests):
        extension = request.get('extension') or 'txt'
        matching_blocks = [
            block for block in blocks
            if block['index'] not in used_blocks
            and block.get('is_payload') is not False
            and not block.get('is_control_json')
            and _code_block_language_matches_extension(str(block.get('language') or ''), extension)
            and _typed_text_artifact_payload_is_plausible(str(block.get('body') or ''), extension)
        ]
        if matching_blocks:
            block = matching_blocks[0]
            used_blocks.add(int(block['index']))
            fulfilled_request_indexes.add(request_index)
            payloads.append({'artifact_request': request, 'content': str(block.get('body') or '')})
            continue

        matching_tags = [
            tag for tag in artifact_tags
            if tag['index'] not in used_tags
            and (
                _normalize_text_artifact_extension(str(tag.get('extension') or '')) == _normalize_text_artifact_extension(extension or '')
                or _normalize_text_artifact_extension(extension or '') == 'txt'
            )
            and _typed_text_artifact_payload_is_plausible(str(tag.get('body') or ''), extension)
        ]
        if matching_tags:
            tag = matching_tags[0]
            used_tags.add(int(tag['index']))
            request_with_tag = _artifact_request_with_tag_metadata(
                request,
                tag_extension=str(tag.get('extension') or ''),
                tag_source_name=str(tag.get('source_name') or ''),
            )
            fulfilled_request_indexes.add(request_index)
            payloads.append({'artifact_request': request_with_tag, 'content': str(tag.get('body') or '')})
            continue

    remaining_requests = [
        request for request_index, request in enumerate(requests)
        if request_index not in fulfilled_request_indexes
    ]
    json_payloads = _json_text_artifact_payloads(text, remaining_requests)
    if json_payloads:
        return payloads + json_payloads

    for request_index, request in enumerate(requests):
        if request_index in fulfilled_request_indexes:
            continue
        extension = request.get('extension') or 'txt'
        if blocks and _normalize_text_artifact_extension(extension or '') in {'txt', 'md'}:
            concrete_blocks = [
                block for block in blocks
                if block['index'] not in used_blocks
                and block.get('is_payload') is not False
                and not block.get('is_control_json')
                and _normalize_text_artifact_extension(str(block.get('extension') or ''))
                and _typed_text_artifact_payload_is_plausible(str(block.get('body') or ''), str(block.get('extension') or extension))
            ]
            if concrete_blocks:
                block = concrete_blocks[0]
                used_blocks.add(int(block['index']))
                fulfilled_request_indexes.add(request_index)
                request_with_block = _artifact_request_with_code_block_metadata(
                    request,
                    block_extension=str(block.get('extension') or ''),
                )
                payloads.append({'artifact_request': request_with_block, 'content': str(block.get('body') or '')})
                continue
            available = [
                block for block in blocks
                if block['index'] not in used_blocks
                and block.get('is_payload') is not False
                and not block.get('is_control_json')
                and _typed_text_artifact_payload_is_plausible(str(block.get('body') or ''), extension)
            ]
            if available:
                block = available[0]
                used_blocks.add(int(block['index']))
                fulfilled_request_indexes.add(request_index)
                payloads.append({'artifact_request': request, 'content': str(block.get('body') or '')})
                continue

    if payloads:
        return payloads

    if len(requests) != 1:
        return []
    artifact_request = requests[0]
    extension = artifact_request.get('extension') or 'txt'
    raw_blocks = [
        (match.group('lang') or '', (match.group('body') or '').strip())
        for match in _TEXT_ARTIFACT_FENCED_BLOCK_RE.finditer(text)
        if (match.group('body') or '').strip()
        and _text_artifact_block_language_is_payload(match.group('lang') or '')
        and not (
            _normalize_text_artifact_block_language(match.group('lang') or '') == 'json'
            and _json_text_artifact_control_wrapper_without_payload((match.group('body') or '').strip())
        )
    ]
    matching_raw_blocks = [
        body for language, body in raw_blocks
        if _code_block_language_matches_extension(language, extension)
        and _typed_text_artifact_payload_is_plausible(body, extension)
    ]
    if matching_raw_blocks:
        return [{'artifact_request': artifact_request, 'content': max(matching_raw_blocks, key=len)}]
    matching_raw_tags = [
        tag for tag in artifact_tags
        if (
            _normalize_text_artifact_extension(str(tag.get('extension') or '')) == _normalize_text_artifact_extension(extension or '')
            or _normalize_text_artifact_extension(extension or '') == 'txt'
        )
        and _typed_text_artifact_payload_is_plausible(str(tag.get('body') or ''), extension)
    ]
    if matching_raw_tags:
        tag = matching_raw_tags[0]
        request_with_tag = _artifact_request_with_tag_metadata(
            artifact_request,
            tag_extension=str(tag.get('extension') or ''),
            tag_source_name=str(tag.get('source_name') or ''),
        )
        return [{'artifact_request': request_with_tag, 'content': str(tag.get('body') or '')}]
    json_payloads = _json_text_artifact_payloads(text, [artifact_request])
    if json_payloads:
        return json_payloads
    if raw_blocks and _normalize_text_artifact_extension(extension or '') in {'txt', 'md'}:
        concrete_raw_blocks = [
            (language, body)
            for language, body in raw_blocks
            if _text_artifact_block_extension(language)
            and _typed_text_artifact_payload_is_plausible(body, _text_artifact_block_extension(language))
        ]
        if concrete_raw_blocks:
            language, body = max(concrete_raw_blocks, key=lambda item: len(item[1]))
            request_with_block = _artifact_request_with_code_block_metadata(
                artifact_request,
                block_extension=_text_artifact_block_extension(language),
            )
            return [{'artifact_request': request_with_block, 'content': body}]
        return [{'artifact_request': artifact_request, 'content': max((body for _language, body in raw_blocks), key=len)}]
    if generated_text_blocks_artifact_persistence(text):
        return []
    if generated_text_is_artifact_self_claim(text):
        return []
    if _json_text_artifact_control_wrapper_without_payload(text):
        return []
    if _typed_text_artifact_payload_is_plausible(text, extension):
        return [{'artifact_request': artifact_request, 'content': text}]
    return []

_TTS_AUTO_LANGUAGE_CODES = {'auto', 'autodetect', 'detect'}
_TTS_LANGUAGE_ALIASES: dict[str, tuple[str, ...]] = {
    'english': ('english', 'en', 'eng'),
    'german': ('german', 'de', 'deu', 'deutsch'),
    'french': ('french', 'fr', 'fra', 'francais', 'français'),
    'spanish': ('spanish', 'es', 'spa', 'espanol', 'español'),
    'italian': ('italian', 'it', 'ita', 'italiano'),
    'portuguese': ('portuguese', 'pt', 'por', 'portugues', 'português'),
    'chinese': ('chinese', 'zh', 'zho', 'mandarin'),
    'japanese': ('japanese', 'ja', 'jpn'),
    'korean': ('korean', 'ko', 'kor'),
    'russian': ('russian', 'ru', 'rus'),
}
_TTS_FALLBACK_LANGUAGE_CODES: dict[str, str] = {
    'english': 'en',
    'german': 'de',
    'french': 'fr',
    'spanish': 'es',
    'italian': 'it',
    'portuguese': 'pt',
    'chinese': 'zh',
    'japanese': 'ja',
    'korean': 'ko',
    'russian': 'ru',
}
_TTS_LANGUAGE_MARKERS: dict[str, tuple[str, ...]] = {
    'english': (
        'the', 'and', 'that', 'with', 'from', 'this', 'was', 'were', 'once', 'little', 'story',
    ),
    'german': (
        'der', 'die', 'das', 'und', 'ist', 'war', 'waren', 'ein', 'eine', 'einem', 'einen', 'eines',
        'nicht', 'mit', 'als', 'dass', 'wurde', 'wusste', 'plötzlich', 'kleiner', 'geschichte',
        'guten', 'tag', 'hallo', 'aus',
    ),
    'french': (
        'le', 'la', 'les', 'des', 'une', 'un', 'et', 'est', 'était', 'dans', 'avec', 'que',
        'bonjour', 'fois', 'petit',
    ),
    'spanish': (
        'el', 'la', 'los', 'las', 'una', 'un', 'y', 'es', 'era', 'con', 'que', 'hola', 'había',
    ),
    'italian': (
        'il', 'lo', 'la', 'gli', 'una', 'un', 'e', 'è', 'era', 'con', 'che', 'ciao', 'volta',
    ),
    'portuguese': (
        'o', 'a', 'os', 'as', 'um', 'uma', 'e', 'é', 'era', 'com', 'que', 'olá', 'vez',
    ),
    'russian': ('и', 'в', 'не', 'что', 'он', 'она', 'это', 'как', 'привет'),
    'japanese': ('です', 'ます', 'した', 'する', 'これ', 'そして'),
    'korean': ('합니다', '했다', '그리고', '있는', '에서'),
    'chinese': ('的', '了', '在', '是', '和', '有', '一个'),
}
_TTS_LANGUAGE_CHARS: dict[str, str] = {
    'german': 'äöüß',
    'french': 'àâçéèêëîïôùûüÿœæ',
    'spanish': 'áéíñóúü¿¡',
    'italian': 'àèéìíîòóù',
    'portuguese': 'áâãàçéêíóôõú',
}
_TTS_TOKEN_RE = re.compile(r"[a-zà-ÿ]+", re.IGNORECASE)
_TTS_SPOKEN_WORD_RE = re.compile(
    r"[^\W_]+(?:['\u2019][^\W_]+)?",
    re.UNICODE,
)
_QWEN3_TTS_GENERATION_BUDGET_POLICY_ID = 'qwen3_tts_adaptive_audio_tokens_v2'
_QWEN3_TTS_SAMPLING_PROFILE_POLICY_ID = 'qwen3_tts_model_native_sampling_v1'
_QWEN3_TTS_AUDIO_TOKENS_PER_SECOND = 12.5
_QWEN3_TTS_WORDS_PER_SECOND = 2.0
_QWEN3_TTS_ORDINARY_CHARS_PER_SECOND = 12.0
_QWEN3_TTS_CJK_CHARS_PER_SECOND = 4.0
_QWEN3_TTS_DURATION_SAFETY_MULTIPLIER = 1.5
_QWEN3_TTS_FIXED_BUFFER_SECONDS = 8.0
_QWEN3_TTS_MIN_GENERATION_TOKENS = 256
_QWEN3_TTS_MAX_GENERATION_TOKENS = 1200
_QWEN3_TTS_TEMPERATURE = 0.9
_QWEN3_TTS_TOP_P = 1.0
_QWEN3_TTS_TOP_K = 50
_QWEN3_TTS_REPETITION_PENALTY = 1.05
_QWEN3_TTS_CHUNKING_POLICY_ID = 'qwen3_tts_sentence_chunks_v1'
_QWEN3_TTS_CHUNK_TRIGGER_SPEECH_SECONDS = 16.0
_QWEN3_TTS_CHUNK_TARGET_SPEECH_SECONDS = 10.0
_QWEN3_TTS_GENERATION_LIMIT_RECOVERY_POLICY_ID = (
    'qwen3_tts_single_sequence_generation_limit_retry_v2'
)
_QWEN3_TTS_GENERATION_LIMIT_RECOVERY_MULTIPLIER = 1.5
_QWEN3_TTS_GENERATION_LIMIT_RECOVERY_ADDITIONAL_TOKENS = 128
_QWEN3_TTS_GENERATION_LIMIT_RECOVERY_MODEL_TYPES = {
    'base',
    'custom_voice',
    'voice_design',
}
_TTS_SENTENCE_END_RE = re.compile(
    r'(?:[.!?…]+["\'’”»\)\]]*|\n{2,})(?=\s|$)'
)
_TTS_CHUNK_BOUNDARY_RE = re.compile(r'(?:[,;:—–-]\s+|\s+)')


@dataclass
class InferContext:
    instance_id: str
    backend: str
    capability: str
    model_name: str
    port: int
    prompt: str
    user_prompt: str
    infer_timeout_sec: int
    pdf_page_timeout_sec: int
    pdf_max_image_side: int
    pdf_synthesize: bool
    task: str = 'transcribe'
    language: Optional[str] = None
    voice: Optional[str] = None
    instruct: Optional[str] = None
    response_format: Optional[str] = None
    speed: float = 1.0
    pitch: float = 1.0
    lang_code: Optional[str] = None
    tts_model_type: Optional[str] = None
    tts_speakers: list[str] = field(default_factory=list)
    tts_languages: list[str] = field(default_factory=list)
    ocr_mode: Optional[str] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    image_seed: Optional[int] = None
    text_artifact_requests: list[dict[str, str]] = field(default_factory=list)
    prompt_is_semantic_materializer_payload: bool = False


@dataclass
class InferArtifacts:
    temp_path: Optional[Path] = None
    file_kind: str = ''
    file_name: str = ''
    file_sha256: str = ''
    image_b64: Optional[str] = None
    text_from_file: str = ''
    text_from_file_truncated: bool = False
    text_from_file_inline_bytes: int = 0
    text_from_file_total_bytes: int = 0
    pdf_page_images: list[str] = field(default_factory=list)
    pdf_warnings: list[str] = field(default_factory=list)
    pdf_total_pages: int = 0
    pdf_render_dpi: int = 180
    pdf_page_retry_dpi: int = 120


def dispatch_infer_request(
    ctx: InferContext,
    artifacts: InferArtifacts,
    ops: Dict[str, Callable[..., Any]],
) -> Tuple[dict, int]:
    if ctx.capability == CAPABILITY_SPEECH_TO_TEXT:
        return _run_speech_to_text(ctx, artifacts, ops)
    if ctx.capability == CAPABILITY_TEXT_TO_SPEECH:
        return _run_text_to_speech(ctx, artifacts, ops)
    if ctx.capability == CAPABILITY_IMAGE_GENERATION:
        return _run_image_generation(ctx, artifacts, ops)
    if ctx.capability == CAPABILITY_VISION_ANALYSIS:
        return _run_vision_analysis(ctx, artifacts, ops)
    return _run_chat_fallback(ctx, artifacts, ops)


def _extract_transcript_text(result: dict) -> str:
    text = str(result.get('text') or '').strip()
    if text:
        return text
    segments = result.get('segments')
    if not isinstance(segments, list):
        return ''
    parts: list[str] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        chunk = str(segment.get('text') or '').strip()
        if chunk:
            parts.append(chunk)
    return '\n'.join(parts).strip()


def _effective_vision_prompt(ctx: InferContext) -> str:
    return resolve_ocr_prompt(
        ctx.model_name,
        user_prompt=ctx.prompt,
        ocr_mode=ctx.ocr_mode,
        generic_fallback=GENERIC_OCR_FALLBACK_PROMPT,
    )


def _run_backend_chat_completion(
    ctx: InferContext,
    ops: Dict[str, Callable[..., Any]],
    messages: list[dict],
    *,
    timeout_sec: int,
) -> dict:
    if ctx.backend == 'mlx':
        return ops['mlx_chat_completions'](
            ctx.port,
            ctx.model_name,
            messages,
            timeout_sec=timeout_sec,
        )
    return ops['openai_chat_completions'](
        ctx.port,
        ctx.model_name,
        messages,
        timeout_sec=timeout_sec,
    )


def _build_pdf_inline_response_content(
    full_content: str,
    *,
    warnings: list[str],
    max_inline_chars: Optional[int],
) -> tuple[str, bool]:
    response_content = str(full_content or '')
    if not response_content or not max_inline_chars or len(response_content) <= max_inline_chars:
        return response_content, False
    warnings.append(
        f'UI output was truncated to {max_inline_chars} chars for stability. '
        'Use the saved markdown artifact for full content.'
    )
    return (
        response_content[:max_inline_chars].rstrip()
        + '\n\n...[UI output truncated for stability; full OCR result saved locally.]'
    ), True


def _run_speech_to_text(ctx: InferContext, artifacts: InferArtifacts, ops: Dict[str, Callable[..., Any]]) -> Tuple[dict, int]:
    if not artifacts.temp_path:
        return {'error': 'speech_to_text requires an audio file.'}, 400
    if artifacts.file_kind != 'audio':
        return {'error': f"Expected an audio file, received: {artifacts.file_kind or 'unknown'}."}, 400
    result = ops['whisper_transcribe'](ctx.port, artifacts.temp_path, task=ctx.task, language=ctx.language)
    transcript_text = _extract_transcript_text(result)
    if not transcript_text:
        return {'error': 'Whisper did not return transcript text.'}, 502
    saved_text_path = ops['persist_transcript_text_locally'](
        transcript_text,
        model_name=ctx.model_name,
        source_file_name=artifacts.file_name or 'audio',
        mode='speech_to_text',
    )
    return (
        {
            'instance_id': ctx.instance_id,
            'capability': ctx.capability,
            'mode': 'speech_to_text',
            'content': transcript_text,
            'saved_text_path': saved_text_path,
            'result': result,
        },
        200,
    )


def _normalize_language_token(value: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(value or '').strip().lower())


def _supported_tts_language_token(language: str, supported_languages: list[str]) -> Optional[str]:
    normalized_language = _normalize_language_token(language)
    aliases = {
        _normalize_language_token(alias)
        for alias in _TTS_LANGUAGE_ALIASES.get(language, (language,))
    }
    if normalized_language:
        aliases.add(normalized_language)
    supported = [str(item or '').strip() for item in supported_languages if str(item or '').strip()]
    if not supported:
        return _TTS_FALLBACK_LANGUAGE_CODES.get(language)
    for token in supported:
        normalized_supported = _normalize_language_token(token)
        if normalized_supported in aliases:
            return token
    return None


def _score_tts_language(text: str, language: str) -> int:
    normalized_text = str(text or '').strip().lower()
    if not normalized_text:
        return 0
    token_counts = Counter(_TTS_TOKEN_RE.findall(normalized_text))
    markers = _TTS_LANGUAGE_MARKERS.get(language, ())
    score = sum(min(token_counts.get(marker, 0), 3) for marker in markers)
    char_markers = _TTS_LANGUAGE_CHARS.get(language, '')
    if char_markers and any(char in normalized_text for char in char_markers):
        score += 4
    if language == 'russian' and re.search(r'[\u0400-\u04ff]', normalized_text):
        score += 5
    if language == 'japanese' and re.search(r'[\u3040-\u30ff]', normalized_text):
        score += 5
    if language == 'korean' and re.search(r'[\uac00-\ud7af]', normalized_text):
        score += 5
    if language == 'chinese' and re.search(r'[\u4e00-\u9fff]', normalized_text):
        score += 5
    return score


def _infer_tts_language_from_text(text: str, supported_languages: list[str]) -> Optional[str]:
    scores = {
        language: _score_tts_language(text, language)
        for language in _TTS_LANGUAGE_ALIASES
    }
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked:
        return None
    top_language, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0
    if top_score < 2 or top_score <= second_score:
        return None
    return _supported_tts_language_token(top_language, supported_languages)


def _resolve_effective_tts_lang_code(
    explicit_lang_code: Optional[str],
    spoken_text: str,
    supported_languages: list[str],
) -> tuple[Optional[str], Optional[str]]:
    explicit = str(explicit_lang_code or '').strip()
    if explicit and _normalize_language_token(explicit) not in _TTS_AUTO_LANGUAGE_CODES:
        return explicit, 'explicit'
    inferred = _infer_tts_language_from_text(spoken_text, supported_languages)
    if inferred:
        return inferred, 'inferred_from_text'
    return None, None


def _is_qwen3_tts_model(model_name: Any) -> bool:
    return bool(
        re.search(
            r'(?<![a-z0-9])qwen3[^a-z0-9]*tts(?![a-z0-9])',
            str(model_name or '').strip().lower(),
        )
    )


def _canonicalize_qwen3_tts_lang_code(
    lang_code: Optional[str],
) -> tuple[Optional[str], bool]:
    """Map known aliases to the full language tokens accepted by Qwen3-TTS."""

    value = str(lang_code or '').strip()
    if not value:
        return None, False
    normalized = _normalize_language_token(value)
    if normalized in _TTS_AUTO_LANGUAGE_CODES:
        return value, False
    for canonical_language, aliases in _TTS_LANGUAGE_ALIASES.items():
        if normalized == _normalize_language_token(canonical_language):
            return value, False
        if normalized in {
            _normalize_language_token(alias)
            for alias in aliases
        }:
            return canonical_language, True
    return value, False


def _is_cjk_speech_character(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x3040 <= codepoint <= 0x30FF
        or 0xAC00 <= codepoint <= 0xD7AF
    )


def build_qwen3_tts_generation_budget(spoken_text: Any) -> dict[str, Any]:
    """Return an observable, conservative Qwen3 audio-token safety budget."""

    text = str(spoken_text or '')
    word_count = len(_TTS_SPOKEN_WORD_RE.findall(text))
    visible_characters = [character for character in text if not character.isspace()]
    cjk_character_count = sum(
        1
        for character in visible_characters
        if _is_cjk_speech_character(character)
    )
    ordinary_character_count = len(visible_characters) - cjk_character_count
    word_estimate_seconds = word_count / _QWEN3_TTS_WORDS_PER_SECOND
    character_estimate_seconds = (
        ordinary_character_count / _QWEN3_TTS_ORDINARY_CHARS_PER_SECOND
        + cjk_character_count / _QWEN3_TTS_CJK_CHARS_PER_SECOND
    )
    estimated_speech_seconds = max(
        word_estimate_seconds,
        character_estimate_seconds,
    )
    buffered_duration_seconds = (
        estimated_speech_seconds * _QWEN3_TTS_DURATION_SAFETY_MULTIPLIER
        + _QWEN3_TTS_FIXED_BUFFER_SECONDS
    )
    calculated_tokens = int(
        math.ceil(
            buffered_duration_seconds
            * _QWEN3_TTS_AUDIO_TOKENS_PER_SECOND
        )
    )
    max_tokens = min(
        _QWEN3_TTS_MAX_GENERATION_TOKENS,
        max(_QWEN3_TTS_MIN_GENERATION_TOKENS, calculated_tokens),
    )
    clamp = (
        'minimum'
        if max_tokens > calculated_tokens
        else 'maximum'
        if max_tokens < calculated_tokens
        else 'none'
    )
    return {
        'kind': 'ollmo.tts_generation_budget',
        'version': 1,
        'policy_id': _QWEN3_TTS_GENERATION_BUDGET_POLICY_ID,
        'model_family': 'qwen3_tts',
        'max_tokens': max_tokens,
        'calculated_tokens_before_clamp': calculated_tokens,
        'clamp': clamp,
        'source_word_count': word_count,
        'source_visible_character_count': len(visible_characters),
        'source_cjk_character_count': cjk_character_count,
        'estimated_speech_seconds': round(estimated_speech_seconds, 6),
        'buffered_duration_seconds': round(buffered_duration_seconds, 6),
        'policy': {
            'audio_tokens_per_second': _QWEN3_TTS_AUDIO_TOKENS_PER_SECOND,
            'words_per_second': _QWEN3_TTS_WORDS_PER_SECOND,
            'ordinary_characters_per_second': _QWEN3_TTS_ORDINARY_CHARS_PER_SECOND,
            'cjk_characters_per_second': _QWEN3_TTS_CJK_CHARS_PER_SECOND,
            'duration_safety_multiplier': _QWEN3_TTS_DURATION_SAFETY_MULTIPLIER,
            'fixed_buffer_seconds': _QWEN3_TTS_FIXED_BUFFER_SECONDS,
            'minimum_tokens': _QWEN3_TTS_MIN_GENERATION_TOKENS,
            'maximum_tokens': _QWEN3_TTS_MAX_GENERATION_TOKENS,
        },
    }


def _build_qwen3_tts_sampling_profile() -> dict[str, Any]:
    """Return the explicit Qwen3 sampling contract sent to MLX Audio."""

    return {
        'kind': 'ollmo.tts_sampling_profile',
        'version': 1,
        'policy_id': _QWEN3_TTS_SAMPLING_PROFILE_POLICY_ID,
        'model_family': 'qwen3_tts',
        'source': 'mlx_audio_qwen3_tts_model_defaults',
        'temperature': _QWEN3_TTS_TEMPERATURE,
        'top_p': _QWEN3_TTS_TOP_P,
        'top_k': _QWEN3_TTS_TOP_K,
        'repetition_penalty': _QWEN3_TTS_REPETITION_PENALTY,
    }


def _trim_text_span(text: str, start: int, end: int) -> tuple[int, int]:
    bounded_start = max(0, min(len(text), start))
    bounded_end = max(bounded_start, min(len(text), end))
    while bounded_start < bounded_end and text[bounded_start].isspace():
        bounded_start += 1
    while bounded_end > bounded_start and text[bounded_end - 1].isspace():
        bounded_end -= 1
    return bounded_start, bounded_end


def _qwen3_tts_estimated_speech_seconds(text: str) -> float:
    return float(
        build_qwen3_tts_generation_budget(text).get('estimated_speech_seconds')
        or 0.0
    )


def _split_qwen3_tts_span_to_target(
    text: str,
    start: int,
    end: int,
) -> list[tuple[int, int]]:
    start, end = _trim_text_span(text, start, end)
    if start >= end:
        return []
    if (
        _qwen3_tts_estimated_speech_seconds(text[start:end])
        <= _QWEN3_TTS_CHUNK_TARGET_SPEECH_SECONDS
    ):
        return [(start, end)]

    spans: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        remaining_start, remaining_end = _trim_text_span(text, cursor, end)
        if remaining_start >= remaining_end:
            break
        remaining = text[remaining_start:remaining_end]
        if (
            _qwen3_tts_estimated_speech_seconds(remaining)
            <= _QWEN3_TTS_CHUNK_TARGET_SPEECH_SECONDS
        ):
            spans.append((remaining_start, remaining_end))
            break

        selected_cut = 0
        first_oversized_cut = 0
        for boundary in _TTS_CHUNK_BOUNDARY_RE.finditer(remaining):
            cut = boundary.end()
            candidate = remaining[:cut].strip()
            if not candidate:
                continue
            estimate = _qwen3_tts_estimated_speech_seconds(candidate)
            if estimate <= _QWEN3_TTS_CHUNK_TARGET_SPEECH_SECONDS:
                selected_cut = cut
                continue
            first_oversized_cut = cut
            break
        if not selected_cut:
            selected_cut = first_oversized_cut
        if not selected_cut:
            visible_limit = max(
                1,
                int(
                    _QWEN3_TTS_CHUNK_TARGET_SPEECH_SECONDS
                    * _QWEN3_TTS_CJK_CHARS_PER_SECOND
                ),
            )
            selected_cut = min(len(remaining), visible_limit)
        absolute_cut = min(remaining_end, remaining_start + selected_cut)
        chunk_start, chunk_end = _trim_text_span(
            text,
            remaining_start,
            absolute_cut,
        )
        if chunk_start >= chunk_end:
            absolute_cut = min(remaining_end, remaining_start + 1)
            chunk_start, chunk_end = _trim_text_span(
                text,
                remaining_start,
                absolute_cut,
            )
        if chunk_start < chunk_end:
            spans.append((chunk_start, chunk_end))
        cursor = max(absolute_cut, remaining_start + 1)
    return spans


def build_qwen3_tts_chunk_plan(spoken_text: Any) -> dict[str, Any]:
    """Return deterministic ordered sentence/clause spans for long Qwen speech."""

    text = str(spoken_text or '')
    source_sha256 = hashlib.sha256(text.encode('utf-8')).hexdigest()
    full_estimated_seconds = _qwen3_tts_estimated_speech_seconds(text)
    sentence_spans: list[tuple[int, int]] = []
    cursor = 0
    for boundary in _TTS_SENTENCE_END_RE.finditer(text):
        span = _trim_text_span(text, cursor, boundary.end())
        if span[0] < span[1]:
            sentence_spans.append(span)
        cursor = boundary.end()
    trailing = _trim_text_span(text, cursor, len(text))
    if trailing[0] < trailing[1]:
        sentence_spans.append(trailing)
    if not sentence_spans and text.strip():
        sentence_spans = [_trim_text_span(text, 0, len(text))]

    chunk_spans: list[tuple[int, int]] = []
    pending: Optional[tuple[int, int]] = None
    for sentence_start, sentence_end in sentence_spans:
        if pending is None:
            pending = (sentence_start, sentence_end)
            continue
        candidate = text[pending[0]:sentence_end]
        if (
            _qwen3_tts_estimated_speech_seconds(candidate)
            <= _QWEN3_TTS_CHUNK_TARGET_SPEECH_SECONDS
        ):
            pending = (pending[0], sentence_end)
            continue
        chunk_spans.extend(
            _split_qwen3_tts_span_to_target(text, pending[0], pending[1])
        )
        pending = (sentence_start, sentence_end)
    if pending is not None:
        chunk_spans.extend(
            _split_qwen3_tts_span_to_target(text, pending[0], pending[1])
        )

    ordered_span_coverage = bool(chunk_spans)
    previous_end = 0
    for span_start, span_end in chunk_spans:
        if (
            span_start < previous_end
            or text[previous_end:span_start].strip()
            or not text[span_start:span_end].strip()
        ):
            ordered_span_coverage = False
            break
        previous_end = span_end
    if text[previous_end:].strip():
        ordered_span_coverage = False

    applied = bool(
        full_estimated_seconds > _QWEN3_TTS_CHUNK_TRIGGER_SPEECH_SECONDS
        and len(chunk_spans) > 1
        and ordered_span_coverage
    )
    chunks = [
        {
            'index': index,
            'source_span_start': span_start,
            'source_span_end': span_end,
            'text': text[span_start:span_end],
            'text_sha256': hashlib.sha256(
                text[span_start:span_end].encode('utf-8')
            ).hexdigest(),
            'estimated_speech_seconds': round(
                _qwen3_tts_estimated_speech_seconds(
                    text[span_start:span_end]
                ),
                6,
            ),
        }
        for index, (span_start, span_end) in enumerate(chunk_spans, start=1)
    ]
    return {
        'kind': 'ollmo.tts_chunking_plan',
        'version': 1,
        'policy_id': _QWEN3_TTS_CHUNKING_POLICY_ID,
        'model_family': 'qwen3_tts',
        'status': 'planned' if applied else 'not_applied',
        'applied': applied,
        'reason': (
            'estimated speech exceeds the single-request long-form threshold'
            if applied
            else 'source remains within the single-request long-form threshold'
            if full_estimated_seconds <= _QWEN3_TTS_CHUNK_TRIGGER_SPEECH_SECONDS
            else 'safe ordered multi-chunk coverage was not available'
        ),
        'source_sha256': source_sha256,
        'source_character_count': len(text),
        'estimated_speech_seconds': round(full_estimated_seconds, 6),
        'trigger_speech_seconds': _QWEN3_TTS_CHUNK_TRIGGER_SPEECH_SECONDS,
        'target_chunk_speech_seconds': _QWEN3_TTS_CHUNK_TARGET_SPEECH_SECONDS,
        'ordered_span_coverage': ordered_span_coverage,
        'chunk_count': len(chunks) if applied else 1,
        'chunks': chunks if applied else [],
    }


def _build_qwen3_tts_generation_limit_recovery(
    initial_budget: Mapping[str, Any],
    *,
    integrity_evidence: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Return one bounded Qwen single-sequence retry budget after cap exhaustion."""

    budget = dict(initial_budget)
    model_family = str(budget.get('model_family') or '').strip().lower()
    tts_model_type = str(budget.get('tts_model_type') or '').strip().lower()
    generation_scope = str(budget.get('generation_scope') or '').strip().lower()
    supported_sequence = bool(
        model_family == 'qwen3_tts'
        and tts_model_type
        in _QWEN3_TTS_GENERATION_LIMIT_RECOVERY_MODEL_TYPES
        and generation_scope == 'single_sequence'
    )
    try:
        initial_max_tokens = int(budget.get('max_tokens') or 0)
    except (TypeError, ValueError):
        initial_max_tokens = 0
    calculated_recovery_tokens = max(
        initial_max_tokens
        + _QWEN3_TTS_GENERATION_LIMIT_RECOVERY_ADDITIONAL_TOKENS,
        int(
            math.ceil(
                initial_max_tokens
                * _QWEN3_TTS_GENERATION_LIMIT_RECOVERY_MULTIPLIER
            )
        ),
    )
    recovery_max_tokens = min(
        _QWEN3_TTS_MAX_GENERATION_TOKENS,
        calculated_recovery_tokens,
    )
    applied = bool(
        supported_sequence
        and initial_max_tokens > 0
        and recovery_max_tokens > initial_max_tokens
    )
    recovery_budget = dict(budget)
    recovery_budget.update(
        {
            'policy_id': _QWEN3_TTS_GENERATION_LIMIT_RECOVERY_POLICY_ID,
            'base_policy_id': str(budget.get('policy_id') or '').strip() or None,
            'max_tokens': recovery_max_tokens,
            'calculated_tokens_before_clamp': calculated_recovery_tokens,
            'clamp': (
                'maximum'
                if recovery_max_tokens < calculated_recovery_tokens
                else 'none'
            ),
            'recovery_trigger_reason_code': (
                'TTS_AUDIO_GENERATION_LIMIT_EXHAUSTED'
            ),
            'recovery_initial_max_tokens': initial_max_tokens,
        }
    )
    return {
        'kind': 'ollmo.tts_generation_limit_recovery',
        'version': 1,
        'policy_id': _QWEN3_TTS_GENERATION_LIMIT_RECOVERY_POLICY_ID,
        'status': 'eligible' if applied else 'not_eligible',
        'applied': applied,
        'reason': (
            'one larger bounded Qwen single-sequence retry budget is available'
            if applied
            else 'model scope is unsupported, the initial budget is invalid, or the hard maximum is already active'
        ),
        'trigger_reason_code': 'TTS_AUDIO_GENERATION_LIMIT_EXHAUSTED',
        'model_family_scope': 'qwen3_tts',
        'model_type_scope': sorted(
            _QWEN3_TTS_GENERATION_LIMIT_RECOVERY_MODEL_TYPES
        ),
        'tts_model_type': tts_model_type or None,
        'generation_scope': generation_scope or None,
        'maximum_retry_count': 1,
        'initial_max_tokens': initial_max_tokens,
        'calculated_recovery_tokens_before_clamp': calculated_recovery_tokens,
        'recovery_max_tokens': recovery_max_tokens,
        'clamp': (
            'maximum'
            if recovery_max_tokens < calculated_recovery_tokens
            else 'none'
        ),
        'policy': {
            'multiplier': _QWEN3_TTS_GENERATION_LIMIT_RECOVERY_MULTIPLIER,
            'additional_tokens': (
                _QWEN3_TTS_GENERATION_LIMIT_RECOVERY_ADDITIONAL_TOKENS
            ),
            'maximum_tokens': _QWEN3_TTS_MAX_GENERATION_TOKENS,
        },
        'trigger_primary_reason_code': str(
            (integrity_evidence or {}).get('reason_code') or ''
        ).strip() or None,
        'trigger_defect_codes': [
            str(item).strip()
            for item in (integrity_evidence or {}).get('defect_codes') or []
            if str(item).strip()
        ] if isinstance((integrity_evidence or {}).get('defect_codes'), list) else [],
        'trigger_generation_limit_evidence': dict(
            (integrity_evidence or {}).get('generation_limit_evidence') or {}
        ) if isinstance(
            (integrity_evidence or {}).get('generation_limit_evidence'),
            Mapping,
        ) else {},
        'generation_budget': recovery_budget if applied else None,
    }


def _qwen3_tts_budget_with_scope(
    spoken_text: str,
    *,
    tts_model_type: str,
    generation_scope: str,
) -> dict[str, Any]:
    budget = dict(build_qwen3_tts_generation_budget(spoken_text))
    budget['tts_model_type'] = str(tts_model_type or '').strip().lower() or 'unknown'
    budget['generation_scope'] = generation_scope
    return budget


def _qwen3_tts_generation_scope(
    spoken_text: str,
    *,
    tts_model_type: str,
) -> str:
    model_type = str(tts_model_type or '').strip().lower()
    if model_type in {'voice_design', 'custom_voice'}:
        return 'single_sequence'
    if model_type == 'base':
        nonempty_line_count = len(
            [line for line in str(spoken_text or '').splitlines() if line.strip()]
        )
        return (
            'segmented_sequence'
            if nonempty_line_count > 1
            else 'single_sequence'
        )
    return 'unverified_sequence'


def _tts_audio_bytes_integrity_evidence(
    audio_bytes: Any,
    source_text: str,
    *,
    source_sha256: str,
    generation_budget: Mapping[str, Any],
    tts_model_type: str,
) -> dict[str, Any]:
    raw = bytes(audio_bytes or b'')
    if not raw:
        return {
            'kind': 'ollmo.tts_audio_integrity_evidence',
            'version': 1,
            'status': 'failed',
            'reason_code': 'TTS_AUDIO_CHUNK_BYTES_MISSING',
            'materialization_eligible': False,
            'source_sha256': source_sha256,
        }
    with tempfile.NamedTemporaryFile(suffix='.wav') as handle:
        handle.write(raw)
        handle.flush()
        evidence = build_tts_audio_integrity_evidence(
            handle.name,
            source_text,
            source_sha256=source_sha256,
            generation_budget=generation_budget,
            model_family='qwen3_tts',
            tts_model_type=tts_model_type,
            integrity_profile=TTS_QWEN_SENTENCE_CHUNK_INTEGRITY_PROFILE,
        )
    evidence = dict(evidence)
    evidence.pop('artifact_path', None)
    evidence['artifact_scope'] = 'ephemeral_backend_chunk'
    evidence['persisted'] = False
    return evidence


def _run_text_to_speech(ctx: InferContext, artifacts: InferArtifacts, ops: Dict[str, Callable[..., Any]]) -> Tuple[dict, int]:
    if ctx.backend != 'mlx':
        return {'error': "text_to_speech is currently supported only for MLX."}, 400
    if artifacts.file_kind and artifacts.file_kind not in {'text'}:
        return {'error': 'For text_to_speech, use only a text prompt or text file.'}, 400
    model_name_lower = str(ctx.model_name or '').lower()
    tts_model_type = str(ctx.tts_model_type or '').strip().lower()
    available_speakers = [str(speaker or '').strip() for speaker in (ctx.tts_speakers or []) if str(speaker or '').strip()]
    if not tts_model_type:
        if 'qwen3-tts' in model_name_lower and 'customvoice' in model_name_lower:
            tts_model_type = 'custom_voice'
        elif 'qwen3-tts' in model_name_lower and 'voicedesign' in model_name_lower:
            tts_model_type = 'voice_design'
        elif 'qwen3-tts' in model_name_lower and 'base' in model_name_lower:
            tts_model_type = 'base'
        elif 'kitten-tts' in model_name_lower:
            tts_model_type = 'kitten_tts'
    effective_voice = str(ctx.voice or '').strip()
    if tts_model_type == 'kitten_tts':
        if not effective_voice and available_speakers:
            effective_voice = available_speakers[0]
        if effective_voice and available_speakers and effective_voice.lower() not in {speaker.lower() for speaker in available_speakers}:
            speaker_list = ', '.join(available_speakers)
            return (
                {
                    'error': (
                        f"Speaker '{effective_voice}' is not supported by this Kitten-TTS model. "
                        f"Available: {speaker_list}."
                    )
                },
                400,
            )
        if not effective_voice:
            speaker_list = ', '.join(available_speakers) if available_speakers else 'no automatically detected speakers'
            return (
                {
                    'error': (
                        "This Kitten-TTS model requires a speaker, but Ollmo could not derive a valid "
                        f"default from the metadata. Available: {speaker_list}."
                    )
                },
                400,
            )
    if (
        tts_model_type == 'custom_voice'
        and not effective_voice
    ):
        speaker_list = ', '.join(QWEN3_CUSTOMVOICE_SPEAKERS)
        return (
            {
                'error': (
                    "This Qwen3-TTS-CustomVoice model requires a speaker from the list. "
                    f"Available: {speaker_list}. "
                    "If you want to describe the voice in natural language, use a VoiceDesign model."
                )
            },
            400,
        )
    if (
        effective_voice
        and tts_model_type == 'custom_voice'
        and effective_voice.lower() not in {speaker.lower() for speaker in QWEN3_CUSTOMVOICE_SPEAKERS}
    ):
        speaker_list = ', '.join(QWEN3_CUSTOMVOICE_SPEAKERS)
        return (
            {
                'error': (
                    f"Speaker '{effective_voice}' is not supported by this Qwen3-TTS-CustomVoice model. "
                    f"Available: {speaker_list}. "
                    "For free-form voice description in natural language, use a VoiceDesign model."
                )
            },
            400,
        )
    if tts_model_type == 'voice_design' and not str(ctx.instruct or '').strip():
        return (
            {
                'error': (
                    "This Qwen3-TTS-VoiceDesign model requires a description in the "
                    "'Style / Instruct' field. Use a CustomVoice model instead for fixed speakers."
                )
            },
            400,
        )

    prompt = (
        str(ctx.prompt or '').strip()
        if ctx.prompt_is_semantic_materializer_payload
        else extract_legacy_tts_wrapper_text(ctx.prompt)
    )
    if artifacts.text_from_file:
        prompt = f'{prompt}\n\n{artifacts.text_from_file}'.strip() if prompt else artifacts.text_from_file.strip()
    if not prompt:
        return {'error': 'text_to_speech requires a text prompt.'}, 400

    is_qwen3_tts = _is_qwen3_tts_model(ctx.model_name)
    effective_lang_code, lang_code_source = _resolve_effective_tts_lang_code(
        ctx.lang_code,
        prompt,
        ctx.tts_languages,
    )
    if is_qwen3_tts and effective_lang_code:
        effective_lang_code, lang_code_canonicalized = _canonicalize_qwen3_tts_lang_code(
            effective_lang_code,
        )
        if lang_code_canonicalized:
            lang_code_source = (
                f'{lang_code_source or "resolved"}_qwen3_alias_canonicalized'
            )
    if is_qwen3_tts and not effective_lang_code:
        effective_lang_code = 'auto'
        lang_code_source = 'qwen3_model_default'
    semantic_source = build_tts_semantic_source(
        prompt,
        source_text_source='inference_final_spoken_prompt',
        lang_code=effective_lang_code,
    )

    generation_budget = (
        _qwen3_tts_budget_with_scope(
            prompt,
            tts_model_type=tts_model_type,
            generation_scope=_qwen3_tts_generation_scope(
                prompt,
                tts_model_type=tts_model_type,
            ),
        )
        if is_qwen3_tts
        else None
    )
    sampling_profile = _build_qwen3_tts_sampling_profile() if is_qwen3_tts else None
    base_speech_kwargs = {
        'instruct': ctx.instruct,
        'voice': effective_voice or None,
        'response_format': ctx.response_format,
        'speed': ctx.speed,
        'pitch': ctx.pitch,
        'lang_code': effective_lang_code,
        'timeout_sec': ctx.infer_timeout_sec,
    }
    if sampling_profile:
        base_speech_kwargs.update(
            {
                'temperature': sampling_profile['temperature'],
                'top_p': sampling_profile['top_p'],
                'top_k': sampling_profile['top_k'],
                'repetition_penalty': sampling_profile['repetition_penalty'],
            }
        )

    resolved_response_format = str(ctx.response_format or 'wav').strip().lower()
    chunking_evidence: Optional[dict[str, Any]] = None
    single_sequence_recovery_evidence: Optional[dict[str, Any]] = None
    result: dict[str, Any]
    chunk_failure_reason = ''
    chunk_failure_bytes = b''
    chunk_failure_content_type = 'audio/wav'
    chunk_plan = (
        build_qwen3_tts_chunk_plan(prompt)
        if is_qwen3_tts
        and tts_model_type in {'base', 'voice_design', 'custom_voice'}
        and resolved_response_format in {'wav', 'wave', 'x-wav'}
        else None
    )
    if chunk_plan and chunk_plan.get('applied') is True:
        generation_budget = _qwen3_tts_budget_with_scope(
            prompt,
            tts_model_type=tts_model_type,
            generation_scope='chunked_sequence',
        )
        generation_budget['chunk_count'] = int(chunk_plan.get('chunk_count') or 0)
        chunk_diagnostics: list[dict[str, Any]] = []
        chunk_audio_bytes: list[bytes] = []
        passed_chunk_count = 0
        recovered_chunk_count = 0
        backend_call_count = 0
        recovery_attempt_count = 0

        def synthesize_chunk_attempt(
            chunk_text: str,
            *,
            source_sha256: str,
            attempt_index: int,
            attempt_role: str,
            attempt_budget: Mapping[str, Any],
        ) -> tuple[bytes, str, dict[str, Any], dict[str, Any]]:
            speech_kwargs = dict(base_speech_kwargs)
            speech_kwargs['max_tokens'] = int(
                attempt_budget.get('max_tokens') or 0
            )
            attempt_result = ops['mlx_audio_speech'](
                ctx.port,
                ctx.model_name,
                chunk_text,
                **speech_kwargs,
            )
            attempt_audio = bytes(attempt_result.get('audio_bytes') or b'')
            attempt_content_type = str(
                attempt_result.get('content_type') or 'audio/wav'
            )
            attempt_integrity = _tts_audio_bytes_integrity_evidence(
                attempt_audio,
                chunk_text,
                source_sha256=source_sha256,
                generation_budget=attempt_budget,
                tts_model_type=tts_model_type,
            )
            attempt_record = {
                key: value
                for key, value in {
                    'attempt_index': attempt_index,
                    'role': attempt_role,
                    'generation_budget': dict(attempt_budget),
                    'audio_size_bytes': len(attempt_audio),
                    'audio_sha256': (
                        hashlib.sha256(attempt_audio).hexdigest()
                        if attempt_audio
                        else None
                    ),
                    'content_type': attempt_content_type,
                    'integrity_evidence': attempt_integrity,
                    'selected': False,
                }.items()
                if value not in (None, '', [], {})
            }
            return (
                attempt_audio,
                attempt_content_type,
                attempt_integrity,
                attempt_record,
            )

        for raw_chunk in chunk_plan.get('chunks') or []:
            chunk_text = str(raw_chunk.get('text') or '')
            chunk_source_sha256 = str(raw_chunk.get('text_sha256') or '').strip()
            chunk_budget = _qwen3_tts_budget_with_scope(
                chunk_text,
                tts_model_type=tts_model_type,
                generation_scope=_qwen3_tts_generation_scope(
                    chunk_text,
                    tts_model_type=tts_model_type,
                ),
            )
            (
                raw_audio,
                chunk_content_type,
                chunk_integrity,
                initial_attempt,
            ) = synthesize_chunk_attempt(
                chunk_text,
                source_sha256=chunk_source_sha256,
                attempt_index=1,
                attempt_role='initial',
                attempt_budget=chunk_budget,
            )
            backend_call_count += 1
            attempts = [initial_attempt]
            terminal_budget = chunk_budget
            generation_limit_recovery: Optional[dict[str, Any]] = None
            chunk_passed = bool(
                str(chunk_integrity.get('status') or '').strip().lower()
                == 'passed'
                and chunk_integrity.get('materialization_eligible') is True
            )
            if (
                not chunk_passed
                and tts_audio_has_qwen_generation_limit_exhaustion(
                    chunk_integrity,
                    generation_budget=chunk_budget,
                )
            ):
                generation_limit_recovery = (
                    _build_qwen3_tts_generation_limit_recovery(
                        chunk_budget,
                        integrity_evidence=chunk_integrity,
                    )
                )
                if generation_limit_recovery.get('applied') is True:
                    retry_budget = generation_limit_recovery.get(
                        'generation_budget'
                    )
                    if isinstance(retry_budget, Mapping):
                        (
                            raw_audio,
                            chunk_content_type,
                            chunk_integrity,
                            retry_attempt,
                        ) = synthesize_chunk_attempt(
                            chunk_text,
                            source_sha256=chunk_source_sha256,
                            attempt_index=2,
                            attempt_role='generation_limit_recovery',
                            attempt_budget=retry_budget,
                        )
                        backend_call_count += 1
                        recovery_attempt_count += 1
                        terminal_budget = dict(retry_budget)
                        attempts.append(retry_attempt)
                        chunk_passed = bool(
                            str(
                                chunk_integrity.get('status') or ''
                            ).strip().lower()
                            == 'passed'
                            and chunk_integrity.get(
                                'materialization_eligible'
                            )
                            is True
                        )
                        generation_limit_recovery = {
                            **generation_limit_recovery,
                            'status': 'passed' if chunk_passed else 'failed',
                            'retry_count': 1,
                            'selected_attempt_index': 2,
                            'terminal_reason_code': str(
                                chunk_integrity.get('reason_code')
                                or 'TTS_AUDIO_CHUNK_INTEGRITY_FAILED'
                            ).strip(),
                        }
            attempts[-1]['selected'] = True
            chunk_record = {
                key: value
                for key, value in {
                    'index': raw_chunk.get('index'),
                    'source_span_start': raw_chunk.get('source_span_start'),
                    'source_span_end': raw_chunk.get('source_span_end'),
                    'text_sha256': chunk_source_sha256,
                    'estimated_speech_seconds': raw_chunk.get(
                        'estimated_speech_seconds'
                    ),
                    'status': (
                        'recovered'
                        if chunk_passed and len(attempts) > 1
                        else 'passed'
                        if chunk_passed
                        else 'failed'
                    ),
                    'generation_budget': terminal_budget,
                    'audio_size_bytes': len(raw_audio),
                    'audio_sha256': (
                        hashlib.sha256(raw_audio).hexdigest()
                        if raw_audio
                        else None
                    ),
                    'content_type': chunk_content_type,
                    'integrity_evidence': chunk_integrity,
                    'attempt_count': len(attempts),
                    'recovery_attempt_count': max(0, len(attempts) - 1),
                    'attempts': attempts,
                    'generation_limit_recovery': generation_limit_recovery,
                }.items()
                if value not in (None, '', [], {})
            }
            chunk_diagnostics.append(chunk_record)
            if not chunk_passed:
                chunk_failure_reason = str(
                    chunk_integrity.get('reason_code')
                    or 'TTS_AUDIO_CHUNK_INTEGRITY_FAILED'
                ).strip()
                chunk_failure_bytes = raw_audio
                chunk_failure_content_type = chunk_content_type
                break
            passed_chunk_count += 1
            if len(attempts) > 1:
                recovered_chunk_count += 1
            chunk_audio_bytes.append(raw_audio)

        join_evidence: dict[str, Any] = {}
        joined_audio = b''
        if not chunk_failure_reason:
            try:
                joined_audio, join_evidence = join_pcm_wav_bytes(chunk_audio_bytes)
            except ValueError as exc:
                chunk_failure_reason = 'TTS_AUDIO_CHUNK_JOIN_FAILED'
                chunk_failure_bytes = chunk_audio_bytes[0] if chunk_audio_bytes else b''
                chunk_failure_content_type = 'audio/wav'
                join_evidence = {
                    'kind': 'ollmo.pcm_wav_join',
                    'version': 1,
                    'status': 'failed',
                    'reason_code': chunk_failure_reason,
                    'error_type': type(exc).__name__,
                }
        chunking_evidence = {
            key: value
            for key, value in {
                **{
                    key: value
                    for key, value in chunk_plan.items()
                    if key != 'chunks'
                },
                'status': 'failed' if chunk_failure_reason else 'passed',
                'chunks': chunk_diagnostics,
                'completed_chunk_count': len(chunk_diagnostics),
                'attempted_chunk_count': len(chunk_diagnostics),
                'passed_chunk_count': passed_chunk_count,
                'backend_call_count': backend_call_count,
                'generation_limit_recovery_attempt_count': (
                    recovery_attempt_count
                ),
                'recovered_chunk_count': recovered_chunk_count,
                'failed_chunk_index': (
                    chunk_diagnostics[-1].get('index')
                    if chunk_failure_reason and chunk_diagnostics
                    else None
                ),
                'failure_reason_code': chunk_failure_reason or None,
                'join_evidence': join_evidence or None,
                'joined_audio_size_bytes': len(joined_audio) if joined_audio else None,
                'joined_audio_sha256': (
                    hashlib.sha256(joined_audio).hexdigest()
                    if joined_audio
                    else None
                ),
            }.items()
            if value not in (None, '', [], {})
        }
        if chunk_failure_reason:
            result = {
                'audio_bytes': chunk_failure_bytes,
                'content_type': chunk_failure_content_type,
                'result': {
                    'bytes': len(chunk_failure_bytes),
                    'diagnostic_only': True,
                    'reason_code': chunk_failure_reason,
                },
            }
        else:
            result = {
                'audio_bytes': joined_audio,
                'content_type': 'audio/wav',
                'result': {
                    'bytes': len(joined_audio),
                    'chunk_count': len(chunk_audio_bytes),
                    'joined_pcm_wav': True,
                },
            }
    else:
        speech_kwargs = dict(base_speech_kwargs)
        if generation_budget:
            speech_kwargs['max_tokens'] = generation_budget['max_tokens']
        result = ops['mlx_audio_speech'](
            ctx.port,
            ctx.model_name,
            prompt,
            **speech_kwargs,
        )
        initial_audio_candidate = bytes(result.get('audio_bytes') or b'')
        if (
            is_qwen3_tts
            and tts_model_type
            in _QWEN3_TTS_GENERATION_LIMIT_RECOVERY_MODEL_TYPES
            and resolved_response_format in {'wav', 'wave', 'x-wav'}
            and isinstance(generation_budget, Mapping)
            and str(
                generation_budget.get('generation_scope') or ''
            ).strip().lower()
            == 'single_sequence'
            and len(initial_audio_candidate) >= 12
            and initial_audio_candidate[:4] == b'RIFF'
            and initial_audio_candidate[8:12] == b'WAVE'
        ):
            initial_audio = initial_audio_candidate
            initial_content_type = str(
                result.get('content_type') or 'audio/wav'
            )
            initial_integrity = _tts_audio_bytes_integrity_evidence(
                initial_audio,
                prompt,
                source_sha256=str(
                    semantic_source.get('tts_source_text_sha256') or ''
                ).strip(),
                generation_budget=generation_budget,
                tts_model_type=tts_model_type,
            )
            if tts_audio_has_qwen_generation_limit_exhaustion(
                initial_integrity,
                generation_budget=generation_budget,
            ):
                recovery = _build_qwen3_tts_generation_limit_recovery(
                    generation_budget,
                    integrity_evidence=initial_integrity,
                )
                retry_budget = recovery.get('generation_budget')
                if (
                    recovery.get('applied') is True
                    and isinstance(retry_budget, Mapping)
                ):
                    retry_kwargs = dict(base_speech_kwargs)
                    retry_kwargs['max_tokens'] = int(
                        retry_budget.get('max_tokens') or 0
                    )
                    retry_result = ops['mlx_audio_speech'](
                        ctx.port,
                        ctx.model_name,
                        prompt,
                        **retry_kwargs,
                    )
                    retry_audio = bytes(
                        retry_result.get('audio_bytes') or b''
                    )
                    retry_content_type = str(
                        retry_result.get('content_type') or 'audio/wav'
                    )
                    retry_integrity = _tts_audio_bytes_integrity_evidence(
                        retry_audio,
                        prompt,
                        source_sha256=str(
                            semantic_source.get(
                                'tts_source_text_sha256'
                            )
                            or ''
                        ).strip(),
                        generation_budget=retry_budget,
                        tts_model_type=tts_model_type,
                    )
                    retry_passed = bool(
                        str(
                            retry_integrity.get('status') or ''
                        ).strip().lower()
                        == 'passed'
                        and retry_integrity.get(
                            'materialization_eligible'
                        )
                        is True
                    )
                    initial_attempt = {
                        'attempt_index': 1,
                        'role': 'initial',
                        'generation_budget': dict(generation_budget),
                        'audio_size_bytes': len(initial_audio),
                        'audio_sha256': (
                            hashlib.sha256(initial_audio).hexdigest()
                            if initial_audio
                            else None
                        ),
                        'content_type': initial_content_type,
                        'integrity_evidence': initial_integrity,
                        'selected': False,
                    }
                    retry_attempt = {
                        'attempt_index': 2,
                        'role': 'generation_limit_recovery',
                        'generation_budget': dict(retry_budget),
                        'audio_size_bytes': len(retry_audio),
                        'audio_sha256': (
                            hashlib.sha256(retry_audio).hexdigest()
                            if retry_audio
                            else None
                        ),
                        'content_type': retry_content_type,
                        'integrity_evidence': retry_integrity,
                        'selected': True,
                    }
                    single_sequence_recovery_evidence = {
                        **recovery,
                        'status': 'passed' if retry_passed else 'failed',
                        'retry_count': 1,
                        'attempt_count': 2,
                        'selected_attempt_index': 2,
                        'terminal_reason_code': str(
                            retry_integrity.get('reason_code')
                            or 'TTS_AUDIO_INTEGRITY_UNAVAILABLE'
                        ).strip(),
                        'attempts': [initial_attempt, retry_attempt],
                    }
                    generation_budget = dict(retry_budget)
                    result = dict(retry_result)

    saved_audio_path = ops['persist_audio_bytes_locally'](
        result.get('audio_bytes'),
        ctx.model_name,
        response_format=ctx.response_format,
        content_type=result.get('content_type'),
    )
    if not saved_audio_path:
        return {'error': 'TTS audio could not be saved locally.'}, 500

    integrity_evidence = build_tts_audio_integrity_evidence(
        saved_audio_path,
        prompt,
        source_sha256=semantic_source.get('tts_source_text_sha256'),
        generation_budget=generation_budget,
        model_family='qwen3_tts' if is_qwen3_tts else None,
        tts_model_type=tts_model_type or None,
    )
    if chunking_evidence:
        integrity_evidence = dict(integrity_evidence)
        integrity_evidence['chunking_evidence'] = chunking_evidence
    if single_sequence_recovery_evidence:
        integrity_evidence = dict(integrity_evidence)
        integrity_evidence['generation_limit_recovery'] = (
            single_sequence_recovery_evidence
        )
    if chunk_failure_reason:
        integrity_evidence = dict(integrity_evidence)
        defect_codes = [
            str(item).strip()
            for item in (integrity_evidence.get('defect_codes') or [])
            if str(item).strip()
        ]
        if chunk_failure_reason not in defect_codes:
            defect_codes.append(chunk_failure_reason)
        integrity_evidence.update(
            {
                'status': 'failed',
                'reason_code': chunk_failure_reason,
                'materialization_eligible': False,
                'defect_codes': defect_codes,
                'diagnostic_scope': 'failed_qwen_chunk',
            }
        )

    payload = {
        'instance_id': ctx.instance_id,
        'capability': ctx.capability,
        'mode': 'text_to_speech',
        'content': 'Audio generated.',
        'saved_audio_path': saved_audio_path,
        'audio_mimetype': result.get('content_type'),
        'result': result.get('result'),
        'tts_semantic_source': semantic_source,
        'tts_audio_integrity_evidence': integrity_evidence,
    }
    if effective_lang_code:
        payload['lang_code'] = effective_lang_code
    if lang_code_source:
        payload['lang_code_source'] = lang_code_source
    if effective_voice:
        payload['voice'] = effective_voice
    if ctx.instruct:
        payload['instruct'] = ctx.instruct
    if ctx.response_format:
        payload['response_format'] = ctx.response_format
    if generation_budget:
        payload['tts_generation_budget'] = generation_budget
    if tts_model_type:
        payload['tts_model_type'] = tts_model_type
    if sampling_profile:
        payload['tts_sampling_profile'] = sampling_profile
    return payload, 200


def _run_image_generation(
    ctx: InferContext,
    artifacts: InferArtifacts,
    ops: Dict[str, Callable[..., Any]],
) -> Tuple[dict, int]:
    if not ctx.prompt:
        return {'error': 'image_generation requires a text prompt.'}, 400
    has_reference_image = bool(artifacts.image_b64)
    data_out: dict[str, Any] = {}
    saved_image_path = None
    image_data_url = None

    if not has_reference_image:
        image_data_url = ops['ollama_openai_image_generation'](
            ctx.port,
            ctx.model_name,
            ctx.prompt,
            width=ctx.image_width,
            height=ctx.image_height,
        )
    if not image_data_url:
        data_out = ops['ollama_generate'](
            ctx.port,
            ctx.model_name,
            ctx.prompt,
            images=[artifacts.image_b64] if artifacts.image_b64 else None,
            timeout_sec=ctx.infer_timeout_sec,
            options={
                key: value
                for key, value in (
                    ('width', ctx.image_width),
                    ('height', ctx.image_height),
                    ('seed', ctx.image_seed),
                )
                if value is not None
            } or None,
            allow_port_fallback=False,
        )
        saved_image_path = ops['extract_saved_image_path_from_generate_output'](data_out)
        image_data_url = ops['extract_image_data_url_from_generate_output'](data_out)
    if image_data_url and not saved_image_path:
        saved_image_path = ops['persist_image_data_url_locally'](image_data_url, ctx.model_name)
    content = ops['extract_generate_content'](data_out)
    image_seed = None
    extract_seed = ops.get('extract_generate_seed')
    if callable(extract_seed):
        image_seed = extract_seed(data_out)
    if image_seed is None:
        image_seed = ctx.image_seed
    if not content:
        if image_data_url:
            content = 'Image generated.'
        else:
            content = (
                'Image request completed, but no inline image payload was returned. '
                'Check Ollama server logs/CLI output for saved image path.'
            )
    return (
        {
            'instance_id': ctx.instance_id,
            'capability': ctx.capability,
            'mode': 'image_generation_edit' if has_reference_image else 'image_generation',
            'content': content,
            'image_data_url': image_data_url,
            'saved_image_path': saved_image_path,
            'seed': image_seed,
            'reference_image_count': 1 if has_reference_image else 0,
            'reference_image_kind': artifacts.file_kind if has_reference_image else None,
            'result': data_out,
        },
        200,
    )


def _run_vision_analysis(
    ctx: InferContext,
    artifacts: InferArtifacts,
    ops: Dict[str, Callable[..., Any]],
) -> Tuple[dict, int]:
    if artifacts.file_kind == 'pdf':
        return _run_pdf_vision_analysis(ctx, artifacts, ops)

    if not artifacts.image_b64 and not ctx.prompt:
        return {'error': 'For vision_analysis, provide an image file, PDF, or prompt.'}, 400

    effective_prompt = _effective_vision_prompt(ctx)
    if ctx.backend in {'mlx', 'llama_cpp'}:
        mlx_out = _run_backend_chat_completion(
            ctx,
            ops,
            [_build_mlx_multimodal_user_message(effective_prompt, artifacts.image_b64)],
            timeout_sec=ctx.infer_timeout_sec,
        )
        return (
            {
                'instance_id': ctx.instance_id,
                'capability': ctx.capability,
                'mode': 'vision_analysis',
                'content': mlx_out.get('content', ''),
                'result': mlx_out.get('result'),
            },
            200,
        )

    model_name_lower = str(ctx.model_name or '').lower()
    if artifacts.image_b64 and 'deepseek-ocr' in model_name_lower:
        ocr_content, ocr_error = ops['ocr_image_with_deepseek'](
            port=ctx.port,
            model_name=ctx.model_name,
            image_b64=artifacts.image_b64,
            user_prompt=ctx.prompt,
            timeout_sec=ctx.infer_timeout_sec,
        )
        if ocr_content:
            final_content = ocr_content
            if ctx.prompt and not ops['is_generic_ocr_instruction_prompt'](ctx.prompt):
                try:
                    refine_prompt = (
                        'Apply the user instruction to the OCR result below.\n'
                        'Keep facts faithful to OCR output.\n\n'
                        f'User instruction:\n{ctx.prompt}\n\n'
                        f'[OCR markdown]\n{ocr_content}'
                    )
                    refine_out = ops['ollama_generate'](
                        ctx.port,
                        ctx.model_name,
                        refine_prompt,
                        timeout_sec=min(ctx.infer_timeout_sec, 900),
                        max_retries=1,
                        allow_port_fallback=False,
                    )
                    refined = ops['clean_ocr_output_text'](ops['extract_generate_content'](refine_out))
                    if refined and not ops['looks_like_ocr_prompt_echo'](refined, user_hint=ctx.prompt):
                        final_content = refined
                except Exception as exc:  # noqa: BLE001
                    logging.info('Image OCR refinement skipped: %s', exc)

            payload = {
                'instance_id': ctx.instance_id,
                'capability': ctx.capability,
                'mode': 'vision_analysis_ocr_image',
                'content': final_content,
            }
            if ocr_error:
                payload['warnings'] = [ocr_error]
            return payload, 200
        if ocr_error:
            logging.warning('DeepSeek image OCR fallback to generic mode: %s', ocr_error)

    data_out = ops['ollama_generate'](
        ctx.port,
        ctx.model_name,
        effective_prompt,
        images=[artifacts.image_b64] if artifacts.image_b64 else None,
        timeout_sec=ctx.infer_timeout_sec,
    )
    return (
        {
            'instance_id': ctx.instance_id,
            'capability': ctx.capability,
            'mode': 'vision_analysis',
            'content': ops['extract_generate_content'](data_out),
            'result': data_out,
        },
        200,
    )


def _run_pdf_vision_analysis(
    ctx: InferContext,
    artifacts: InferArtifacts,
    ops: Dict[str, Callable[..., Any]],
) -> Tuple[dict, int]:
    effective_prompt = _effective_vision_prompt(ctx)
    if ctx.backend in {'mlx', 'llama_cpp'}:
        if artifacts.text_from_file:
            mlx_out = _run_backend_chat_completion(
                ctx,
                ops,
                [
                    {
                        'role': 'user',
                        'content': (
                            f'{effective_prompt}\n\n'
                            '[PDF extracted text]\n'
                            f'{artifacts.text_from_file}'
                        ),
                    }
                ],
                timeout_sec=ctx.infer_timeout_sec,
            )
            content_text = mlx_out.get('content', '')
            saved_source_text_path = ops['persist_text_markdown_locally'](
                artifacts.text_from_file,
                model_name=ctx.model_name,
                source_file_name=artifacts.file_name,
                mode='vision_analysis_pdf_text_source',
            )
            saved_text_path = ops['persist_text_markdown_locally'](
                content_text,
                model_name=ctx.model_name,
                source_file_name=artifacts.file_name,
                mode='vision_analysis_pdf_text',
            )
            response_content, content_truncated = _build_pdf_inline_response_content(
                content_text,
                warnings=artifacts.pdf_warnings,
                max_inline_chars=ops['max_pdf_inline_response_chars'],
            )
            payload = {
                'instance_id': ctx.instance_id,
                'capability': ctx.capability,
                'mode': 'vision_analysis_pdf_text',
                'content': response_content,
                'pdf_source': 'text_layer',
                'pdf_total_pages': artifacts.pdf_total_pages or None,
                'pdf_processed_pages': None,
                'warnings': artifacts.pdf_warnings,
                'saved_text_path': saved_text_path,
                'saved_source_text_path': saved_source_text_path,
                'content_truncated': content_truncated,
                'full_content_chars': len(content_text),
                'inline_content_chars': len(response_content),
                'result': mlx_out.get('result'),
            }
            ops['log_pdf_infer_event'](
                instance_id=ctx.instance_id,
                model_name=ctx.model_name,
                backend=ctx.backend,
                capability=ctx.capability,
                prompt=ctx.user_prompt,
                file_name=artifacts.file_name,
                file_sha256=artifacts.file_sha256,
                status='ok',
                mode=payload['mode'],
                content=payload['content'],
                warnings=artifacts.pdf_warnings,
                pdf_source='text_layer',
                pdf_total_pages=artifacts.pdf_total_pages or None,
                pdf_processed_pages=None,
                artifact_path=saved_text_path,
            )
            return payload, 200

        page_results = []
        for idx, page_image_b64 in enumerate(artifacts.pdf_page_images, start=1):
            mlx_out = ops['mlx_chat_completions'](
                ctx.port,
                ctx.model_name,
                [
                    _build_mlx_multimodal_user_message(
                        f'{effective_prompt}\n\nPage {idx} of {len(artifacts.pdf_page_images)}.',
                        page_image_b64,
                    )
                ],
                timeout_sec=ctx.pdf_page_timeout_sec,
            )
            content = str(mlx_out.get('content') or '').strip()
            if content:
                page_results.append(f"[Page {idx}]\n{content}")

        if not page_results:
            return (
                {
                    'error': 'The PDF was processed, but the MLX VLM model returned no text.',
                    'warnings': artifacts.pdf_warnings,
                },
                502,
            )

        final_content = '\n\n---\n\n'.join(page_results)
        saved_text_path = ops['persist_text_markdown_locally'](
            final_content,
            model_name=ctx.model_name,
            source_file_name=artifacts.file_name,
            mode='vision_analysis_pdf_scan',
        )
        response_content, content_truncated = _build_pdf_inline_response_content(
            final_content,
            warnings=artifacts.pdf_warnings,
            max_inline_chars=ops['max_pdf_inline_response_chars'],
        )
        payload = {
            'instance_id': ctx.instance_id,
            'capability': ctx.capability,
            'mode': 'vision_analysis_pdf_scan',
            'content': response_content,
            'pdf_source': 'rendered_pages',
            'pdf_total_pages': artifacts.pdf_total_pages or len(artifacts.pdf_page_images),
            'pdf_processed_pages': len(page_results),
            'warnings': artifacts.pdf_warnings,
            'saved_text_path': saved_text_path,
            'content_truncated': content_truncated,
            'full_content_chars': len(final_content),
            'inline_content_chars': len(response_content),
        }
        ops['log_pdf_infer_event'](
            instance_id=ctx.instance_id,
            model_name=ctx.model_name,
            backend=ctx.backend,
            capability=ctx.capability,
            prompt=ctx.user_prompt,
            file_name=artifacts.file_name,
            file_sha256=artifacts.file_sha256,
            status='ok',
            mode=payload['mode'],
            content=final_content,
            warnings=artifacts.pdf_warnings,
            pdf_source='rendered_pages',
            pdf_total_pages=payload['pdf_total_pages'],
            pdf_processed_pages=payload['pdf_processed_pages'],
            artifact_path=saved_text_path,
        )
        return payload, 200
    if artifacts.text_from_file:
        pdf_prompt_payload = (
            f'{effective_prompt}\n\n'
            '[PDF extracted text]\n'
            f'{artifacts.text_from_file}'
        )
        data_out = ops['ollama_generate'](
            ctx.port,
            ctx.model_name,
            pdf_prompt_payload,
            timeout_sec=ctx.infer_timeout_sec,
        )
        content_text = ops['extract_generate_content'](data_out)
        saved_source_text_path = ops['persist_text_markdown_locally'](
            artifacts.text_from_file,
            model_name=ctx.model_name,
            source_file_name=artifacts.file_name,
            mode='vision_analysis_pdf_text_source',
        )
        saved_text_path = ops['persist_text_markdown_locally'](
            content_text,
            model_name=ctx.model_name,
            source_file_name=artifacts.file_name,
            mode='vision_analysis_pdf_text',
        )
        response_content, content_truncated = _build_pdf_inline_response_content(
            content_text,
            warnings=artifacts.pdf_warnings,
            max_inline_chars=ops['max_pdf_inline_response_chars'],
        )
        payload = {
            'instance_id': ctx.instance_id,
            'capability': ctx.capability,
            'mode': 'vision_analysis_pdf_text',
            'content': response_content,
            'pdf_source': 'text_layer',
            'pdf_total_pages': artifacts.pdf_total_pages or None,
            'pdf_processed_pages': None,
            'warnings': artifacts.pdf_warnings,
            'saved_text_path': saved_text_path,
            'saved_source_text_path': saved_source_text_path,
            'content_truncated': content_truncated,
            'full_content_chars': len(content_text),
            'inline_content_chars': len(response_content),
            'result': data_out,
        }
        ops['log_pdf_infer_event'](
            instance_id=ctx.instance_id,
            model_name=ctx.model_name,
            backend=ctx.backend,
            capability=ctx.capability,
            prompt=ctx.user_prompt,
            file_name=artifacts.file_name,
            file_sha256=artifacts.file_sha256,
            status='ok',
            mode=payload['mode'],
            content=payload['content'],
            warnings=artifacts.pdf_warnings,
            pdf_source='text_layer',
            pdf_total_pages=artifacts.pdf_total_pages or None,
            pdf_processed_pages=None,
            artifact_path=saved_text_path,
        )
        return payload, 200

    page_results_by_number: dict[int, str] = {}
    page_errors: dict[int, str] = {}
    total_for_prompt = len(artifacts.pdf_page_images)
    for idx, page_image_b64 in enumerate(artifacts.pdf_page_images, start=1):
        page_content, page_error = ops['ocr_pdf_page_with_ollama'](
            port=ctx.port,
            model_name=ctx.model_name,
            base_prompt=effective_prompt,
            page_index=idx,
            total_pages=total_for_prompt,
            image_b64=page_image_b64,
            timeout_sec=ctx.pdf_page_timeout_sec,
        )
        if page_content:
            page_results_by_number[idx] = page_content
        elif page_error:
            page_errors[idx] = page_error

    crop_retry_margin_ratio = 0.04
    if page_errors and artifacts.temp_path and artifacts.pdf_render_dpi >= 300:
        failed_pages = sorted(page_errors.keys())
        logging.info(
            'Retrying PDF OCR for failed pages=%s with border-crop margin=%.2f (dpi=%s)',
            failed_pages,
            crop_retry_margin_ratio,
            artifacts.pdf_render_dpi,
        )
        crop_recovered_count = 0
        for page_number in list(failed_pages):
            retry_b64 = ops['render_single_pdf_page_to_base64'](
                artifacts.temp_path,
                page_index=page_number - 1,
                dpi=artifacts.pdf_render_dpi,
                max_image_side_px=ctx.pdf_max_image_side,
                crop_margin_ratio=crop_retry_margin_ratio,
            )
            if not retry_b64:
                continue
            page_content, page_error = ops['ocr_pdf_page_with_ollama'](
                port=ctx.port,
                model_name=ctx.model_name,
                base_prompt=effective_prompt,
                page_index=page_number,
                total_pages=total_for_prompt,
                image_b64=retry_b64,
                timeout_sec=ctx.pdf_page_timeout_sec,
            )
            if page_content:
                page_results_by_number[page_number] = page_content
                page_errors.pop(page_number, None)
                crop_recovered_count += 1
            elif page_error:
                page_errors[page_number] = f"{page_errors[page_number]} Crop retry failed: {page_error}"
        if crop_recovered_count:
            artifacts.pdf_warnings.append(
                f"{crop_recovered_count} page(s) were successfully recovered with border-crop retry ({int(crop_retry_margin_ratio * 100)}%)."
            )

    if page_errors and artifacts.temp_path and artifacts.pdf_page_retry_dpi < artifacts.pdf_render_dpi:
        failed_pages = sorted(page_errors.keys())
        logging.info(
            'Retrying PDF OCR for failed pages=%s with lower dpi=%s (initial_dpi=%s)',
            failed_pages,
            artifacts.pdf_page_retry_dpi,
            artifacts.pdf_render_dpi,
        )
        recovered_count = 0
        for page_number in list(failed_pages):
            retry_b64 = ops['render_single_pdf_page_to_base64'](
                artifacts.temp_path,
                page_index=page_number - 1,
                dpi=artifacts.pdf_page_retry_dpi,
                max_image_side_px=ctx.pdf_max_image_side,
                crop_margin_ratio=(crop_retry_margin_ratio if artifacts.pdf_render_dpi >= 300 else 0.0),
            )
            if not retry_b64:
                page_errors[page_number] = (
                    f"{page_errors[page_number]} Low-DPI rerender failed."
                )
                continue
            page_content, page_error = ops['ocr_pdf_page_with_ollama'](
                port=ctx.port,
                model_name=ctx.model_name,
                base_prompt=effective_prompt,
                page_index=page_number,
                total_pages=total_for_prompt,
                image_b64=retry_b64,
                timeout_sec=ctx.pdf_page_timeout_sec,
            )
            if page_content:
                page_results_by_number[page_number] = page_content
                page_errors.pop(page_number, None)
                recovered_count += 1
            elif page_error:
                page_errors[page_number] = f"{page_errors[page_number]} Retry failed: {page_error}"
        if recovered_count:
            artifacts.pdf_warnings.append(
                f"{recovered_count} page(s) were successfully recovered with reduced DPI ({artifacts.pdf_page_retry_dpi})."
            )

    if page_errors:
        ordered_errors = [page_errors[page_no] for page_no in sorted(page_errors.keys())]
        preview_errors = ordered_errors[:8]
        artifacts.pdf_warnings.extend(preview_errors)
        if len(ordered_errors) > len(preview_errors):
            artifacts.pdf_warnings.append(
                f"Additional page errors: {len(ordered_errors) - len(preview_errors)}"
            )

    page_results: list[str] = []
    for page_number in sorted(page_results_by_number.keys()):
        page_results.append(f"[Page {page_number}]\n{page_results_by_number[page_number]}")
    logging.info(
        'PDF OCR page loop completed: ok=%s failed=%s total=%s',
        len(page_results_by_number),
        len(page_errors),
        total_for_prompt,
    )

    if not page_results:
        all_timeout_like = bool(page_errors) and all(
            ('timeout' in message.lower()) or ('timed out' in message.lower())
            for message in page_errors.values()
        )
        error_message = 'The PDF was processed, but no OCR text was returned.'
        if page_errors:
            top_errors = '; '.join([page_errors[i] for i in sorted(page_errors.keys())[:3]])
            error_message = f'{error_message} Errors: {top_errors}'
        ops['log_pdf_infer_event'](
            instance_id=ctx.instance_id,
            model_name=ctx.model_name,
            backend=ctx.backend,
            capability=ctx.capability,
            prompt=ctx.user_prompt,
            file_name=artifacts.file_name,
            file_sha256=artifacts.file_sha256,
            status='error',
            mode='vision_analysis_pdf_scan',
            error=error_message,
            warnings=artifacts.pdf_warnings,
            pdf_source='rendered_pages',
            pdf_total_pages=artifacts.pdf_total_pages or len(artifacts.pdf_page_images),
            pdf_processed_pages=0,
        )
        return (
            {
                'error': error_message,
                'warnings': artifacts.pdf_warnings,
            },
            504 if all_timeout_like else 502,
        )

    joined_pages = '\n\n---\n\n'.join(page_results)
    final_content = joined_pages
    if ctx.pdf_synthesize and ctx.prompt and len(page_results) > 1:
        max_synthesis_pages = 12
        if len(page_results) > max_synthesis_pages:
            artifacts.pdf_warnings.append(
                f'Global synthesis skipped automatically for large batches ({len(page_results)} pages > {max_synthesis_pages}) '
                'to avoid long hangs/timeouts. Process in smaller chunks if you need one merged final narrative.'
            )
        else:
            synthesis_prompt = (
                f'User request:\n{ctx.prompt}\n\n'
                'Below are page-wise OCR/context notes from a PDF. '
                'Provide one consolidated final answer that covers the full document.\n\n'
                f'{joined_pages}'
            )
            synthesis_timeout_sec = max(120, min(int(ctx.infer_timeout_sec), 600))
            logging.info(
                'PDF synthesis request: pages=%s timeout_sec=%s prompt_chars=%s source_chars=%s',
                len(page_results),
                synthesis_timeout_sec,
                len(ctx.prompt),
                len(joined_pages),
            )
            try:
                synthesis_out = ops['ollama_generate'](
                    ctx.port,
                    ctx.model_name,
                    synthesis_prompt,
                    timeout_sec=synthesis_timeout_sec,
                )
                synthesized_content = ops['extract_generate_content'](synthesis_out).strip()
                if synthesized_content:
                    final_content = synthesized_content
                    logging.info('PDF synthesis completed: output_chars=%s', len(synthesized_content))
                else:
                    artifacts.pdf_warnings.append(
                        'Global synthesis returned empty output; keeping page-wise OCR result.'
                    )
            except ops['request_timeout_error']:
                artifacts.pdf_warnings.append(
                    f'Global synthesis timed out after {synthesis_timeout_sec}s; keeping page-wise OCR result.'
                )
            except ops['request_connection_error'] as exc:
                artifacts.pdf_warnings.append(
                    f'Global synthesis connection lost ({exc}); keeping page-wise OCR result.'
                )
            except ops['request_exception_error'] as exc:
                details = str(exc)
                if getattr(exc, 'response', None) is not None:
                    try:
                        payload = exc.response.json()
                        details = payload.get('error') or payload.get('message') or details
                    except Exception:  # noqa: BLE001
                        details = getattr(exc.response, 'text', details)[:260]
                artifacts.pdf_warnings.append(
                    f'Global synthesis failed ({details}); keeping page-wise OCR result.'
                )
            except Exception as exc:  # noqa: BLE001
                artifacts.pdf_warnings.append(
                    f'Global synthesis unexpected error ({exc}); keeping page-wise OCR result.'
                )
    else:
        if ctx.prompt and len(page_results) > 1:
            artifacts.pdf_warnings.append(
                'Global synthesis was skipped by default to avoid long timeouts. '
                'Set pdf_synthesize=true if you want a single merged answer in one pass.'
            )

    response_content, content_truncated = _build_pdf_inline_response_content(
        final_content,
        warnings=artifacts.pdf_warnings,
        max_inline_chars=ops['max_pdf_inline_response_chars'],
    )

    logging.info(
        'PDF OCR finalize begin: pages_ok=%s pages_failed=%s output_chars_full=%s output_chars_ui=%s warnings=%s',
        len(page_results_by_number),
        len(page_errors),
        len(final_content),
        len(response_content),
        len(artifacts.pdf_warnings),
    )

    payload = {
        'instance_id': ctx.instance_id,
        'capability': ctx.capability,
        'mode': 'vision_analysis_pdf_scan',
        'content': response_content,
        'pdf_source': 'rendered_pages',
        'pdf_total_pages': artifacts.pdf_total_pages or len(artifacts.pdf_page_images),
        'pdf_processed_pages': len(page_results_by_number),
        'warnings': artifacts.pdf_warnings,
        'content_truncated': content_truncated,
        'full_content_chars': len(final_content),
        'inline_content_chars': len(response_content),
    }
    payload['saved_text_path'] = ops['persist_text_markdown_locally'](
        final_content,
        model_name=ctx.model_name,
        source_file_name=artifacts.file_name,
        mode='vision_analysis_pdf_scan',
    )
    logging.info(
        'PDF OCR artifact persisted: path=%s',
        payload.get('saved_text_path') or 'none',
    )
    ops['log_pdf_infer_event'](
        instance_id=ctx.instance_id,
        model_name=ctx.model_name,
        backend=ctx.backend,
        capability=ctx.capability,
        prompt=ctx.user_prompt,
        file_name=artifacts.file_name,
        file_sha256=artifacts.file_sha256,
        status='ok',
        mode=payload['mode'],
        content=final_content,
        warnings=artifacts.pdf_warnings,
        pdf_source='rendered_pages',
        pdf_total_pages=payload['pdf_total_pages'],
        pdf_processed_pages=payload['pdf_processed_pages'],
        artifact_path=payload.get('saved_text_path'),
    )
    logging.info(
        'PDF OCR response ready: instance=%s pages=%s warnings=%s',
        ctx.instance_id,
        payload['pdf_processed_pages'],
        len(artifacts.pdf_warnings),
    )
    return payload, 200


def _build_mlx_multimodal_user_message(prompt: str, image_b64: Optional[str] = None) -> dict:
    text = str(prompt or '').strip()
    if not image_b64:
        return {'role': 'user', 'content': text}
    data_url = image_b64 if str(image_b64).startswith('data:') else f'data:image/png;base64,{image_b64}'
    return {
        'role': 'user',
        'content': [
            {'type': 'input_text', 'text': text},
            {'type': 'input_image', 'image_url': data_url},
        ],
    }


def _run_chat_fallback(
    ctx: InferContext,
    artifacts: InferArtifacts,
    ops: Dict[str, Callable[..., Any]],
) -> Tuple[dict, int]:
    prompt = ctx.prompt
    warnings = list(artifacts.pdf_warnings or [])
    if artifacts.text_from_file:
        file_content_prefix = '[Attached file content]'
        if artifacts.text_from_file_truncated:
            inline_bytes = int(artifacts.text_from_file_inline_bytes or 0)
            total_bytes = int(artifacts.text_from_file_total_bytes or 0)
            file_content_prefix = (
                '[Attached file content truncated to first '
                f'{inline_bytes} of {total_bytes} bytes; request more or use chunks if needed]'
            )
            warnings.append(
                f'Attached text file exceeded inline limit and was truncated to {inline_bytes} of {total_bytes} bytes.'
            )
        if prompt:
            prompt = f'{prompt}\n\n{file_content_prefix}\n{artifacts.text_from_file}'
        else:
            prompt = f'{file_content_prefix}\n{artifacts.text_from_file}'
    elif artifacts.file_kind == 'pdf' and artifacts.pdf_page_images:
        error_message = (
            'Detected a PDF without extractable text. '
            'Use a vision_analysis model (for example deepseek-ocr) for scanned PDFs.'
        )
        ops['log_pdf_infer_event'](
            instance_id=ctx.instance_id,
            model_name=ctx.model_name,
            backend=ctx.backend,
            capability=ctx.capability,
            prompt=ctx.user_prompt,
            file_name=artifacts.file_name,
            file_sha256=artifacts.file_sha256,
            status='error',
            mode='chat',
            error=error_message,
            warnings=warnings,
            pdf_source='rendered_pages',
            pdf_total_pages=artifacts.pdf_total_pages or len(artifacts.pdf_page_images),
            pdf_processed_pages=len(artifacts.pdf_page_images),
        )
        return {'error': error_message, 'warnings': warnings}, 400
    if artifacts.image_b64:
        if ctx.backend in {'mlx', 'llama_cpp'}:
            mlx_out = _run_backend_chat_completion(
                ctx,
                ops,
                [_build_mlx_multimodal_user_message(prompt or 'Describe the attached image.', artifacts.image_b64)],
                timeout_sec=ctx.infer_timeout_sec,
            )
            return (
                {
                    'instance_id': ctx.instance_id,
                    'capability': ctx.capability,
                    'mode': 'chat_with_image',
                    'content': mlx_out.get('content', ''),
                    'warnings': warnings,
                    'result': mlx_out.get('result'),
                },
                200,
            )
        data_out = ops['ollama_generate'](
            ctx.port,
            ctx.model_name,
            prompt or 'Describe the attached image.',
            images=[artifacts.image_b64],
            timeout_sec=ctx.infer_timeout_sec,
        )
        return (
                {
                    'instance_id': ctx.instance_id,
                    'capability': ctx.capability,
                    'mode': 'chat_with_image',
                    'content': ops['extract_generate_content'](data_out),
                    'warnings': warnings,
                    'result': data_out,
                },
                200,
            )

    if not prompt:
        return {'error': 'No prompt was provided.'}, 400
    if ctx.backend in {'mlx', 'llama_cpp'}:
        chat_result = _run_backend_chat_completion(
            ctx,
            ops,
            [{'role': 'user', 'content': prompt}],
            timeout_sec=ctx.infer_timeout_sec,
        )
        content = chat_result.get('content', '')
    else:
        chat_result = ops['ollama_chat'](ctx.port, ctx.model_name, [{'role': 'user', 'content': prompt}])
        content = chat_result.get('content', '')
    payload = {
        'instance_id': ctx.instance_id,
        'capability': ctx.capability,
        'mode': 'chat',
        'content': content,
        'warnings': warnings,
    }
    artifact_requests = [
        dict(item)
        for item in (ctx.text_artifact_requests or [])
        if isinstance(item, dict)
    ]
    if not artifact_requests:
        artifact_requests = detect_text_artifact_requests(
            ctx.user_prompt or ctx.prompt,
            source_available=bool(artifacts.text_from_file or artifacts.temp_path),
            source_extension=Path(artifacts.file_name or '').suffix,
            source_name=Path(artifacts.file_name or '').stem,
            source_path=(
                str(artifacts.temp_path)
                if isinstance(artifacts.temp_path, (str, Path))
                else None
            ),
        )
    persist_text_artifact = ops.get('persist_text_artifact_locally')
    artifact_payloads = extract_text_artifact_payloads(content, artifact_requests)
    saved_text_artifacts: list[dict[str, Any]] = []
    if artifact_payloads and persist_text_artifact:
        for artifact_payload in artifact_payloads:
            artifact_request = artifact_payload.get('artifact_request') if isinstance(artifact_payload, dict) else {}
            artifact_content = str((artifact_payload or {}).get('content') or '').strip()
            if not artifact_content or not isinstance(artifact_request, dict):
                continue
            saved_text_path = persist_text_artifact(
                artifact_content,
                model_name=ctx.model_name,
                source_name=artifact_request.get('source_name') or 'generated-text',
                mode='chat_text_artifact',
                extension=artifact_request.get('extension') or 'txt',
                target_path=artifact_request.get('target_path'),
            )
            if saved_text_path:
                saved_text_artifacts.append(
                    {
                        'path': saved_text_path,
                        'text_artifact_request': artifact_request,
                        'document_output_kind': 'document',
                    }
                )
    if saved_text_artifacts:
        first_artifact = saved_text_artifacts[0]
        payload['saved_text_path'] = first_artifact['path']
        payload['document_output_kind'] = 'document'
        payload['text_artifact_request'] = first_artifact.get('text_artifact_request')
        payload['saved_text_artifacts'] = saved_text_artifacts
        payload['text_artifact_requests'] = [
            item.get('text_artifact_request')
            for item in saved_text_artifacts
            if isinstance(item.get('text_artifact_request'), dict)
        ]
    if artifacts.file_kind == 'pdf':
        ops['log_pdf_infer_event'](
            instance_id=ctx.instance_id,
            model_name=ctx.model_name,
            backend=ctx.backend,
            capability=ctx.capability,
            prompt=ctx.user_prompt,
            file_name=artifacts.file_name,
            file_sha256=artifacts.file_sha256,
            status='ok',
            mode='chat',
            content=payload['content'],
            warnings=warnings,
            pdf_source='text_layer' if artifacts.text_from_file else None,
            pdf_total_pages=artifacts.pdf_total_pages or None,
            pdf_processed_pages=None,
        )
    return payload, 200
