"""Shared normalized prompt intent analysis for Ghost routing and control hints."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Optional

from helpers.model_capabilities import (
    CAPABILITY_IMAGE_GENERATION,
    CAPABILITY_SPEECH_TO_TEXT,
    CAPABILITY_TEXT_TO_SPEECH,
    CAPABILITY_VISION_ANALYSIS,
)
from ollmo_g.text_artifact_revision_intent import classify_named_text_revision_intent
from ollmo_services.tts_source import resolve_explicit_tts_source

_INTENT_PROMPT_NORMALIZATIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r'\bimgae\b', re.IGNORECASE), 'image'),
    (re.compile(r'\bigmae\b', re.IGNORECASE), 'image'),
    (re.compile(r'\biamge\b', re.IGNORECASE), 'image'),
    (re.compile(r'\bpicutre\b', re.IGNORECASE), 'picture'),
    (re.compile(r'\bpitcure\b', re.IGNORECASE), 'picture'),
    (re.compile(r'\billusration\b', re.IGNORECASE), 'illustration'),
    (re.compile(r'\bkreiiere\b', re.IGNORECASE), 'generiere'),
    (re.compile(r'\bkreiere\b', re.IGNORECASE), 'generiere'),
    (re.compile(r'\bkreier(?:e|en|st|t)?\b', re.IGNORECASE), 'generier'),
    (re.compile(r'\baudiofile\b', re.IGNORECASE), 'audio file'),
    (re.compile(r'\baudioformat\b', re.IGNORECASE), 'audio format'),
    (re.compile(r'\bsoundformat\b', re.IGNORECASE), 'sound format'),
    (re.compile(r'\breadaloud\b', re.IGNORECASE), 'read aloud'),
)

_LANGUAGE_HINTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r'\b(german|deutsch(?:e|er|es|en|em)?|aleman|allemand)\b', re.IGNORECASE), 'de'),
    (re.compile(r'\b(english|englisch|ingles|anglais)\b', re.IGNORECASE), 'en'),
    (re.compile(r'\b(french|francais|franzosisch|franzoesisch)\b', re.IGNORECASE), 'fr'),
    (re.compile(r'\b(spanish|espanol|spanisch|castellano)\b', re.IGNORECASE), 'es'),
    (re.compile(r'\b(italian|italiano|italienisch)\b', re.IGNORECASE), 'it'),
    (re.compile(r'\b(portuguese|portugues|portugiesisch)\b', re.IGNORECASE), 'pt'),
    (re.compile(r'\b(japanese|japanisch|nihongo)\b', re.IGNORECASE), 'ja'),
    (re.compile(r'\b(korean|koreanisch|hanguk(?:eo)?)\b', re.IGNORECASE), 'ko'),
    (re.compile(r'\b(russian|russisch|russkiy)\b', re.IGNORECASE), 'ru'),
    (re.compile(r'\b(chinese|chinesisch|zhongwen|mandarin)\b', re.IGNORECASE), 'zh'),
]

_LANGUAGE_CODE_HINTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r'\b(?:lang|language|lang_code|sprache)\s*[:=]?\s*(de)\b', re.IGNORECASE), 'de'),
    (re.compile(r'\b(?:lang|language|lang_code|sprache)\s*[:=]?\s*(en)\b', re.IGNORECASE), 'en'),
    (re.compile(r'\b(?:lang|language|lang_code|sprache)\s*[:=]?\s*(fr)\b', re.IGNORECASE), 'fr'),
    (re.compile(r'\b(?:lang|language|lang_code|sprache)\s*[:=]?\s*(es)\b', re.IGNORECASE), 'es'),
    (re.compile(r'\b(?:lang|language|lang_code|sprache)\s*[:=]?\s*(it)\b', re.IGNORECASE), 'it'),
    (re.compile(r'\b(?:lang|language|lang_code|sprache)\s*[:=]?\s*(pt)\b', re.IGNORECASE), 'pt'),
    (re.compile(r'\b(?:lang|language|lang_code|sprache)\s*[:=]?\s*(ja)\b', re.IGNORECASE), 'ja'),
    (re.compile(r'\b(?:lang|language|lang_code|sprache)\s*[:=]?\s*(ko)\b', re.IGNORECASE), 'ko'),
    (re.compile(r'\b(?:lang|language|lang_code|sprache)\s*[:=]?\s*(ru)\b', re.IGNORECASE), 'ru'),
    (re.compile(r'\b(?:lang|language|lang_code|sprache)\s*[:=]?\s*(zh)\b', re.IGNORECASE), 'zh'),
]

_VOICE_STYLE_HINTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r'\b(female|feminine|woman(?:\'s)?|weiblich(?:e|en|er|es)?|frau|mujer|femenina|voix feminine|voce feminina)\b', re.IGNORECASE), 'female'),
    (re.compile(r'\b(male|masculine|man(?:\'s)?|mann|mannlich(?:e|en|er|es)?|maennlich(?:e|en|er|es)?|hombre|masculina|voix masculine|voce masculina)\b', re.IGNORECASE), 'male'),
    (re.compile(r'\b(warm(?:e|en|er|es)?|warmer|calida|calido|chaleureuse)\b', re.IGNORECASE), 'warm'),
    (re.compile(r'\b(calm|calmer|ruhig(?:e|en|er|es)?|tranquil[oa]|calme)\b', re.IGNORECASE), 'calm'),
    (re.compile(r'\b(soft|softer|gentle|weich(?:e|en|er|es)?|suave|douce)\b', re.IGNORECASE), 'soft'),
    (re.compile(r'\b(clear|klar(?:e|en|er|es)?|clara|claro|claire)\b', re.IGNORECASE), 'clear'),
    (re.compile(r'\b(deep|tief(?:e|en|er|es)?|profund[oa]|grave)\b', re.IGNORECASE), 'deep'),
    (re.compile(r'\b(serious|stern|grave|ernst(?:e|en|er|es)?|ernsthaft(?:e|en|er|es)?|eindringlich(?:e|en|er|es)?)\b', re.IGNORECASE), 'serious'),
    (re.compile(r'\b(urgent|warning|alert|alarm|emergency|warn(?:ing)?slogan|warnhinweis|notfall|dringlich(?:e|en|er|es)?)\b', re.IGNORECASE), 'urgent'),
    (re.compile(r'\b(bright|hell(?:e|en|er|es)?|brillante|lumineuse)\b', re.IGNORECASE), 'bright'),
    (re.compile(r'\b(elegant|elegante?n?)\b', re.IGNORECASE), 'elegant'),
    (re.compile(r'\b(friendly|freundlich(?:e|en)?|amigable|amicale)\b', re.IGNORECASE), 'friendly'),
    (re.compile(r'\b(natural|naturlich(?:e|en)?|natuerlich(?:e|en)?|natural(?:e)?)\b', re.IGNORECASE), 'natural'),
]

_AUDIO_FORMAT_HINTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r'\b(mp3)\b', re.IGNORECASE), 'mp3'),
    (re.compile(r'\b(wav|wave)\b', re.IGNORECASE), 'wav'),
    (re.compile(r'\b(flac)\b', re.IGNORECASE), 'flac'),
]

_LITERAL_PAYLOAD_MASK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r'```[\s\S]*?```'),
    re.compile(r'"[^"\n]*"'),
    re.compile(r'“[^”\n]*”'),
    re.compile(r'„[^“\n]*“'),
    re.compile(r'«[^»\n]*»'),
    re.compile(r'‹[^›\n]*›'),
    re.compile(r'(?<!`)`[^`\n]*`(?!`)'),
    re.compile(r"(?<![\\\w])'[^'\n]*'(?!\w)"),
)
_TEXT_PREPARATION_HINTS: list[re.Pattern[str]] = [
    re.compile(r'\b(extract|pull out|read the text|read the quoted text|ocr)\b', re.IGNORECASE),
    re.compile(r'\b(translate|translation|übersetz(?:e|ung)|uebersetz(?:e|ung))\b', re.IGNORECASE),
    re.compile(r'\b(summarize|summarise|summary|rewrite|rephrase|clean(?:\s+up)?|refine|prepare|draft)\b', re.IGNORECASE),
    re.compile(r'\b(condense|shorten|paraphrase|abridge|adapt|convert|transform)\b', re.IGNORECASE),
    re.compile(r'\b(make|create|write|give|produce|prepare)\b[\s\S]{0,72}\b(summary|version|sentence|paraphrase|rewrite|result)\b', re.IGNORECASE),
    re.compile(r'\b(write|compose|create|generate|invent|make up|come up with)\b[\s\S]{0,48}\b(story|script|narration|voiceover|poem|essay|caption|summary|prompt(?:s)?)\b', re.IGNORECASE),
    re.compile(r'\b(write|compose|create|generate|invent|make up|come up with)\b[\s\S]{0,64}\b(text|paragraph(?:s)?|section(?:s)?|stanza(?:s)?|verse(?:s)?)\b', re.IGNORECASE),
    re.compile(r'\b(schreib(?:e|st)?|verfass(?:e|t)?|dicht(?:e|est)?|erzahl(?:e|st)?|formulier(?:e|st)?|entwirf(?:st)?)\b[\s\S]{0,72}\b(geschichte(?:n)?|gedicht(?:e)?|text(?:e)?|witz(?:e)?|joke(?:s)?|aufsatz|essay|zusammenfassung|prompt(?:s)?|absatz(?:e|en)?|abschnitt(?:e|en)?|strophe(?:n)?)\b', re.IGNORECASE),
]
_TRANSLATION_OUTPUT_HINTS: list[re.Pattern[str]] = [
    re.compile(r'\b(translate|translation|übersetz(?:e|ung)|uebersetz(?:e|ung))\b', re.IGNORECASE),
]
_AUDIO_FOLLOW_UP_HINTS: list[re.Pattern[str]] = [
    re.compile(r"\b(read|say|speak)\b(?:\s+[\w'-]+){0,8}\s+(?:aloud|out loud)\b", re.IGNORECASE),
    re.compile(r'\b(generate|create|make)\b[\s\S]{0,32}\b(audio|audio file|voice clip|voice note)\b', re.IGNORECASE),
    re.compile(r'\b(?:turn|convert|transform)\b[\s\S]{0,72}\b(?:into|to|as)\b[\s\S]{0,32}\b(?:audio|speech|spoken|mp3|wav)\b', re.IGNORECASE),
    re.compile(r'\b(generier(?:e|en|st|t)?|erzeug(?:e|en|st|t)?|erstell(?:e|en|st|t)?|mach(?:e|en|st|t)?)\b[\s\S]{0,48}\b(audio|audio(?:\s|-)?datei|audio(?:\s|-)?fassung(?:en)?|audio(?:\s|-)?version(?:en)?|sprachversion|horversion|hoerversion)\b', re.IGNORECASE),
    re.compile(
        r'\b(?:replace|ersetz(?:e|en|t|st)?)\b[^.;!?]{0,120}'
        r'\b(?:audio[\s-]?branch|audiozweig(?:e)?)\b[^.;!?]{0,80}'
        r'\b(?:with|by|durch)\b[^.;!?]{0,80}'
        r'\b(?:audio[\s-]?(?:version(?:s)?|variant(?:s)?|fassung(?:en)?))\b',
        re.IGNORECASE,
    ),
    re.compile(r'\b(spoken version|voice version|audio clip|voice clip|something i can listen to)\b', re.IGNORECASE),
    re.compile(r'\b(?:lies|lese|liese|sprich|sag(?:e)?)\b[\s\S]{0,72}\b(?:vor|laut)\b', re.IGNORECASE),
    re.compile(r'\b(?:vorlesen|vorgelesen|horversion|hoerversion|sprachversion)\b', re.IGNORECASE),
]
_AUDIO_OUTPUT_NEGATION_HINTS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(?:do not|don't|dont|without|no)\b[\s\S]{0,80}\b"
        r"(?:generate|create|make|produce|render|speak|read(?:\s+\w+){0,3}\s+aloud|audio|voice|tts)\b"
        r"[\s\S]{0,80}\b(?:audio|voice|speech|tts|file|clip|output)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r'\b(?:generier(?:e|en|st|t)?|erzeug(?:e|en|st|t)?|erstell(?:e|en|st|t)?|mach(?:e|en|st|t)?)\b'
        r'[^.,;!?]{0,32}\bkein(?:e|en|er|es)?(?:\s+neue?s?)?\b[^.,;!?]{0,32}'
        r'\b(?:audio|audio(?:\s|-)?datei|sprachversion|horversion|hoerversion|stimme)\b',
        re.IGNORECASE,
    ),
    re.compile(
        r'\bkein(?:e|en|er|es)?(?:\s+neue?s?)?\b[^.,;!?]{0,40}'
        r'\b(?:audio|audio(?:\s|-)?datei|sprachversion|horversion|hoerversion|stimme)\b'
        r'[^.,;!?]{0,40}\b(?:generier(?:e|en|st|t)?|erzeug(?:e|en|st|t)?|erstell(?:e|en|st|t)?|mach(?:e|en|st|t)?)\b',
        re.IGNORECASE,
    ),
]
_AUDIO_MATERIALIZATION_ACTION_RE = re.compile(
    r'\b(?:generate|create|make|produce|render|voice|narrate|'
    r'generier(?:e|en|st|t)?|erzeug(?:e|en|st|t)?|erstell(?:e|en|st|t)?|'
    r'mach(?:e|en|st|t)?|verton(?:e|en|st|t)?)\b'
    r'(?:(?!\b(?:but|however|instead|yet|aber|sondern|jedoch|stattdessen)\b)[^.;!?]){0,96}'
    r'\b(?:audio|audio[\s-]?file|voice\s+clip|speech|tts|mp3|wav|'
    r'audio[\s-]?datei|sprachversion|horversion|hoerversion)\b|'
    r'\b(?:read|say|speak|lies|lese|sprich|sag(?:e)?)\b'
    r'(?:(?!\b(?:but|however|instead|yet|aber|sondern|jedoch|stattdessen)\b)[^.;!?]){0,72}'
    r'\b(?:aloud|out\s+loud|vor|laut)\b',
    re.IGNORECASE,
)
_VISUAL_FOLLOW_UP_HINTS: list[re.Pattern[str]] = [
    re.compile(r'\b(show|display|visuali[sz]e|illustrate|render)\b[\s\S]{0,48}\b(it|them|this|that)\b[\s\S]{0,40}\b(as|into)?[\s\S]{0,20}\b(image|picture|photo|illustration|scene)\b', re.IGNORECASE),
    re.compile(r'\b(show (?:it|them|this|that) to me)\b[\s\S]{0,24}\b(as|into)?[\s\S]{0,20}\b(image|picture|photo|illustration|scene)\b', re.IGNORECASE),
    re.compile(r'\b(show|display|visuali[sz]e|illustrate|render)\b[\s\S]{0,32}\b(?:an?|the)\b[\s\S]{0,8}\b(image|picture|photo|illustration|scene)\b[\s\S]{0,24}\bof\b[\s\S]{0,12}\b(it|them|this|that|me|him|her)\b', re.IGNORECASE),
    re.compile(r'\b(turn|make|convert|transform)\b[\s\S]{0,48}\b(it|them|this|that)\b[\s\S]{0,32}\b(into|as)\b[\s\S]{0,20}\b(image|picture|photo|illustration|scene)\b', re.IGNORECASE),
    re.compile(r'\b(zeig(?:e)?|visualisiere|stelle(?:\s+\w+){0,2}\s+dar)\b[\s\S]{0,48}\b(es|sie|das|dies|diese|ihn|mich)\b[\s\S]{0,40}\b(als|zu)?[\s\S]{0,20}\b(bild(?:er)?|foto(?:s)?|illustration(?:en)?|szene(?:n)?)\b', re.IGNORECASE),
    re.compile(r'\b(zeig(?:e)?(?:\s+\w+){0,3}\s+mir)\b[\s\S]{0,24}\b(als|zu)?[\s\S]{0,20}\b(bild(?:er)?|foto(?:s)?|illustration(?:en)?|szene(?:n)?)\b', re.IGNORECASE),
    re.compile(r'\b(?:beschreib(?:e|st|t|en)?|schildere?|schilder(?:e|st|t|n)?)\b[\s\S]{0,140}\b(?:bild(?:er)?|foto(?:s)?|illustration(?:en)?|szene(?:n)?)\b[\s\S]{0,96}\b(?:und|danach|nachher|anschliessend|anschliesend|anschließend)\b[\s\S]{0,40}\b(?:generier(?:e|en|st|t)?|erzeug(?:e|en|st|t)?|erstell(?:e|en|st|t)?|mach(?:e|en|st|t)?)\b', re.IGNORECASE),
    re.compile(r'\b(?:for|fur|fuer)\s+(?:each|every|jedem|jeder|jeden|jede)\b[\s\S]{0,24}\b(?:an?|one|single|ein|eine|einen)\b[\s\S]{0,16}\b(image|picture|photo|illustration|scene|bild(?:er)?|foto(?:s)?|illustration(?:en)?|szene(?:n)?)\b', re.IGNORECASE),
    re.compile(r'\b(?:an?|one|single|ein|eine|einen)\b[\s\S]{0,16}\b(image|picture|photo|illustration|scene|bild(?:er)?|foto(?:s)?|illustration(?:en)?|szene(?:n)?)\b[\s\S]{0,24}\b(?:for|of|von|fur|fuer)\s+(?:each|every|jedem|jeder|jeden|jede)\b', re.IGNORECASE),
]
_VISUAL_TEXT_PREPARATION_HINTS: list[re.Pattern[str]] = [
    re.compile(r'\b(describe|depict|portray)\b[\s\S]{0,96}\b(vivid|detail|detailed|rich|atmospheric|immersive)\b', re.IGNORECASE),
    re.compile(r'\b(beschreibe|schildere|zeichne)\b[\s\S]{0,96}\b(lebendig|detail|details|detailliert|reich|atmospharisch|immersiv)\b', re.IGNORECASE),
    re.compile(r'\b(?:describe|beschreib(?:e|st|t|en)?|beschreibe|schilder(?:e|st|t|n)?|schildere)\b[\s\S]{0,96}\b(?:image(?:s)?|picture(?:s)?|photo(?:s)?|scene(?:s)?|bild(?:er)?|foto(?:s)?|szene(?:n)?)\b', re.IGNORECASE),
    re.compile(r'\b(describe|beschreib(?:e|st|en)?|beschreibe|schilder(?:e|st|n)?)\b[\s\S]{0,96}\b(place(?:s)?|location(?:s)?|scene(?:s)?|situation(?:s|en)?|scenario(?:s)?|world(?:s)?|idea(?:s)?|concept(?:s)?|ort(?:e)?|platze|plaetze|szen(?:en|arien)|welt(?:en)?|idee(?:n)?|konzept(?:e)?)\b', re.IGNORECASE),
    re.compile(r'\b(place(?:s)?|location(?:s)?|scene(?:s)?|situation(?:s|en)?|scenario(?:s)?|world(?:s)?|idea(?:s)?|concept(?:s)?|ort(?:e)?|platze|plaetze|szen(?:en|arien)|welt(?:en)?|idee(?:n)?|konzept(?:e)?)\b[\s\S]{0,96}\b(describe|beschreib(?:e|st|en)?|beschreibe|schilder(?:e|st|n)?)\b', re.IGNORECASE),
    re.compile(r'\b(write|compose|invent|come up with|dream up)\b[\s\S]{0,48}\b(prompt(?:s)?|scene(?:s)?|situation(?:s)?|scenario(?:s)?|place(?:s)?|location(?:s)?|concept(?:s)?)\b', re.IGNORECASE),
    re.compile(r'\b(stell(?:\s+\w+){0,2}\s+dir|denk(?:\s+\w+){0,2}\s+dir|erfinde)\b[\s\S]{0,64}\b(ort(?:e)?|platze|plaetze|situation(?:en)?|szen(?:en|arien)|welten|ideen|prompt(?:s)?)\b', re.IGNORECASE),
]
_VISUAL_CREATIVE_DELEGATION_HINTS: list[re.Pattern[str]] = [
    re.compile(r'\b(you write the prompt(?:s)?|write the prompt(?:s)? yourself|invent the prompt(?:s)?|come up with the prompt(?:s)?|dream up the prompt(?:s)?)\b', re.IGNORECASE),
    re.compile(r'\b(choice is yours|your choice|you decide|you choose|free to choose|follow your own taste|use your own taste|surprise me|surprise us)\b', re.IGNORECASE),
    re.compile(r'\b(du bist frei in der wahl|frei in der wahl|nach deinem(?:\s+eigenen)? gusto|nach deinem geschmack|nach deinem(?:\s+eigenen)? thema|nach deinem(?:\s+eigenen)? topic|mit deinem(?:\s+eigenen)? thema|eigenes thema|your own theme|your own topic|du entscheidest|such dir(?:\s+\w+){0,3}\s+aus|wahle du(?:\s+\w+){0,3}\s+aus|uberrasch(?:e| mich))\b', re.IGNORECASE),
]
_NARRATION_SCRIPT_HINT_RE = re.compile(r'\b(narration script|voiceover script|script for narration|script for voiceover)\b', re.IGNORECASE)
_MATERIALIZATION_TARGET_RE = re.compile(
    r'\b('
    r'image(?:s)?|picture(?:s)?|photo(?:s)?|illustration(?:s)?|render(?:s)?|scene(?:s)?|'
    r'audio(?:s|[\s-]?fassung(?:en)?|[\s-]?version(?:en)?)?|voice|voiceover|speech|tts|html|landing page|asset(?:s)?|artifact(?:s)?|'
    r'bild(?:er|idee(?:n)?|variante(?:n)?|version(?:en)?|konzept(?:e)?)?|foto(?:s)?|'
    r'illustration(?:en)?|szene(?:n)?|schnittzeichnung(?:en)?|nachtaufnahme(?:n)?|audio datei|stimme'
    r')\b',
    re.IGNORECASE,
)
_ARTIFACT_PREPARATION_SOURCE_PATTERN = (
    r'(?:prompt(?:s)?|script(?:s)?|placeholder(?:s)?|description(?:s)?|prose|text(?:s)?|'
    r'transcript(?:s)?|draft(?:s)?|brief(?:s)?|outline(?:s)?|idea(?:s)?|plan(?:s)?|'
    r'caption(?:s)?|summary|preparation|vorbereitung|beschreibung(?:en)?|skript(?:e)?|'
    r'platzhalter|entwurf(?:e)?|idee(?:n)?|text(?:e)?|transkript(?:e)?)'
)
_ARTIFACT_FULFILLMENT_TARGET_PATTERN = (
    r'(?:artifact(?:s)?|artefact(?:s)?|artefakt(?:e|en)?|output(?:s)?|obligation(?:s)?|'
    r'fulfill(?:ment)?|fulfil(?:ment)?|completion|file(?:s)?|datei(?:en)?|'
    r'image(?:s)?|picture(?:s)?|bild(?:er)?|audio|mp3|wav|html|css|'
    r'generated\s+(?:image|audio|file)|erzeugte(?:s|n|r|m)?\s+(?:bild|audio|datei))'
)
_ARTIFACT_FULFILLMENT_NEGATION_RE: list[re.Pattern[str]] = [
    re.compile(
        rf'\b{_ARTIFACT_PREPARATION_SOURCE_PATTERN}\b[\s\S]{{0,160}}\b'
        r'(?:as\s+preparation\s+only|only\s+preparation|preparation\s+only|'
        r'nur\s+als\s+vorbereitung|nur\s+vorbereitung)\b[\s\S]{0,100}\b'
        r'(?:not|no|nicht|kein(?:e|en|er|es)?)\b[\s\S]{0,100}\b'
        rf'{_ARTIFACT_FULFILLMENT_TARGET_PATTERN}\b',
        re.IGNORECASE,
    ),
    re.compile(
        r'\b(?:treat|consider|regard|handle|count|accept|use|werte|werten|'
        r'betrachte|behandle|zaehle|zähle)\w*\b[\s\S]{0,100}\b'
        rf'{_ARTIFACT_PREPARATION_SOURCE_PATTERN}\b[\s\S]{{0,160}}\b'
        r'(?:not|no|nicht|kein(?:e|en|er|es)?)\b[\s\S]{0,100}\b'
        rf'{_ARTIFACT_FULFILLMENT_TARGET_PATTERN}\b',
        re.IGNORECASE,
    ),
    re.compile(
        rf'\b{_ARTIFACT_PREPARATION_SOURCE_PATTERN}\b[\s\S]{{0,140}}\b'
        r'(?:does\s+not|do\s+not|must\s+not|should\s+not|cannot|can[\'’]?t|'
        r'darf\s+nicht|zaehlt\s+nicht|zahlt\s+nicht|zählt\s+nicht|gilt\s+nicht)\b'
        r'[\s\S]{0,120}\b(?:count|count\s+as|satisfy|fulfill|fulfil|replace|'
        r'stand\s+in\s+for|zaehlen|zählen|gelten|ersetzen)?\b[\s\S]{0,100}\b'
        rf'{_ARTIFACT_FULFILLMENT_TARGET_PATTERN}\b',
        re.IGNORECASE,
    ),
    re.compile(
        rf'\b{_ARTIFACT_PREPARATION_SOURCE_PATTERN}\b[\s\S]{{0,120}}\b'
        r'(?:is|are|counts?|gilt|gelten|zaehlt|zahlt|zählt|ist|sind)\b'
        r'[\s\S]{0,60}\b(?:not|no|nicht|kein(?:e|en|er|es)?)\b[\s\S]{0,100}\b'
        rf'{_ARTIFACT_FULFILLMENT_TARGET_PATTERN}\b',
        re.IGNORECASE,
    ),
    re.compile(
        r'\b(?:do\s+not|don[\'’]?t|must\s+not|should\s+not|never|nicht|darf\s+nicht)\b'
        r'[\s\S]{0,100}\b(?:count|treat|accept|use|regard|werten|zaehlen|zählen|gelten)\b'
        r'[\s\S]{0,100}\b'
        rf'{_ARTIFACT_PREPARATION_SOURCE_PATTERN}\b[\s\S]{{0,120}}\b(?:as|als)\b'
        r'[\s\S]{0,100}\b'
        rf'{_ARTIFACT_FULFILLMENT_TARGET_PATTERN}\b',
        re.IGNORECASE,
    ),
]
_MATERIALIZATION_CARDINALITY_CONSTRAINT_RE = re.compile(
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
_MATERIALIZATION_ONLY_OUTPUT_CONTRAST_RE = re.compile(
    r"\b(?:do\s+not|don[\'’]?t|dont)\s+only\s+(?:output|return|write|provide|give|show|display)\b|"
    r"\b(?:not|nicht)\s+only\s+(?:output|return|write|provide|give|show|display)\b|"
    r"\bnicht\s+nur\s+(?:ausgeben|zurueckgeben|zurückgeben|schreiben|zeigen|anzeigen)\b",
    re.IGNORECASE,
)
_EXPLICIT_DEFER_MATERIALIZATION_RE: list[re.Pattern[str]] = [
    re.compile(
        r'(?:^|[.;!?\n])\s*(?:no|without)\s+(?:an?\s+)?'
        r'(?:image(?:s)?|picture(?:s)?|photo(?:s)?|illustration(?:s)?)'
        r'(?:\s+(?:please|required|needed|now|yet))?\s*(?=$|[.;!?\n])',
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:do not|don't|dont|not yet|not now|hold off on|wait before)\b[\s\S]{0,96}\b"
        r'(?:generate|create|make|render|produce|materiali[sz]e|show|display|read(?:\s+\w+){0,3}\s+aloud|'
        r'speak|turn(?:\s+\w+){0,3}\s+into)\b',
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:do not|don't|dont|not yet|not now|hold off on|wait before)\b[\s\S]{0,96}\b"
        r'(?:image(?:s)?|picture(?:s)?|photo(?:s)?|illustration(?:s)?|render(?:s)?|audio|voice|voiceover|speech|'
        r'html|landing page|asset(?:s)?|artifact(?:s)?)\b',
        re.IGNORECASE,
    ),
    re.compile(
        r'\b(?:later|later on|for later|afterwards|in a later turn|in the next turn)\b[\s\S]{0,64}\b'
        r'(?:generate|create|make|render|produce|read(?:\s+\w+){0,3}\s+aloud|speak|audio|voice|image(?:s)?|'
        r'picture(?:s)?|photo(?:s)?|illustration(?:s)?|html)\b',
        re.IGNORECASE,
    ),
    re.compile(
        r'\b(?:first|initially)\b[\s\S]{0,40}\b(?:produce|output|return|give|provide|draft|write)\b'
        r'[\s\S]{0,96}\b(?:draft|brief|outline|script|shot map|memo|pack|section(?:s)?|heading(?:s)?)\b'
        r'[\s\S]{0,40}\bonly\b',
        re.IGNORECASE,
    ),
    re.compile(
        r'\b(?:generier(?:e|en|st|t)?|erzeug(?:e|en|st|t)?|erstell(?:e|en|st|t)?|'
        r'mach(?:e|en|st|t)?|zeichne|visualisier(?:e|en|st|t)?)\b'
        r'[^.;!?]{0,64}\b(?:noch\s+)?kein(?:e|en|er|es)?\b[^.;!?]{0,48}'
        r'\b(?:bild(?:er)?|foto(?:s)?|illustration(?:en)?|szene(?:n)?|animation(?:en)?)\b',
        re.IGNORECASE,
    ),
    re.compile(
        r'\b(?:halt(?:e|en|et|est)?|behalt(?:e|en|et|est)?|reserve|reservier(?:e|en|t)?|'
        r'hold|keep|reserve)\b'
        r'[^.;!?]{0,140}\b(?:bild(?:er)?|image(?:s)?|picture(?:s)?|photo(?:s)?|'
        r'foto(?:s)?|audio|voice|speech|tts|stimme|sprachversion|hoerver(?:sion)?|hörver(?:sion)?)\b'
        r'[^.;!?]{0,140}\b(?:zur(?:ue|ü|u)ck|spaeter|später|spatere?|later|future|option(?:en)?|candidate(?:s)?|kandidat(?:en)?|'
        r'moegliche(?:n|r|s)?|mögliche(?:n|r|s)?|mogliche(?:n|r|s)?|possible)\b',
        re.IGNORECASE,
    ),
    re.compile(
        r'\b(?:bild(?:er)?|image(?:s)?|picture(?:s)?|photo(?:s)?|foto(?:s)?|audio|voice|speech|tts|'
        r'stimme|sprachversion|hoerver(?:sion)?|hörver(?:sion)?)\b'
        r'[^.;!?]{0,140}\b(?:moegliche(?:n|r|s)?|mögliche(?:n|r|s)?|mogliche(?:n|r|s)?|possible|spaetere(?:n|r|s)?|'
        r'spätere(?:n|r|s)?|spatere(?:n|r|s)?|later|future)\b'
        r'[^.;!?]{0,140}\b(?:schritt(?:e)?|phase(?:n)?|option(?:en)?|candidate(?:s)?|kandidat(?:en)?|'
        r'zur(?:ue|ü|u)ck|reserve|reservier(?:e|en|t)?|hold|keep)\b',
        re.IGNORECASE,
    ),
]
_TEXT_REVISION_ACTION_RE = re.compile(
    r'\b('
    r'critique|review|revise|revision|edit|rewrite|rework|refine|tighten|polish|lock|finali[sz]e|'
    r'fix|correct|trim|audit|cleanup|clean up|improve'
    r')\b',
    re.IGNORECASE,
)
_TEXT_REVISION_OUTPUT_RE = re.compile(
    r'\b(?:output|return)\s+only\b|\bplain text only\b|\bthese headings exactly\b|\bno headings\b|\bno preface\b',
    re.IGNORECASE,
)
_TEXT_REVISION_TARGET_RE = re.compile(
    r'\b('
    r'draft|brief|script|shot map|outline|memo|pack|section(?:s)?|heading(?:s)?|copy|narration|voiceover|'
    r'character bible|keyframes|production pack|locked|html outline|html|css|stylesheet|text'
    r')\b',
    re.IGNORECASE,
)
_TEMPERAMENT_RULES: list[tuple[re.Pattern[str], str, int, str]] = [
    (re.compile(r'\b(fix|debug|repair|broken|bug|failing|not working|regression|error|issue)\b', re.IGNORECASE), 'repair', 3, 'repair_request'),
    (re.compile(r'\b(explore|investigate|research|look into|compare|alternatives|possibilities|brainstorm options)\b', re.IGNORECASE), 'explorer', 3, 'exploration_request'),
    (re.compile(r'\b(imagine|invent|dream up|make up|worldbuild|whimsical|mystical|poetic|lyrical|atmospheric|surreal)\b', re.IGNORECASE), 'improviser', 3, 'expressive_open_ended_request'),
    (re.compile(r'\b(story|poem|song|lyrics|melody|music|soundtrack|fairytale|myth|narrative)\b', re.IGNORECASE), 'improviser', 2, 'creative_form_request'),
]

_EXPLICIT_ASPECT_RE = re.compile(r'\b(1:1|4:3|3:4|3:2|2:3|16:9|9:16)\b', re.IGNORECASE)
_ASPECT_HINTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r'\b(square|quadratisch|cuadrad[oa]|carre)\b', re.IGNORECASE), '1:1'),
    (re.compile(r'\b(vertical|tall|story format|phone wallpaper|vertikal|verticale)\b', re.IGNORECASE), '9:16'),
    (re.compile(r'\b(portrait|hochformat|retrato|portrait vertical)\b', re.IGNORECASE), '3:4'),
    (re.compile(r'\b(landscape|wide|widescreen|cinematic|banner|horizontal|querformat|panoramico)\b', re.IGNORECASE), '16:9'),
]
_VISUAL_OUTPUT_COUNT_WORDS: dict[str, int] = {
    'a': 1,
    'an': 1,
    'one': 1,
    'single': 1,
    'ein': 1,
    'eine': 1,
    'einen': 1,
    'eins': 1,
    'two': 2,
    'both': 2,
    'couple': 2,
    'zwei': 2,
    'beide': 2,
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
_AUDIO_OUTPUT_COUNT_WORDS: dict[str, int] = {
    'a': 1,
    'an': 1,
    'one': 1,
    'single': 1,
    'ein': 1,
    'eine': 1,
    'einen': 1,
    'eins': 1,
    'two': 2,
    'both': 2,
    'couple': 2,
    'zwei': 2,
    'beide': 2,
    'three': 3,
    'drei': 3,
    'four': 4,
    'vier': 4,
    'five': 5,
    'funf': 5,
    'fuenf': 5,
    'six': 6,
    'sechs': 6,
    'seven': 7,
    'sieben': 7,
    'eight': 8,
    'acht': 8,
    'nine': 9,
    'neun': 9,
    'ten': 10,
    'zehn': 10,
    'eleven': 11,
    'elf': 11,
    'twelve': 12,
    'zwolf': 12,
    'zwoelf': 12,
    'thirteen': 13,
    'dreizehn': 13,
    'fourteen': 14,
    'vierzehn': 14,
    'fifteen': 15,
    'funfzehn': 15,
    'fuenfzehn': 15,
    'sixteen': 16,
    'sechzehn': 16,
    'seventeen': 17,
    'siebzehn': 17,
    'eighteen': 18,
    'achtzehn': 18,
    'nineteen': 19,
    'neunzehn': 19,
    'twenty': 20,
    'zwanzig': 20,
}
_MAX_REQUESTED_AUDIO_OUTPUT_COUNT = 6
_AUDIO_OUTPUT_NOUN_PATTERN = (
    r'(?:audio(?:s)?|audio[\s-]?datei(?:en)?|audio[\s-]?fassung(?:en)?|'
    r'audio[\s-]?version(?:en)?|audio[\s-]?variante(?:n)?|audio[\s-]?zweig(?:e)?|'
    r'voice[\s-]?(?:clip(?:s)?|version(?:s)?|variant(?:s)?)|spoken[\s-]?version(?:s)?|'
    r'sprachversion(?:en)?|horversion(?:en)?|hoerversion(?:en)?)'
)
_ANSWER_AS_AUDIO_DELIVERY_RE = re.compile(
    r'\b(?P<delivery_action>'
    r'give|provide|return|deliver|send|output|'
    r'gib|geb(?:e|en|t)?|liefer(?:e|n|st|t)?|send(?:e|en|est|et)?|'
    r'schick(?:e|en|st|t)?|reich(?:e|en|st|t)?|stell(?:e|en|st|t)?'
    r')\b'
    r'(?P<delivery_scope>[^.;!?\n]{0,160}?)'
    r'\b(?P<response_target>'
    r'answer(?:s)?|response(?:s)?|repl(?:y|ies)|result(?:s)?|output(?:s)?|'
    r'antwort(?:en)?|antworttext(?:e)?|ergebnis(?:se)?|ausgabe(?:n)?'
    r')\b'
    r'(?P<format_scope>[^.;!?\n]{0,96}?)'
    r'\b(?P<format_bridge>'
    r'as|als|in\s+(?:the\s+|der\s+)?form\s+(?:of|von|einer?|einem)'
    r')\b'
    r'(?P<audio_scope>[^.;!?\n]{0,56}?)'
    rf'\b(?P<audio_target>{_AUDIO_OUTPUT_NOUN_PATTERN})\b',
    re.IGNORECASE,
)
_ANSWER_AS_AUDIO_DEFER_RE = re.compile(
    r'\b(?:noch\s+nicht|noch\s+kein(?:e|en|er|es)?|erst\s+spaeter|erst\s+spater|'
    r'spaeter|spater|nicht\s+jetzt|not\s+yet|not\s+now|later|later\s+on|'
    r'in\s+(?:a\s+)?later\s+turn)\b',
    re.IGNORECASE,
)
_ANSWER_AS_AUDIO_POST_TARGET_NEGATION_RE = re.compile(
    r"\b(?:nicht|kein(?:e|en|er|es)?|ohne|not|no|without|do\s+not|don[\'’]?t)\b",
    re.IGNORECASE,
)
_ANSWER_AS_AUDIO_PRE_TARGET_NEGATION_RE = re.compile(
    r"\b(?:nicht|kein(?:e|en|er|es)?|not|no)\b(?:\s+\w+){0,2}\s*$",
    re.IGNORECASE,
)
_ANSWER_AS_AUDIO_PRE_ACTION_NEGATION_RE = re.compile(
    r"\b(?:do\s+not|don[\'’]?t|dont|nicht|kein(?:e|en|er|es)?|no)\s*$",
    re.IGNORECASE,
)
_ANSWER_AS_AUDIO_POSITIVE_CONTRAST_RE = re.compile(
    r'\b(?:nicht\s+nur|not\s+only)\b[^.;!?\n]{0,80}'
    r'\b(?:sondern(?:\s+auch)?|but(?:\s+also)?)\b',
    re.IGNORECASE,
)
_AUDIO_NEGATION_TARGET_RE = re.compile(
    rf'\b{_AUDIO_OUTPUT_NOUN_PATTERN}\b|'
    r'\b(?:voice|voiceover|speech|tts|mp3|wav|stimme|sprachversion|horversion|hoerversion)\b|'
    r'\b(?:read|speak)\b[^.;!?]{0,32}\b(?:aloud|out\s+loud)\b',
    re.IGNORECASE,
)
_AUDIO_DISTRIBUTED_LANGUAGE_OUTPUT_RE = re.compile(
    rf'\b(?:one|a|an|ein|eine|einen)\b[^.;!?]{{0,24}}\b'
    r'(?:english|german|englisch\w*|deutsch\w*)\b[^.;!?]{0,32}\b(?:and|und)\b'
    r'[^.;!?]{0,24}\b(?:one|a|an|ein|eine|einen)\b[^.;!?]{0,24}\b'
    r'(?:english|german|englisch\w*|deutsch\w*)\b[^.;!?]{0,32}'
    rf'\b{_AUDIO_OUTPUT_NOUN_PATTERN}\b',
    re.IGNORECASE,
)
_AUDIO_OUTPUT_COUNT_RE = re.compile(
    rf'\b(?P<count>\d+|{"|".join(sorted(_AUDIO_OUTPUT_COUNT_WORDS.keys(), key=len, reverse=True))})\b'
    rf'(?:[\s,;:()/-]+\w+){{0,4}}[\s,;:()/-]+(?P<noun>{_AUDIO_OUTPUT_NOUN_PATTERN})\b',
    re.IGNORECASE,
)
_AUDIO_OUTPUT_SUFFIX_COUNT_RE = re.compile(
    rf'\b(?P<noun>{_AUDIO_OUTPUT_NOUN_PATTERN})\b'
    rf'\s*(?:[\[(]\s*(?P<count>\d+|{"|".join(sorted(_AUDIO_OUTPUT_COUNT_WORDS.keys(), key=len, reverse=True))})\s*[\])]|'
    rf'(?P<count_x>\d+)\s*[x×]?|'
    rf'[x×]\s*(?P<count_prefixed_x>\d+))(?=\W|$)',
    re.IGNORECASE,
)
_AUDIO_OUTPUT_DELIVERABLE_ACTION_RE = re.compile(
    r'\b(?:generate|create|make|produce|render|narrate|voice|read|speak|'
    r'generier(?:e|en|st|t)?|erzeug(?:e|en|st|t)?|erstell(?:e|en|st|t)?|'
    r'mach(?:e|en|st|t)?|verton(?:e|en|st|t)?|lies|lese|sprich)\b|'
    r'\b(?:replace|ersetz(?:e|en|t|st)?)\b[^.;!?]{0,120}\b(?:with|by|durch)\b',
    re.IGNORECASE,
)
_AUDIO_COUNT_INTERVENING_NON_AUDIO_NOUN_RE = re.compile(
    r'^\s*(?:'
    r'reason(?:s)?|argument(?:s)?|point(?:s)?|paragraph(?:s)?|section(?:s)?|sentence(?:s)?|'
    r'line(?:s)?|idea(?:s)?|example(?:s)?|criterion|criteria|comment(?:s)?|note(?:s)?|'
    r'text(?:s)?|candidate\s+text(?:s)?|text[\s-]?variant(?:s)?|'
    r'text[\s-]?version(?:s)?|draft(?:s)?|option(?:s)?|'
    r'grund(?:e|en)?|argument(?:e|en)?|punkt(?:e|en)?|absatz(?:e|en)?|'
    r'satz(?:e|en)?|satze|zeile(?:n)?|idee(?:n)?|beispiel(?:e|en)?|'
    r'kriteri(?:um|en)|kommentar(?:e|en)?|notiz(?:en)?|'
    r'text[\s-]?variante(?:n)?|text[\s-]?version(?:en)?|entwurf(?:e|en)?|option(?:en)?'
    r')\b',
    re.IGNORECASE,
)
_VISUAL_PRESERVATION_RE = re.compile(
    r'\b(?:bewahr|behalt|erhalt|preserv|keep|retain)\w*\b'
    r'[^.;!?]{0,120}\b(?:bild(?:analyse)?|image(?:\s+analysis)?|picture|visual(?:\s+analysis)?)\b'
    r'[^.;!?]{0,96}\b(?:unverandert|unveraendert|unchanged|intact|as[\s-]?is)\b',
    re.IGNORECASE,
)
_VISUAL_NO_REGENERATION_RE = re.compile(
    r'\b(?:generier|erzeug|erstell|render|create|generate|regenerat)\w*\b'
    r'[^.;!?]{0,56}\b(?:das\s+bild|bild|image|picture|es|it)\b'
    r'[^.;!?]{0,36}\b(?:nicht|not|never)\b[^.;!?]{0,24}\b(?:neu|erneut|again|anew)?\b|'
    r'\b(?:do\s+not|don[\'’]?t|never|nicht)\b[^.;!?]{0,32}'
    r'\b(?:regenerat|recreat|neu\s+generier|erneut\s+erzeug)\w*\b|'
    r'\b(?:do\s+not|don[\'’]?t|never)\b[^.;!?]{0,24}'
    r'\b(?:generate|create|render)\w*\b[^.;!?]{0,32}'
    r'\b(?:it|the\s+image|image|picture)\b[^.;!?]{0,24}\b(?:again|anew)?\b',
    re.IGNORECASE,
)
_VISUAL_NO_REANALYSIS_RE = re.compile(
    r'\b(?:analysier|analy[sz]|inspect|review|pruef|pruf)\w*\b'
    r'[^.;!?]{0,48}\b(?:das\s+bild|bildanalyse|bild|image\s+analysis|analysis|image|picture|es|it|sie|them|diese)\b'
    r'[^.;!?]{0,36}\b(?:nicht|not|never)\b[^.;!?]{0,24}\b(?:neu|erneut|again|anew)?\b|'
    r'\b(?:do\s+not|don[\'’]?t|never|nicht)\b[^.;!?]{0,32}'
    r'\b(?:re[-\s]?analy[sz]|erneut\s+analysier)\w*\b',
    re.IGNORECASE,
)
_VISUAL_GENERATION_ACTION_RE = re.compile(
    r'\b(?:generate|create|make|produce|render|design|draw|illustrate|paint|sketch|'
    r'generier(?:e|en|st|t)?|erzeug(?:e|en|st|t)?|erstell(?:e|en|st|t)?|'
    r'mach(?:e|en|st|t)?|render(?:e|n|st|t)?|zeichn(?:e|en|est|et)?|'
    r'mal(?:e|en|st|t)?|gestalt(?:e|en|est|et)?)\b',
    re.IGNORECASE,
)
_VISUAL_ANALYSIS_ACTION_RE = re.compile(
    r'\b(?:analy[sz]e|analyse|analysier|inspect|examine|review|untersuch|pruef|pruf|bewert)\w*\b',
    re.IGNORECASE,
)
_VISUAL_ACTION_TARGET_RE = re.compile(
    r'\b(?:image(?:s)?|picture(?:s)?|photo(?:s)?|illustration(?:s)?|render(?:s)?|'
    r'bild(?:er|es|ern|e)?|foto(?:s)?|aufnahme(?:n)?|poster(?:bild(?:er)?)?)\b',
    re.IGNORECASE,
)
_VISUAL_SEPARATE_TARGET_RE = re.compile(
    r'\b(?:new|newly|additional|another|other|separate|fresh|attached|uploaded|generated|created|'
    r'neu(?:e|en|er|es|em)?|zusaetzlich(?:e|en|er|es|em)?|zusatzlich(?:e|en|er|es|em)?|'
    r'weiter(?:e|en|er|es|em)?|getrennt(?:e|en|er|es|em)?|angehangt(?:e|en|er|es|em)?|'
    r'angehaengt(?:e|en|er|es|em)?|hochgeladen(?:e|en|er|es|em)?|'
    r'erzeugt(?:e|en|er|es|em)?|generiert(?:e|en|er|es|em)?)\b',
    re.IGNORECASE,
)
_VISUAL_PLURAL_RESULT_REFERENCE_RE = re.compile(
    r'\b(?:both|all|them|these|beide|alle|jedes|jede|jeden|sie|diese)\b',
    re.IGNORECASE,
)
_VISUAL_ACTION_NEGATION_RE = re.compile(
    r'\b(?:do\s+not|don[\'’]?t|never|without|not|no|nicht|nie|ohne|kein(?:e|en|er|es)?)\b',
    re.IGNORECASE,
)
_VISUAL_ACTION_POLARITY_RESET_RE = re.compile(
    r'\b(?:but|however|instead|yet|aber|sondern|jedoch|stattdessen)\b',
    re.IGNORECASE,
)
_VISUAL_ACTION_LEADING_NEGATION_RE = re.compile(
    r'\b(?:do\s+not|don[\'’]?t|never|not|nicht|nie)\b[^.;!?\n]{0,120}$|'
    r'\b(?:without|ohne|no|kein(?:e|en|er|es)?)\s*$',
    re.IGNORECASE,
)
_VISUAL_ACTION_TRAILING_NEGATION_RE = re.compile(
    r'^\s*(?:not|never|nicht|nie)\b',
    re.IGNORECASE,
)
_ADDITIVE_MATERIALIZATION_RE = re.compile(
    r"\b(?:(?:do\s+not|don[\'’]?t|dont|not)\s+just|not\s+only|nicht\s+nur)\b",
    re.IGNORECASE,
)
_VISUAL_OUTPUT_NOUN_PATTERN = (
    r'(?:image(?:s)?|picture(?:s)?|photo(?:s)?|illustration(?:s)?|render(?:s)?|'
    r'variant(?:s)?|version(?:s)?|scene(?:s)?|bild(?:er)?|foto(?:s)?|'
    r'tierbild(?:er)?|haustierbild(?:er)?|tierfoto(?:s)?|haustierfoto(?:s)?|'
    r'tier[\s-]?selfie(?:s)?|haustier[\s-]?selfie(?:s)?|pet[\s-]?selfie(?:s)?|'
    r'animal[\s-]?selfie(?:s)?|selfie(?:s)?|'
    r'illustration(?:en)?|variante(?:n)?|version(?:en)?|szene(?:n)?|'
    r'bildidee(?:n)?|bildvariante(?:n)?|bildversion(?:en)?|bildkonzept(?:e)?|'
    r'schnittzeichnung(?:en)?|nachtaufnahme(?:n)?)'
)
_VISUAL_IDEA_OUTPUT_NOUN_PATTERN = r'(?:image\s+idea(?:s)?|visual\s+idea(?:s)?|bildidee(?:n)?)'
_VISUAL_OUTPUT_PLURAL_NOUN_PATTERN = (
    r'(?:images|pictures|photos|illustrations|renders|variants|versions|scenes|'
    r'bilder|fotos|illustrationen|varianten|versionen|szenen|'
    r'tierbilder|haustierbilder|tierfotos|haustierfotos|tier[\s-]?selfies|haustier[\s-]?selfies|pet[\s-]?selfies|'
    r'animal[\s-]?selfies|selfies|'
    r'bildideen|bildvarianten|bildversionen|bildkonzepte|schnittzeichnungen|nachtaufnahmen)'
)
_VISUAL_OUTPUT_DISTRIBUTIVE_SOURCE_NOUN_PATTERN = (
    r'(?:\w*ort(?:e)?|\w*place(?:s)?|\w*location(?:s)?|\w*scene(?:s)?|'
    r'\w*situation(?:s|en)?|\w*scenario(?:s)?|\w*szen(?:en|arien)|'
    r'\w*welt(?:en)?|\w*world(?:s)?|\w*idee(?:n)?|\w*idea(?:s)?|'
    r'\w*concept(?:s)?|\w*konzept(?:e)?|\w*tier(?:e)?|\w*animal(?:s)?|'
    r'\w*selfie(?:s)?|\w*portrait(?:s)?|\w*paragraph(?:s)?|'
    r'\w*section(?:s)?|\w*part(?:s)?|\w*teil(?:e)?|\w*verse(?:s)?|'
    r'\w*stanza(?:s)?|\w*strophe(?:n)?|\w*absatz(?:e|en)?|'
    r'\w*abschnitt(?:e|en)?)'
)
_VISUAL_OUTPUT_COUNT_RE = re.compile(
    rf'\b(?P<count>\d+|{"|".join(sorted(_VISUAL_OUTPUT_COUNT_WORDS.keys(), key=len, reverse=True))})\b'
    rf'(?:[\s,;:()/-]+\w+){{0,4}}[\s,;:()/-]+(?P<noun>{_VISUAL_OUTPUT_NOUN_PATTERN})\b',
    re.IGNORECASE,
)
_VISUAL_OUTPUT_SUFFIX_COUNT_RE = re.compile(
    rf'\b(?P<noun>{_VISUAL_OUTPUT_NOUN_PATTERN})\b'
    rf'\s*(?:[\[(]\s*(?P<count>\d+|{"|".join(sorted(_VISUAL_OUTPUT_COUNT_WORDS.keys(), key=len, reverse=True))})\s*[\])]|'
    rf'(?P<count_x>\d+)\s*[x×]?|'
    rf'[x×]\s*(?P<count_prefixed_x>\d+))(?=\W|$)',
    re.IGNORECASE,
)
_VISUAL_IDEA_OUTPUT_COUNT_RE = re.compile(
    rf'\b(?P<count>\d+|{"|".join(sorted(_VISUAL_OUTPUT_COUNT_WORDS.keys(), key=len, reverse=True))})\b'
    rf'(?:[\s,;:()/-]+\w+){{0,2}}[\s,;:()/-]+(?P<noun>{_VISUAL_IDEA_OUTPUT_NOUN_PATTERN})\b',
    re.IGNORECASE,
)
_VISUAL_OUTPUT_DISTRIBUTIVE_SOURCE_COUNT_RE = re.compile(
    rf'(?=(\b(?P<count>\d+|{"|".join(sorted(_VISUAL_OUTPUT_COUNT_WORDS.keys(), key=len, reverse=True))})\b'
    rf'(?:[\s,;:()/-]+\w+){{0,6}}[\s,;:()/-]+(?P<noun>{_VISUAL_OUTPUT_DISTRIBUTIVE_SOURCE_NOUN_PATTERN})\b))',
    re.IGNORECASE,
)
_VISUAL_OUTPUT_DISTRIBUTIVE_RE: list[re.Pattern[str]] = [
    re.compile(
        rf'\b(?:je|jeweils)\s+(?:ein|eine|einen|one|single)\b(?:[\s,;:()/-]+\w+){{0,2}}[\s,;:()/-]+{_VISUAL_OUTPUT_NOUN_PATTERN}\b',
        re.IGNORECASE,
    ),
    re.compile(
        rf'\b(?:for|fur|fuer)\s+(?:each|every|jedem|jeder|jeden|jede)\b(?:[\s,;:()/-]+\w+){{0,2}}[\s,;:()/-]+(?:an?|one|single|ein|eine|einen)\b(?:[\s,;:()/-]+\w+){{0,2}}[\s,;:()/-]+{_VISUAL_OUTPUT_NOUN_PATTERN}\b',
        re.IGNORECASE,
    ),
    re.compile(
        rf'\b(?:an?|one|single)\s+{_VISUAL_OUTPUT_NOUN_PATTERN}\b[\s\S]{{0,20}}\b(?:of|for)\s+(?:each|every|them|those)\b',
        re.IGNORECASE,
    ),
    re.compile(
        rf'\b(?:ein|eine|einen)\s+{_VISUAL_OUTPUT_NOUN_PATTERN}\b[\s\S]{{0,20}}\b(?:von|fur|fuer)\s+(?:jedem|jeder|allen|ihnen|davon)\b',
        re.IGNORECASE,
    ),
]
_VISUAL_OUTPUT_SOURCE_TRANSFER_RE: list[re.Pattern[str]] = [
    re.compile(
        rf'\b(?:davon|daraus|von ihnen|aus ihnen)\b[\s\S]{{0,24}}\b{_VISUAL_OUTPUT_PLURAL_NOUN_PATTERN}\b',
        re.IGNORECASE,
    ),
    re.compile(
        rf'\b(?:aus diesen|von diesen|mit diesen|from these|from them|using these)\b[\s\S]{{0,24}}\b{_VISUAL_OUTPUT_PLURAL_NOUN_PATTERN}\b',
        re.IGNORECASE,
    ),
    re.compile(
        rf'\b{_VISUAL_OUTPUT_PLURAL_NOUN_PATTERN}\b[\s\S]{{0,24}}\b(?:davon|daraus|von ihnen|aus ihnen)\b',
        re.IGNORECASE,
    ),
    re.compile(
        rf'\b{_VISUAL_OUTPUT_PLURAL_NOUN_PATTERN}\b[\s\S]{{0,24}}\b(?:aus diesen|von diesen|mit diesen|from these|from them|using these)\b',
        re.IGNORECASE,
    ),
    re.compile(
        rf'\b(?:of|from|for)\s+(?:them|those|these)\b[\s\S]{{0,24}}\b{_VISUAL_OUTPUT_PLURAL_NOUN_PATTERN}\b',
        re.IGNORECASE,
    ),
    re.compile(
        rf'\b{_VISUAL_OUTPUT_PLURAL_NOUN_PATTERN}\b[\s\S]{{0,24}}\b(?:of|from|for)\s+(?:them|those|these)\b',
        re.IGNORECASE,
    ),
]
_VISUAL_OUTPUT_BOTH_RESPONSE_RE = re.compile(
    r'\b(both|beide)\b[\s\S]{0,32}\b(response|reply|answer|antwort)\b',
    re.IGNORECASE,
)
_VISUAL_OUTPUT_RANGE_MIN_RE = re.compile(
    rf'\b(?:at least|minimum of|min(?:imum)?|mindestens|wenigstens)\s+(?P<count>\d+|{"|".join(sorted(_VISUAL_OUTPUT_COUNT_WORDS.keys(), key=len, reverse=True))})\b',
    re.IGNORECASE,
)
_VISUAL_OUTPUT_COUNT_TOKEN_PATTERN = r'\d+|' + '|'.join(
    sorted(_VISUAL_OUTPUT_COUNT_WORDS.keys(), key=len, reverse=True)
)
_LOCAL_VISUAL_ASSET_PAGE_CONTEXT_RE = re.compile(
    r'\b('
    r'landing[\s-]?page|landingpage|website|webseite|site|page|seite|html|css|hero|unterseite|'
    r'navigation|browser|startseite|suiten|suite|bildbereich(?:e|en)?|'
    r'image[\s-]?section(?:s)?|visual[\s-]?section(?:s)?|bild[\s-]?bereich(?:e|en)?'
    r')\b',
    re.IGNORECASE,
)
_LOCAL_VISUAL_ASSET_BINDING_RE = re.compile(
    r'\b(?:alle\s+)?(?:bild(?:er)?|foto(?:s)?|image(?:s)?|photo(?:s)?|picture(?:s)?|tierbild(?:er)?|haustierbild(?:er)?|tierfoto(?:s)?|haustierfoto(?:s)?)\b'
    r'[\s\S]{0,96}\b(?:lokal(?:e|en|er|es)?|local|locally)\b'
    r'[\s\S]{0,96}\b(?:generiert|generated|eingebunden|embedded|verlinkt|linked|gespeichert|saved)\b|'
    r'\b(?:lokal(?:e|en|er|es)?|local|locally)\b'
    r'[\s\S]{0,96}\b(?:generiert|generated|eingebunden|embedded|verlinkt|linked|gespeichert|saved)\b'
    r'[\s\S]{0,96}\b(?:bild(?:er)?|foto(?:s)?|image(?:s)?|photo(?:s)?|picture(?:s)?|tierbild(?:er)?|haustierbild(?:er)?|tierfoto(?:s)?|haustierfoto(?:s)?)\b|'
    r'\b(?:lokal(?:e|en|er|es)?|local|locally)\b[\s-]*(?:image|bild|photo|foto)[\s-]*(?:asset(?:s)?|datei(?:en)?|file(?:s)?)\b|'
    r'\b(?:html|css)\b[\s\S]{0,120}\b(?:bild(?:er)?|foto(?:s)?|image(?:s)?|photo(?:s)?|picture(?:s)?|tierbild(?:er)?|haustierbild(?:er)?|tierfoto(?:s)?|haustierfoto(?:s)?)\b'
    r'[\s\S]{0,96}\b(?:lokal(?:e|en|er|es)?|local|locally)\b[\s\S]{0,64}\b(?:artefakt(?:e|en)?|artifact(?:s)?)\b'
    r'[\s\S]{0,96}\b(?:zusammenpass(?:en)?|zusammen\s+passen|passen|fit|match|sauber|linked|verlinkt)\b',
    re.IGNORECASE,
)
_VISUAL_ASSET_NOUN_RE = re.compile(
    r'\b(?:bild[\s-]?assets?|image[\s-]?assets?|visual[\s-]?assets?|'
    r'photo[\s-]?assets?|bild[\s-]?datei(?:en)?|image[\s-]?file(?:s)?)\b',
    re.IGNORECASE,
)
_LOCAL_VISUAL_ASSET_CO_DELIVERY_RE = re.compile(
    rf'\b(?:all[\s-]?inclusive|alles\s+inklusive|inklusive|including|with|mit|samt|nebst|'
    rf'd\s*\.\s*h\s*\.)\b[\s\S]{{0,140}}\b{_VISUAL_OUTPUT_NOUN_PATTERN}\b|'
    rf'\b{_VISUAL_OUTPUT_NOUN_PATTERN}\b[\s\S]{{0,96}}\b(?:html|css|landing[\s-]?page|'
    rf'landingpage|website|webseite)\b',
    re.IGNORECASE,
)
_NO_EXTERNAL_VISUAL_ASSET_RE = re.compile(
    r'\b(?:keine|kein|no|without)\b[\s\S]{0,48}\b(?:externen?|external)\b'
    r'[\s\S]{0,48}\b(?:bild(?:er)?|foto(?:s)?|image(?:s)?|photo(?:s)?|picture(?:s)?)\b',
    re.IGNORECASE,
)
_STRUCTURAL_VISUAL_SECTION_COUNT_RE = re.compile(
    rf'\b(?P<count>{_VISUAL_OUTPUT_COUNT_TOKEN_PATTERN})\b'
    r'(?:[\s,;:()/-]+\w+){0,3}[\s,;:()/-]+'
    r'(?P<noun>bildbereich(?:e|en)?|bild[\s-]?bereich(?:e|en)?|'
    r'bildabschnitt(?:e|en)?|image[\s-]?section(?:s)?|'
    r'visual[\s-]?section(?:s)?|image[\s-]?area(?:s)?|'
    r'photo[\s-]?section(?:s)?)\b',
    re.IGNORECASE,
)
_STRUCTURAL_VISUAL_SECTION_AFTER_COUNT_RE = re.compile(
    r'(?:[\s,;:()/-]+\w+){0,3}[\s,;:()/-]+'
    r'(?P<noun>bildbereich(?:e|en)?|bild[\s-]?bereich(?:e|en)?|'
    r'bildabschnitt(?:e|en)?|image[\s-]?section(?:s)?|'
    r'visual[\s-]?section(?:s)?|image[\s-]?area(?:s)?|'
    r'photo[\s-]?section(?:s)?)\b',
    re.IGNORECASE,
)
_LOCAL_VISUAL_ASSET_SUBPAGE_RE = re.compile(
    r'\b(?:unterseite(?:n)?|subpage(?:s)?|second\s+page|zweite(?:n|r|s)?\s+seite|'
    r'zweite(?:n|r|s)?\s+unterseite|additional\s+page(?:s)?)\b',
    re.IGNORECASE,
)
_VISUAL_COUNT_TEXT_REVIEW_NOUN_RE = re.compile(
    r'\b('
    r'damage(?:s)?|issue(?:s)?|problem(?:s)?|point(?:s)?|finding(?:s)?|detail(?:s)?|'
    r'fact(?:s)?|note(?:s)?|observation(?:s)?|defect(?:s)?|reason(?:s)?|argument(?:s)?|'
    r'sentence(?:s)?|phrase(?:s)?|line(?:s)?|text(?:s)?|draft(?:s)?|candidate(?:s)?|'
    r'sch[aä]d(?:e|en)?|schaeden|schaden|problem(?:e)?|punkt(?:e)?|befund(?:e)?|'
    r'detail(?:s)?|merkmal(?:e)?|fehler|defekt(?:e)?|beobachtung(?:en)?|'
    r'satz(?:e|en)?|saetze|satze|zeile(?:n)?'
    r')\b',
    re.IGNORECASE,
)
_VISUAL_COUNT_AUDIO_VERSION_QUALIFIER_RE = re.compile(
    r'\b(?:spoken|audio|voice|voiceover|speech|tts|text[\s-]?to[\s-]?speech|'
    r'gesprochen\w*|sprach\w*|hoer\w*|hör\w*)\b',
    re.IGNORECASE,
)
_VISUAL_AMBIGUOUS_VERSION_NOUN_RE = re.compile(
    r'^(?:version(?:s)?|variant(?:s)?|version(?:en)?|variante(?:n)?)$',
    re.IGNORECASE,
)
_VISUAL_COUNT_LIST_BRIDGE_RE = re.compile(
    rf'\b(?:artifact(?:s)?|artefakt(?:e|en)?|output(?:s)?|ausgabe(?:n)?|item(?:s)?|element(?:e|en)?)\b'
    rf'\s*[:;]\s*(?:\d+|{"|".join(sorted(_VISUAL_OUTPUT_COUNT_WORDS.keys(), key=len, reverse=True))})\b',
    re.IGNORECASE,
)
_VISUAL_SELECTED_ONLY_OUTPUT_RE = re.compile(
    r'\b(?:generate|create|render|show|make|generier(?:e|en)?|erzeuge|erstelle|zeige)\b'
    r'[\s\S]{0,96}\b(?:only\s+(?:the\s+)?best|best\s+one|nur\s+(?:die|den|das)\s+beste(?:n|r|s)?(?:\s+davon)?)\b'
    r'[\s\S]{0,96}\b(?:image|picture|photo|illustration|bild|foto|illustration)\b|'
    r'\bnur\s+(?:die|den|das)\s+beste(?:n|r|s)?(?:\s+davon)?\b'
    r'[\s\S]{0,64}\b(?:als|as)?\s*(?:image|picture|photo|illustration|bild|foto|illustration)\b',
    re.IGNORECASE,
)
_VISUAL_OUTPUT_DELIVERABLE_CONTEXT_RE = re.compile(
    r'\b('
    r'need(?:s|ed)?|require(?:s|d)?|want(?:s|ed)?|asked|request(?:s|ed)?|'
    r'deliver(?:s|ed|able|ables|y)?|provide(?:s|d)?|return(?:s|ed)?|'
    r'include(?:s|d|ing)?|contain(?:s|ed|ing)?|with|plus|alongside|'
    r'output(?:s)?|artifact(?:s)?|artefact(?:s)?|asset(?:s)?|file(?:s)?|'
    r'final|end|website|site|page|landing\s+page|html|css|code|brief|'
    r'brauch(?:e|st|t|en)?|benotig(?:e|st|t|en)?|liefer(?:e|st|t|n)?|'
    r'gib|geben|datei(?:en)?|webseite|seite'
    r')\b|'
    r'\b(?:at\s+the\s+end|am\s+ende|code[\s-]*(?:file(?:s)?|datei(?:en)?))\b',
    re.IGNORECASE,
)
_VISUAL_OUTPUT_STRICT_COUNT_CONTEXT_RE = re.compile(
    r'\b(?:exactly|at\s+least|minimum(?:\s+of)?|min(?:imum)?|genau|mindestens|wenigstens)\b',
    re.IGNORECASE,
)
_VISUAL_OUTPUT_REVIEW_ONLY_CONTEXT_RE = re.compile(
    r'\b('
    r'describe|analy[sz]e|inspect|review|compare|classify|identify|read|ocr|'
    r'summari[sz]e|attached|uploaded|existing|source|reference|these|those|'
    r'beschreib(?:e|st|t|en)?|analysier(?:e|st|t|en)?|pruef(?:e|st|t|en)?|'
    r'pruf(?:e|st|t|en)?|vergleich(?:e|st|t|en)?|angehangt|hochgeladen|'
    r'vorhanden|quelle|referenz|diese(?:n|r|s|m)?'
    r')\b',
    re.IGNORECASE,
)
_META_EXPLANATION_OPENING_RE = re.compile(
    r'^\s*(?:hey\s+\w+[\s,.!?:-]*)?(?:'
    r'explain|walk me through|tell me|show me|'
    r'erklar(?:e| mir)?|erklar mir|zeig(?:e)? mir|beschreib(?:e)? mir|erlauter(?:e)? mir'
    r')\b',
    re.IGNORECASE,
)
_META_CAPABILITY_EXPLANATION_OPENING_RE = re.compile(
    r'^\s*(?:(?:can|could|would)\s+you\s+|please\s+)?(?:'
    r'explain|walk\s+me\s+through|tell\s+me\s+(?:about|how)|what\s+is|'
    r'how\s+(?:does|do|would|can)|why\s+(?:does|do|would|can)|discuss|'
    r'erkl[aä]r(?:e|\s+mir)?|f[uü]hr(?:e)?\s+mich\s+durch|'
    r'sag(?:e)?\s+mir\s+(?:etwas\s+über|wie)|was\s+ist|'
    r'wie\s+(?:funktioniert|funktionieren|würde|wuerde|kann)|'
    r'warum\s+(?:funktioniert|funktionieren|würde|wuerde|kann)|erl[aä]uter(?:e)?'
    r')\b',
    re.IGNORECASE,
)
_META_PROCESS_HINT_RE = re.compile(
    r'\b(?:'
    r'runtime|architektur|architecture|ablauf|flow|workflow|prozess|process|'
    r'phase graph|request lifecycle|lifecycle|how (?:it|you) works|wie du funktionierst|'
    r'funktioniert|freeze|late fill|planner|truth|begriff(?:e)?|term(?:s)?'
    r')\b',
    re.IGNORECASE,
)
_META_CAPABILITY_DISCUSSION_RE = re.compile(
    r'\b(?:image|visual|audio|speech|voice|text[\s-]?to[\s-]?speech|'
    r'speech[\s-]?to[\s-]?text|html|css|json|bild|audio|sprach)'
    r'[\s-]*(?:generation|synthesis|creation|routing|materialization|materialisation|'
    r'pipeline|workflow|artifact(?:s)?|artefact(?:s)?|erzeugung|generierung|'
    r'synthese|routing|materialisierung|artefakt(?:e|en)?)\b|'
    r'\b(?:how\s+to|why\s+(?:would|should|does)|wie\s+man|warum\s+(?:sollte|würde|wuerde))\b'
    r'[^.!?\n]{0,80}\b(?:create|generate|make|render|build|save|materiali[sz]e|'
    r'erstell(?:e|en)?|erzeug(?:e|en)?|generier(?:e|en)?|speicher(?:e|n)?|materialisier(?:e|en)?)\b'
    r'[^.!?\n]{0,80}\b(?:image|audio|file|artifact|artefact|html|css|json|'
    r'bild|datei|artefakt)\b',
    re.IGNORECASE,
)
_META_FOLLOW_UP_MATERIALIZATION_RE = re.compile(
    r'\b(?:and(?:\s+then)?|then|next|also|finally|afterwards|afterward|plus|'
    r'und(?:\s+dann)?|dann|danach|anschlie(?:ß|ss)end|zus[aä]tzlich)\b'
    r'[^.;!?\n]{0,80}\b(?:create|generate|make|produce|render|build|save|materiali[sz]e|'
    r'erstell(?:e|en)?|erzeug(?:e|en)?|generier(?:e|en)?|mach(?:e|en)?|'
    r'render(?:e|n)?|bau(?:e|en)?|speicher(?:e|n)?|materialisier(?:e|en)?)\b'
    r'[^.;!?\n]{0,80}\b(?:image|audio|file|artifact|artefact|html|css|json|'
    r'bild|datei|artefakt)\b',
    re.IGNORECASE,
)
_META_EXAMPLE_HINT_RE = re.compile(
    r'\b(?:'
    r'example|for example|beispiel|zum beispiel|konkreten beispiel|concrete example|'
    r'denselben ablauf|same flow|hypothetical|if (?:the|a|someone) user asks|'
    r'if i ask|wenn (?:der|ein) user|wenn ich dich frage|only actual runtime terms|'
    r'nur echte begriffe(?: aus der runtime)?'
    r')\b',
    re.IGNORECASE,
)
_META_EXAMPLE_QUOTE_RE = re.compile(r'[:]\s*["“„«].{8,}["”»]', re.IGNORECASE | re.DOTALL)

_TTS_POSITIVE_RULES: list[tuple[re.Pattern[str], int, str]] = [
    (re.compile(r'\b(/tts|/speak|tts:|speak:|read aloud|read this aloud|read it aloud|read this to me|read this for me|speak this|speak it|'
                r'read me aloud|read that aloud|read me this aloud|read this out loud|read it out loud|read aloud to me|'
                r'generate (?:me )?(?:an )?audio|create (?:an )?audio|make (?:an )?audio|make an audio file|voice version|'
                r'spoken version|narrate|voiceover audio|text to speech)\b', re.IGNORECASE), 5, 'direct_tts_request'),
    (re.compile(r'\b(tts[\s-]*audio|tts[\s-]*(?:artifact|artefact|artefakt)|audio[\s-]*(?:artifact|artefact|artefakt)|audioartefakt)\b', re.IGNORECASE), 5, 'direct_tts_artifact_request'),
    (re.compile(r"\b(read|say|speak)\b(?:\s+[\w'-]+){0,8}\s+(?:aloud|out loud)\b", re.IGNORECASE), 5, 'direct_tts_phrase_request'),
    (re.compile(r'\b(generier(?:e|en|st|t)?|erzeug(?:e|en|st|t)?|erstell(?:e|en|st|t)?|mach(?:e|en|st|t)?)\b[\s\S]{0,48}\b(audio|audio(?:\s|-)?datei|audio(?:\s|-)?fassung(?:en)?|audio(?:\s|-)?version(?:en)?|sprachversion|horversion|hoerversion)\b', re.IGNORECASE), 5, 'direct_tts_request_de'),
    (
        re.compile(
            r'\b(?:replace|ersetz(?:e|en|t|st)?)\b[^.;!?]{0,120}'
            r'\b(?:audio[\s-]?branch|audiozweig(?:e)?)\b[^.;!?]{0,80}'
            r'\b(?:with|by|durch)\b[^.;!?]{0,80}'
            r'\b(?:audio[\s-]?(?:version(?:s)?|variant(?:s)?|fassung(?:en)?))\b',
            re.IGNORECASE,
        ),
        5,
        'direct_audio_variant_replacement',
    ),
    (re.compile(r'\b(give|make|create|write|compose|send)\b[\s\S]{0,96}\b(something|anything)\s+(?:i\s+can\s+)?listen\s+to\b', re.IGNORECASE), 5, 'indirect_tts_listen_request'),
    (re.compile(r'\b(in )?(audio|sound) format\b', re.IGNORECASE), 3, 'audio_output_format_request'),
    (re.compile(r'\b(vorlesen|lies(?:\s+\w+){0,8}\s+vor|lese(?:\s+\w+){0,8}\s+vor|sprich(?:\s+\w+){0,8}\s+vor|vertonen)\b', re.IGNORECASE), 5, 'direct_tts_request_de'),
    (re.compile(r'\b(lee(?:\s+\w+){0,3}\s+en voz alta|archivo de audio|voz|narracion|narracion en voz|voz femenina)\b', re.IGNORECASE), 4, 'direct_tts_request_es'),
    (re.compile(r'\b(lire(?:\s+\w+){0,3}\s+a voix haute|fichier audio|voix feminine|voix masculine)\b', re.IGNORECASE), 4, 'direct_tts_request_fr'),
    (re.compile(r'\b(blind|blind friend|visually impaired|screen reader|sehbehindert|blinde?r?|cieg[oa])\b', re.IGNORECASE), 3, 'accessibility_audio_need'),
    (re.compile(r'\b(female voice|male voice|männliche stimme|mannliche stimme|maennliche stimme|weibliche stimme)\b', re.IGNORECASE), 5, 'voice_style_audio_request'),
]

_TTS_NEGATIVE_RULES: list[tuple[re.Pattern[str], int, str]] = [
    (re.compile(r'\b(transcribe|transcript|speech to text|audio to text|ocr)\b', re.IGNORECASE), 3, 'stt_conflict'),
]

_IMAGE_POSITIVE_RULES: list[tuple[re.Pattern[str], int, str]] = [
    (re.compile(r'(^|\b)(generate|create|make|produce|render|design|draw|illustrate|paint|sketch|visualize|visualise|turn)\b[\s\S]{0,80}\b(image(?:s)?|img|photo(?:s)?|picture(?:s)?|portrait(?:s)?|scene(?:s)?|shot(?:s)?|poster(?:s)?|wallpaper(?:s)?|logo(?:s)?|icon(?:s)?|banner(?:s)?|cover art|cover(?:s)?|thumbnail(?:s)?|sticker(?:s)?|meme(?:s)?|illustration(?:s)?|artwork(?:s)?|drawing(?:s)?|sketch(?:es)?|painting(?:s)?|avatar(?:s)?|headshot(?:s)?|selfie(?:s)?|comic(?:s)?|flyer(?:s)?|concept art)\b', re.IGNORECASE), 5, 'direct_visual_generation'),
    (re.compile(r'(^|\b)(generiere|generieren|erstell(?:e|en|st)?|erstelle|erstellen|erzeuge|erzeugen|mache|machen|rendere|rendern|zeichne|zeichnen|male|malen|gestalte|gestalten|visualisiere|visualisieren)\b[\s\S]{0,80}\b(bild(?:er|idee(?:n)?|variante(?:n)?|version(?:en)?|konzept(?:e)?)?|foto(?:s)?|tierbild(?:er)?|haustierbild(?:er)?|tierfoto(?:s)?|haustierfoto(?:s)?|tier[\s-]?selfie(?:s)?|haustier[\s-]?selfie(?:s)?|pet[\s-]?selfie(?:s)?|portrat|portrait|szene(?:n)?|aufnahme(?:n)?|nachtaufnahme(?:n)?|schnittzeichnung(?:en)?|poster|cover art|cover|wallpaper|logo|icon|illustration(?:en)?|kunstwerk(?:e)?|banner)\b', re.IGNORECASE), 5, 'direct_visual_generation_de'),
    (re.compile(r'\b(bild(?:er)?|foto(?:s)?|tierbild(?:er)?|haustierbild(?:er)?|tierfoto(?:s)?|haustierfoto(?:s)?|tier[\s-]?selfie(?:s)?|haustier[\s-]?selfie(?:s)?|pet[\s-]?selfie(?:s)?|illustration(?:en)?|szene(?:n)?|aufnahme(?:n)?|kunstwerk(?:e)?)\b[\s\S]{0,32}\b(erstelle|erstellen|erzeuge|erzeugen|mache|machen|generiere|generieren|zeichne|zeichnen|male|malen|gestalte|gestalten|visualisiere|visualisieren)\b', re.IGNORECASE), 4, 'visual_object_then_generation_de'),
    (re.compile(r'\b(image(?:s)?|photo(?:s)?|picture(?:s)?|illustration(?:s)?|scene(?:s)?|poster(?:s)?|wallpaper(?:s)?|logo(?:s)?|icon(?:s)?|banner(?:s)?|thumbnail(?:s)?)\b[\s\S]{0,32}\b(generate|create|make|produce|render|design|draw|illustrate|paint|sketch|visualize|visualise)\b', re.IGNORECASE), 4, 'visual_object_then_generation'),
    (re.compile(r'\b(image|picture|photo|illustration|visual)[\s-]*(?:artifact|artefact)|\bbild[\s-]*(?:artefakt|artifact)|\bbildartefakt\b', re.IGNORECASE), 8, 'direct_visual_artifact_request'),
    (re.compile(r'\b(?:materiali[sz]e|materialisiere|materialisieren)\b[\s\S]{0,160}\b(?:image\s+idea(?:s)?|visual\s+idea(?:s)?|bildidee(?:n)?)\b|\b(?:image\s+idea(?:s)?|visual\s+idea(?:s)?|bildidee(?:n)?)\b[\s\S]{0,160}\b(?:materiali[sz]e|materialisiere|materialisieren)\b', re.IGNORECASE), 8, 'materialized_visual_idea_request'),
    (re.compile(r'\b(?:also|additionally|zusatzlich|zusaetzlich|zusätzlich)\b[\s\S]{0,64}\b(?:an?|ein(?:e|en|es)?)?\s*(?:posterbild|poster|bild|image|picture|illustration)\b', re.IGNORECASE), 7, 'additive_visual_output_request'),
    (re.compile(r'(^|\b)(zeichne|male)\b[\s\S]{0,48}\b(mir|bitte|doch|mal|schnell|einen?|eine|den|die|das)\b', re.IGNORECASE), 4, 'direct_visual_creation_de'),
    (re.compile(r'\b(make|turn|convert|transform)\b[\s\S]{0,40}\b(into|as)\b[\s\S]{0,30}\b(poster|cover art|wallpaper|logo|illustration|thumbnail)\b', re.IGNORECASE), 4, 'indirect_visual_output'),
    (re.compile(r'\b(mach|wandle|konvertiere|transformiere)\b[\s\S]{0,40}\b(als|zu|daraus)\b[\s\S]{0,30}\b(poster|cover art|wallpaper|logo|illustration)\b', re.IGNORECASE), 4, 'indirect_visual_output_de'),
    (re.compile(r'\b(place|building|scene|landscape|room|house|village|city|cove|shore|forest|temple|castle)\b[\s\S]{0,96}\b(show (?:it|them|this|that) to me|show me)\b', re.IGNORECASE), 4, 'indirect_visual_show_request'),
    (re.compile(r'\b(show (?:it|them|this|that) to me|show me)\b[\s\S]{0,80}\b(moonlight|moonlit|at night|night|daytime|sunset|sunrise|lighting)\b', re.IGNORECASE), 4, 'visual_lighting_show_request'),
    (re.compile(r'\b(show|display|visuali[sz]e|illustrate|render)\b[\s\S]{0,32}\b(?:an?|the)\b[\s\S]{0,8}\b(image|picture|photo|illustration|scene)\b[\s\S]{0,24}\bof\b[\s\S]{0,12}\b(it|them|this|that|me|him|her)\b', re.IGNORECASE), 4, 'visual_output_of_reference'),
    (re.compile(r'\b(visualize|visualise|cover art|poster|wallpaper|logo|illustration|thumbnail)\b', re.IGNORECASE), 2, 'visual_output_noun'),
]

_IMAGE_NEGATIVE_RULES: list[tuple[re.Pattern[str], int, str]] = [
    (re.compile(r'\b(summary|poem|story|essay|email|letter|article|markdown|json|yaml|audio|speech|voiceover|tts|transcribe|transcript|ocr|translation|translate)\b', re.IGNORECASE), 3, 'text_or_audio_conflict'),
]

_VISION_POSITIVE_RULES: list[tuple[re.Pattern[str], int, str]] = [
    (re.compile(r'\b(describe this image|describe the image|what is in this image|what\'s in this image|analyze this image|read the text in this image|ocr|screenshot)\b', re.IGNORECASE), 5, 'direct_vision_request'),
    (re.compile(r'\b(beschreibe dieses bild|was ist auf diesem bild|lies den text aus diesem bild|ocr)\b', re.IGNORECASE), 5, 'direct_vision_request_de'),
]

_STT_POSITIVE_RULES: list[tuple[re.Pattern[str], int, str]] = [
    (re.compile(r'\b(transcribe|transcription|speech to text|audio to text|transcript this)\b', re.IGNORECASE), 5, 'direct_stt_request'),
    (re.compile(r'\b(transkribiere|transkript|sprache zu text|audio zu text)\b', re.IGNORECASE), 5, 'direct_stt_request_de'),
    (
        re.compile(
            r'\b(?:analy[sz]e|analyse|analysier|review|assess|inspect|pruef|prüf|bewert)\w*\b'
            r'[\s\S]{0,120}\b(?:audio|aufnahme|recording|spoken\s+version|gesprochene(?:n|r|s)?\s+version|voice|stimme|tonlage)\b|'
            r'\b(?:audio|aufnahme|recording|spoken\s+version|gesprochene(?:n|r|s)?\s+version|voice|stimme|tonlage)\b'
            r'[\s\S]{0,120}\b(?:analy[sz]e|analyse|analysier|review|assess|inspect|pruef|prüf|bewert)\w*\b',
            re.IGNORECASE,
        ),
        5,
        'audio_artifact_analysis_request',
    ),
]


def normalize_intent_text(text: Any) -> str:
    normalized = unicodedata.normalize('NFKD', str(text or ''))
    normalized = ''.join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.lower()
    for pattern, replacement in _INTENT_PROMPT_NORMALIZATIONS:
        normalized = pattern.sub(replacement, normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized


def _materialization_negation_scope(text: str, start: int, end: int) -> str:
    prompt = str(text or '')
    start_index = max(0, int(start or 0))
    end_index = min(len(prompt), max(start_index, int(end or start_index)))
    lower = max(0, start_index - 180)
    upper = min(len(prompt), end_index + 180)
    for boundary in ('.', '!', '?', ';', '\n'):
        boundary_index = prompt.rfind(boundary, lower, start_index)
        if boundary_index >= 0:
            lower = max(lower, boundary_index + 1)
    next_boundaries = [
        index
        for boundary in ('.', '!', '?', ';', '\n')
        for index in [prompt.find(boundary, end_index, upper)]
        if index >= 0
    ]
    if next_boundaries:
        upper = min(upper, min(next_boundaries) + 1)
    return prompt[lower:upper].strip()


def prompt_negates_artifact_fulfillment_only(
    prompt: Any,
    normalized_prompt: Any | None = None,
) -> bool:
    """Return true when negation says preparation does not fulfill an artifact."""

    raw_prompt = str(prompt or '').strip()
    normalized = str(normalized_prompt or '').strip() if normalized_prompt is not None else ''
    if raw_prompt and not normalized:
        normalized = normalize_intent_text(raw_prompt)
    candidates = [item for item in (raw_prompt, normalized) if item]
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if any(pattern.search(candidate) for pattern in _ARTIFACT_FULFILLMENT_NEGATION_RE):
            return True
    return False


def materialization_negation_match_is_artifact_fulfillment_only(
    prompt: Any,
    start: int,
    end: int,
) -> bool:
    prompt_text = str(prompt or '')
    if not prompt_text:
        return False
    scope = _materialization_negation_scope(prompt_text, start, end)
    if not scope:
        return False
    return prompt_negates_artifact_fulfillment_only(scope)


def _materialization_negation_match_is_cardinality_constraint(
    prompt: Any,
    start: int,
    end: int,
) -> bool:
    prompt_text = str(prompt or '')
    if not prompt_text:
        return False
    scope = _materialization_negation_scope(prompt_text, start, end)
    if not scope:
        return False
    return bool(_MATERIALIZATION_CARDINALITY_CONSTRAINT_RE.search(scope))


def materialization_negation_match_is_output_contrast(
    prompt: Any,
    start: int,
    end: int,
) -> bool:
    prompt_text = str(prompt or '')
    if not prompt_text:
        return False
    scope = _materialization_negation_scope(prompt_text, start, end)
    if not scope:
        return False
    return bool(_MATERIALIZATION_ONLY_OUTPUT_CONTRAST_RE.search(scope))


def _score_rules(
    normalized_text: str,
    rules: list[tuple[re.Pattern[str], int, str]],
) -> tuple[int, list[str]]:
    score = 0
    cues: list[str] = []
    for pattern, weight, cue in rules:
        if pattern.search(normalized_text):
            score += weight
            if cue not in cues:
                cues.append(cue)
    return score, cues


def infer_prompt_languages(prompt: str) -> list[str]:
    normalized = normalize_intent_text(prompt)
    languages: list[str] = []
    for pattern, code in _LANGUAGE_CODE_HINTS:
        if pattern.search(normalized) and code not in languages:
            languages.append(code)
    for pattern, code in _LANGUAGE_HINTS:
        if pattern.search(normalized) and code not in languages:
            languages.append(code)
    return languages


def infer_voice_descriptors(prompt: str) -> list[str]:
    normalized = normalize_intent_text(prompt)
    descriptors: list[str] = []
    for pattern, descriptor in _VOICE_STYLE_HINTS:
        if pattern.search(normalized) and descriptor not in descriptors:
            descriptors.append(descriptor)
    return descriptors


def infer_audio_response_format(prompt: str) -> Optional[str]:
    normalized = normalize_intent_text(prompt)
    for pattern, fmt in _AUDIO_FORMAT_HINTS:
        if pattern.search(normalized):
            return fmt
    return None


def infer_image_aspect_ratio(prompt: str) -> Optional[str]:
    normalized = normalize_intent_text(prompt)
    explicit = _EXPLICIT_ASPECT_RE.search(normalized)
    if explicit:
        return explicit.group(1)
    for pattern, ratio in _ASPECT_HINTS:
        if pattern.search(normalized):
            return ratio
    return None


def infer_temperament_hint(prompt: str) -> tuple[Optional[str], list[str]]:
    normalized = normalize_intent_text(prompt)
    scores: dict[str, int] = {}
    cues_by_mode: dict[str, list[str]] = {}
    for pattern, mode, weight, cue in _TEMPERAMENT_RULES:
        if not pattern.search(normalized):
            continue
        scores[mode] = scores.get(mode, 0) + int(weight)
        mode_cues = cues_by_mode.setdefault(mode, [])
        if cue not in mode_cues:
            mode_cues.append(cue)
    if not scores:
        return None, []
    ranked = sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)
    mode, score = ranked[0]
    if score < 3:
        return None, []
    return mode, cues_by_mode.get(mode, [])


def _visual_output_count_from_token(token: str) -> int:
    normalized_token = str(token or '').strip().lower()
    if not normalized_token:
        return 0
    if normalized_token.isdigit():
        parsed = int(normalized_token)
        return parsed if parsed > 0 else 0
    return int(_VISUAL_OUTPUT_COUNT_WORDS.get(normalized_token, 0))


def _audio_output_count_from_token(token: str) -> int:
    normalized_token = str(token or '').strip().lower()
    if not normalized_token:
        return 0
    if normalized_token.isdigit():
        parsed = int(normalized_token)
        return parsed if parsed > 0 else 0
    return int(_AUDIO_OUTPUT_COUNT_WORDS.get(normalized_token, 0))


def intent_span_is_literal_payload(text: str, start: int, end: int) -> bool:
    """Return whether a matched intent span is data inside a quote or code fence."""

    prompt = str(text or '')
    bounded_start = max(0, int(start or 0))
    bounded_end = min(len(prompt), max(bounded_start, int(end or bounded_start)))
    for opening, closing in (
        ('```', '```'),
        ('"', '"'),
        ('“', '”'),
        ('„', '“'),
        ('«', '»'),
        ('‹', '›'),
        ('`', '`'),
    ):
        opening_index = prompt.rfind(opening, 0, bounded_start + 1)
        if opening_index < 0:
            continue
        if opening == closing and prompt[:bounded_start].count(opening) % 2 == 0:
            continue
        closing_index = prompt.find(closing, max(opening_index + len(opening), bounded_end))
        if closing_index >= bounded_end:
            return True
    return any(
        match.start() < bounded_start and bounded_end < match.end()
        for match in re.finditer(r"(?<![\\\w])'[^'\n]*'(?!\w)", prompt)
    )


def _intent_span_is_quoted(text: str, start: int, end: int) -> bool:
    return intent_span_is_literal_payload(text, start, end)


def _mask_literal_payloads(text: str) -> str:
    """Preserve offsets while removing quoted/fenced data from command analysis."""

    source = str(text or '')
    masked = list(source)
    spans = sorted(
        {
            (match.start(), match.end())
            for pattern in _LITERAL_PAYLOAD_MASK_PATTERNS
            for match in pattern.finditer(source)
        }
    )
    spoken_source = resolve_explicit_tts_source(source)
    if spoken_source:
        spans.append(
            (
                int(spoken_source.get('start') or 0),
                int(spoken_source.get('end') or 0),
            )
        )
    for start, end in spans:
        masked[start:end] = ' ' * (end - start)
    return ''.join(masked)


def mask_intent_literal_payloads(text: str) -> str:
    """Expose the offset-preserving literal mask to phase-graph consumers."""

    return _mask_literal_payloads(text)


def _answer_as_audio_delivery_state(
    normalized_prompt: str,
) -> tuple[tuple[tuple[int, int], ...], bool, bool]:
    """Return active delivery spans plus narrow negation/defer state."""

    prompt = str(normalized_prompt or '').strip()
    if not prompt:
        return (), False, False
    active_spans: list[tuple[int, int]] = []
    saw_negated = False
    saw_deferred = False
    for match in _ANSWER_AS_AUDIO_DELIVERY_RE.finditer(prompt):
        if _intent_span_is_quoted(prompt, match.start(), match.end()):
            continue
        clause = _materialization_negation_scope(prompt, match.start(), match.end())
        clause_start = max(
            [prompt.rfind(boundary, 0, match.start()) for boundary in ('.', ';', '!', '?', '\n')]
            + [-1]
        ) + 1
        pre_action = prompt[clause_start:match.start()]
        before_audio = prompt[match.start():match.start('audio_target')]
        if _ANSWER_AS_AUDIO_DEFER_RE.search(before_audio):
            saw_deferred = True
            continue
        post_target = prompt[match.end('response_target'):match.start('audio_target')]
        pre_target = prompt[match.end('delivery_action'):match.start('response_target')]
        has_positive_contrast = bool(_ANSWER_AS_AUDIO_POSITIVE_CONTRAST_RE.search(clause))
        if (
            not has_positive_contrast
            and (
                _ANSWER_AS_AUDIO_PRE_ACTION_NEGATION_RE.search(pre_action)
                or _ANSWER_AS_AUDIO_POST_TARGET_NEGATION_RE.search(post_target)
                or _ANSWER_AS_AUDIO_PRE_TARGET_NEGATION_RE.search(pre_target)
            )
        ):
            saw_negated = True
            continue
        active_spans.append((match.start(), match.end()))
    if active_spans:
        return tuple(active_spans), False, False
    return (), saw_negated, saw_deferred


def _intent_spans_share_clause(
    normalized_prompt: str,
    first: tuple[int, int],
    second: tuple[int, int],
) -> bool:
    first_start, first_end = first
    second_start, second_end = second
    if first_start <= second_start and second_end <= first_end:
        return True
    if second_start <= first_start and first_end <= second_end:
        return True
    between_start = min(first_end, second_end)
    between_end = max(first_start, second_start)
    if between_start > between_end:
        return True
    return not re.search(r'[.;!?\n]', str(normalized_prompt or '')[between_start:between_end])


def _audio_output_count_match_is_deliverable(
    normalized_prompt: str,
    match: re.Match[str],
    *,
    answer_as_audio_delivery_spans: tuple[tuple[int, int], ...] = (),
) -> bool:
    count_start = match.start('count') if 'count' in match.re.groupindex else -1
    count_end = match.end('count') if count_start >= 0 else -1
    noun_start = match.start('noun') if 'noun' in match.re.groupindex else -1
    if 0 <= count_end <= noun_start:
        between = str(normalized_prompt or '')[count_end:noun_start]
        if _AUDIO_COUNT_INTERVENING_NON_AUDIO_NOUN_RE.search(between):
            return False
    scope = _materialization_negation_scope(
        normalized_prompt,
        match.start(),
        match.end(),
    )
    answer_delivery_in_clause = any(
        _intent_spans_share_clause(
            normalized_prompt,
            delivery_span,
            (match.start(), match.end()),
        )
        for delivery_span in answer_as_audio_delivery_spans
    )
    if not scope or not (
        _AUDIO_OUTPUT_DELIVERABLE_ACTION_RE.search(scope)
        or answer_delivery_in_clause
    ):
        return False
    return not any(pattern.search(scope) for pattern in _AUDIO_OUTPUT_NEGATION_HINTS)


def _infer_requested_audio_output_count(
    normalized_prompt: str,
    *,
    answer_as_audio_delivery_spans: tuple[tuple[int, int], ...] = (),
) -> tuple[int, bool, bool, int]:
    """Return bounded audio count, obligation, overflow audit, and raw requested count."""

    prompt = str(normalized_prompt or '').strip()
    if not prompt:
        return 0, False, False, 0
    counts: list[int] = []
    raw_counts: list[int] = []
    overflow = False
    for match in _AUDIO_DISTRIBUTED_LANGUAGE_OUTPUT_RE.finditer(prompt):
        if not _audio_output_count_match_is_deliverable(
            prompt,
            match,
            answer_as_audio_delivery_spans=answer_as_audio_delivery_spans,
        ):
            continue
        counts.append(2)
        raw_counts.append(2)
    for match in _AUDIO_OUTPUT_COUNT_RE.finditer(prompt):
        if not _audio_output_count_match_is_deliverable(
            prompt,
            match,
            answer_as_audio_delivery_spans=answer_as_audio_delivery_spans,
        ):
            continue
        count = _audio_output_count_from_token(match.group('count'))
        if count > 0:
            raw_counts.append(count)
        if count > _MAX_REQUESTED_AUDIO_OUTPUT_COUNT:
            overflow = True
            continue
        if count > 0:
            counts.append(count)
    for match in _AUDIO_OUTPUT_SUFFIX_COUNT_RE.finditer(prompt):
        if not _audio_output_count_match_is_deliverable(
            prompt,
            match,
            answer_as_audio_delivery_spans=answer_as_audio_delivery_spans,
        ):
            continue
        count = max(
            _audio_output_count_from_token(match.group('count') or ''),
            _audio_output_count_from_token(match.group('count_x') or ''),
            _audio_output_count_from_token(match.group('count_prefixed_x') or ''),
        )
        if count > 0:
            raw_counts.append(count)
        if count > _MAX_REQUESTED_AUDIO_OUTPUT_COUNT:
            overflow = True
            continue
        if count > 0:
            counts.append(count)
    if overflow:
        return 0, False, True, max(raw_counts, default=0)
    requested_count = max(counts) if counts else 0
    return requested_count, requested_count > 0, False, requested_count


def _visual_preservation_flags(normalized_prompt: str) -> tuple[bool, bool, list[str]]:
    prompt = str(normalized_prompt or '').strip()
    preservation_match = _VISUAL_PRESERVATION_RE.search(prompt) if prompt else None
    if not preservation_match:
        return False, False, []
    preserve_artifact = bool(_VISUAL_NO_REGENERATION_RE.search(prompt))
    preserved_scope = preservation_match.group(0)
    preserve_analysis = bool(
        _VISUAL_NO_REANALYSIS_RE.search(prompt)
        or re.search(r'\b(?:bildanalyse|image\s+analysis|visual\s+analysis)\b', preserved_scope)
    )
    cues: list[str] = []
    if preserve_artifact:
        cues.append('visual_artifact_preservation_without_regeneration')
    if preserve_analysis:
        cues.append('visual_analysis_preservation_without_reanalysis')
    return preserve_artifact, preserve_analysis, cues


def _visual_action_window(prompt: str, start: int, *, limit: int = 180) -> str:
    tail = str(prompt or '')[max(0, start):max(0, start) + max(1, limit)]
    boundary = re.search(r'[.;!?\n]', tail)
    return tail[:boundary.start()] if boundary else tail


def visual_action_is_negated(prompt: str, start: int, end: int) -> bool:
    """Evaluate an action's polarity without dropping its governing clause prefix."""

    text = str(prompt or '')
    action_start = max(0, int(start or 0))
    action_end = min(len(text), max(action_start, int(end or action_start)))
    lower = max(0, action_start - 180)
    upper = min(len(text), action_end + 180)
    for boundary in ('.', '!', '?', ';', '\n'):
        boundary_index = text.rfind(boundary, lower, action_start)
        if boundary_index >= 0:
            lower = max(lower, boundary_index + 1)
    next_boundaries = [
        index
        for boundary in ('.', '!', '?', ';', '\n')
        for index in [text.find(boundary, action_end, upper)]
        if index >= 0
    ]
    if next_boundaries:
        upper = min(upper, min(next_boundaries))
    for reset in _VISUAL_ACTION_POLARITY_RESET_RE.finditer(text, lower, upper):
        if reset.end() <= action_start:
            lower = reset.end()
        elif reset.start() >= action_end:
            upper = reset.start()
            break

    prefix = text[lower:action_start]
    # "not only/don't just create" is additive, not a materialization negation.
    prefix = re.sub(
        r"\b(?:(?:do\s+not|don[\'’]?t|dont|not)\s+just|not\s+only|nicht\s+nur)\b",
        '',
        prefix,
        flags=re.IGNORECASE,
    )
    if _VISUAL_ACTION_LEADING_NEGATION_RE.search(prefix):
        return True
    if _VISUAL_ACTION_NEGATION_RE.search(text[action_start:action_end]):
        return True
    return bool(_VISUAL_ACTION_TRAILING_NEGATION_RE.search(text[action_end:upper]))


def _latest_affirmative_visual_generation_action_start(prompt: str) -> int:
    """Return the last executable image-creation action offset, or ``-1``."""

    text = str(prompt or '')
    latest_start = -1
    for action in _VISUAL_GENERATION_ACTION_RE.finditer(text):
        if intent_span_is_literal_payload(text, action.start(), action.end()):
            continue
        window = _visual_action_window(text, action.start())
        target = _VISUAL_ACTION_TARGET_RE.search(window)
        if not target:
            continue
        if not visual_action_is_negated(
            text,
            action.start(),
            action.start() + target.end(),
        ):
            latest_start = action.start()
    return latest_start


def _has_affirmative_audio_materialization_action(prompt: str) -> bool:
    """Return whether the current command scope contains executable TTS work."""

    text = str(prompt or '')
    return any(
        not intent_span_is_literal_payload(text, match.start(), match.end())
        and not visual_action_is_negated(text, match.start(), match.end())
        for match in _AUDIO_MATERIALIZATION_ACTION_RE.finditer(text)
    )


def _audio_negation_match_governs_negated_action(
    prompt: str,
    match: re.Match[str],
) -> bool:
    """Keep broad negation matches only when they govern an audio action or noun."""

    text = str(prompt or '')
    overlapping_actions = [
        action
        for action in _AUDIO_MATERIALIZATION_ACTION_RE.finditer(text)
        if action.start() < match.end() and match.start() < action.end()
    ]
    if overlapping_actions:
        return any(
            visual_action_is_negated(text, action.start(), action.end())
            for action in overlapping_actions
        )
    return bool(_AUDIO_NEGATION_TARGET_RE.search(match.group(0)))


def _separate_visual_work_flags(normalized_prompt: str) -> tuple[bool, bool, list[str]]:
    """Identify affirmative visual work aimed at a new/separate artifact scope."""

    prompt = str(normalized_prompt or '').strip()
    if not prompt:
        return False, False, []

    separate_generation = False
    for action in _VISUAL_GENERATION_ACTION_RE.finditer(prompt):
        if intent_span_is_literal_payload(prompt, action.start(), action.end()):
            continue
        window = _visual_action_window(prompt, action.start())
        target = _VISUAL_ACTION_TARGET_RE.search(window)
        if not target:
            continue
        target_scope = window[:min(len(window), target.end() + 32)]
        if visual_action_is_negated(prompt, action.start(), action.start() + target.end()):
            continue
        counted_target = any(
            _visual_output_count_from_token(match.group('count')) > 0
            for match in _VISUAL_OUTPUT_COUNT_RE.finditer(target_scope)
        )
        if _VISUAL_SEPARATE_TARGET_RE.search(target_scope) or counted_target:
            separate_generation = True
            break

    separate_analysis = False
    for action in _VISUAL_ANALYSIS_ACTION_RE.finditer(prompt):
        if intent_span_is_literal_payload(prompt, action.start(), action.end()):
            continue
        window = _visual_action_window(prompt, action.start())
        target = _VISUAL_ACTION_TARGET_RE.search(window)
        target_end = target.end() if target else min(len(window), 64)
        target_scope = window[:min(len(window), target_end + 32)]
        if visual_action_is_negated(prompt, action.start(), action.start() + target_end):
            continue
        if target and _VISUAL_SEPARATE_TARGET_RE.search(target_scope):
            separate_analysis = True
            break
        if separate_generation and _VISUAL_PLURAL_RESULT_REFERENCE_RE.search(target_scope):
            separate_analysis = True
            break

    cues: list[str] = []
    if separate_generation:
        cues.append('separate_visual_generation_request')
    if separate_analysis:
        cues.append('separate_visual_analysis_request')
    return separate_generation, separate_analysis, cues


def _visual_output_count_match_is_text_review_count(normalized_prompt: str, match: re.Match[str]) -> bool:
    between = str(normalized_prompt or '')[match.end('count'):match.start('noun')]
    noun = str(match.group('noun') or '').strip()
    return bool(
        _VISUAL_COUNT_TEXT_REVIEW_NOUN_RE.search(between)
        or _VISUAL_COUNT_LIST_BRIDGE_RE.search(between)
        or (
            _VISUAL_AMBIGUOUS_VERSION_NOUN_RE.fullmatch(noun)
            and _VISUAL_COUNT_AUDIO_VERSION_QUALIFIER_RE.search(between)
        )
    )


def _visual_output_count_match_spans(normalized_prompt: str) -> list[tuple[int, int]]:
    prompt = str(normalized_prompt or '').strip()
    spans: list[tuple[int, int]] = []
    if not prompt:
        return spans
    for match in _VISUAL_OUTPUT_COUNT_RE.finditer(prompt):
        if _visual_output_count_match_is_text_review_count(prompt, match):
            continue
        count = _visual_output_count_from_token(match.group('count'))
        if count > 0:
            spans.append((match.start(), match.end()))
    for match in _VISUAL_OUTPUT_SUFFIX_COUNT_RE.finditer(prompt):
        count = max(
            _visual_output_count_from_token(match.group('count') or ''),
            _visual_output_count_from_token(match.group('count_x') or ''),
            _visual_output_count_from_token(match.group('count_prefixed_x') or ''),
        )
        if count > 0:
            spans.append((match.start(), match.end()))
    return spans


def _has_counted_visual_output_obligation(normalized_prompt: str, requested_count: int) -> bool:
    prompt = str(normalized_prompt or '').strip()
    if not prompt or requested_count <= 0:
        return False
    for start, end in _visual_output_count_match_spans(prompt):
        lower = max(0, start - 160)
        upper = min(len(prompt), end + 220)
        context = prompt[lower:upper]
        has_review_only_context = bool(_VISUAL_OUTPUT_REVIEW_ONLY_CONTEXT_RE.search(context))
        has_deliverable_context = bool(_VISUAL_OUTPUT_DELIVERABLE_CONTEXT_RE.search(context))
        has_strict_count_context = bool(_VISUAL_OUTPUT_STRICT_COUNT_CONTEXT_RE.search(context))
        suffix = prompt[end:min(len(prompt), end + 8)].lstrip()
        heading_like_count = suffix.startswith(':')
        if has_review_only_context and not has_deliverable_context:
            continue
        if has_deliverable_context or has_strict_count_context or heading_like_count:
            return True
    return False


def _infer_requested_visual_output_count(normalized_prompt: str) -> int:
    prompt = str(normalized_prompt or '').strip()
    if not prompt:
        return 0
    if _VISUAL_SELECTED_ONLY_OUTPUT_RE.search(prompt):
        return 1
    direct_counts: list[int] = []
    for match in _VISUAL_IDEA_OUTPUT_COUNT_RE.finditer(prompt):
        count = _visual_output_count_from_token(match.group('count'))
        if count > 0:
            direct_counts.append(count)
    for match in _VISUAL_OUTPUT_COUNT_RE.finditer(prompt):
        if _visual_output_count_match_is_text_review_count(prompt, match):
            continue
        count = _visual_output_count_from_token(match.group('count'))
        if count > 0:
            direct_counts.append(count)
    for match in _VISUAL_OUTPUT_SUFFIX_COUNT_RE.finditer(prompt):
        count = max(
            _visual_output_count_from_token(match.group('count') or ''),
            _visual_output_count_from_token(match.group('count_x') or ''),
            _visual_output_count_from_token(match.group('count_prefixed_x') or ''),
        )
        if count > 0:
            direct_counts.append(count)
    source_counts = [
        _visual_output_count_from_token(match.group('count'))
        for match in _VISUAL_OUTPUT_DISTRIBUTIVE_SOURCE_COUNT_RE.finditer(prompt)
        if _visual_output_count_from_token(match.group('count')) > 1
    ]
    distributive_count = 0
    transferred_source_count = max(source_counts) if source_counts and any(
        pattern.search(prompt) for pattern in _VISUAL_OUTPUT_SOURCE_TRANSFER_RE
    ) else 0
    distributive_starts = [
        match.start()
        for pattern in _VISUAL_OUTPUT_DISTRIBUTIVE_RE
        for match in [pattern.search(prompt)]
        if match
    ]
    if distributive_starts:
        distributive_prefix = prompt[: min(distributive_starts)]
        distributive_counts = [
            _visual_output_count_from_token(match.group('count'))
            for match in _VISUAL_OUTPUT_DISTRIBUTIVE_SOURCE_COUNT_RE.finditer(distributive_prefix)
            if _visual_output_count_from_token(match.group('count')) > 1
        ]
        if distributive_counts:
            distributive_count = max(distributive_counts)
    if direct_counts:
        explicit_direct_count = max(direct_counts)
        if explicit_direct_count > 1:
            return explicit_direct_count
        if transferred_source_count > explicit_direct_count:
            return transferred_source_count
        if distributive_count > explicit_direct_count:
            return distributive_count
        return explicit_direct_count
    if transferred_source_count > 0:
        return transferred_source_count
    if distributive_count > 0:
        return distributive_count
    range_match = _VISUAL_OUTPUT_RANGE_MIN_RE.search(prompt)
    if range_match:
        count = _visual_output_count_from_token(range_match.group('count'))
        if count > 0:
            return count
    if _VISUAL_OUTPUT_BOTH_RESPONSE_RE.search(prompt):
        return 2
    return 0


def _local_visual_asset_requirement_cues(normalized_prompt: str) -> list[str]:
    prompt = str(normalized_prompt or '').strip()
    if not prompt:
        return []
    if not _LOCAL_VISUAL_ASSET_PAGE_CONTEXT_RE.search(prompt):
        return []
    cues: list[str] = []
    if _LOCAL_VISUAL_ASSET_BINDING_RE.search(prompt):
        cues.append('local_visual_asset_binding_requirement')
    if (
        _VISUAL_ASSET_NOUN_RE.search(prompt)
        or _LOCAL_VISUAL_ASSET_CO_DELIVERY_RE.search(prompt)
    ):
        cues.append('visual_asset_page_requirement')
    if _NO_EXTERNAL_VISUAL_ASSET_RE.search(prompt):
        cues.append('no_external_visual_asset_requirement')
    return cues


def _infer_local_visual_asset_output_count(normalized_prompt: str) -> tuple[int, str]:
    prompt = str(normalized_prompt or '').strip()
    if not prompt:
        return 0, ''
    counts: list[int] = []
    count_token_re = re.compile(rf'\b(?:{_VISUAL_OUTPUT_COUNT_TOKEN_PATTERN})\b', re.IGNORECASE)
    for match in count_token_re.finditer(prompt):
        tail = prompt[match.end():match.end() + 96]
        noun_match = _STRUCTURAL_VISUAL_SECTION_AFTER_COUNT_RE.search(tail)
        if not noun_match:
            continue
        next_count = count_token_re.search(tail)
        if next_count and next_count.start() < noun_match.start('noun'):
            continue
        count = _visual_output_count_from_token(match.group(0))
        if count > 0:
            counts.append(count)
    if counts and _LOCAL_VISUAL_ASSET_SUBPAGE_RE.search(prompt) and _VISUAL_ASSET_NOUN_RE.search(prompt):
        return max(counts) + 1, 'structural_visual_sections_plus_subpage_assets'
    if counts:
        return max(counts), 'structural_visual_sections'
    return 0, ''


def _is_meta_execution_explanation_request(prompt: str, normalized_prompt: str) -> bool:
    raw_prompt = str(prompt or '').strip()
    normalized = str(normalized_prompt or '').strip()
    if not raw_prompt or not normalized:
        return False
    standard_opening = bool(_META_EXPLANATION_OPENING_RE.search(normalized))
    capability_opening = bool(
        _META_CAPABILITY_EXPLANATION_OPENING_RE.search(normalized)
    )
    if not standard_opening and not capability_opening:
        return False
    capability_discussion = bool(_META_CAPABILITY_DISCUSSION_RE.search(normalized))
    if capability_opening and capability_discussion:
        return not bool(_META_FOLLOW_UP_MATERIALIZATION_RE.search(normalized))
    if not standard_opening or not _META_PROCESS_HINT_RE.search(normalized):
        return False
    has_example_hint = bool(_META_EXAMPLE_HINT_RE.search(normalized))
    has_quoted_example = bool(_META_EXAMPLE_QUOTE_RE.search(raw_prompt))
    return has_example_hint or has_quoted_example


def _has_explicit_materialization_deferal(
    prompt: str,
    normalized_prompt: str,
    *,
    mentions_materialization_targets: bool,
) -> bool:
    raw_prompt = str(prompt or '').strip()
    normalized = str(normalized_prompt or '').strip()
    if not raw_prompt or not normalized or not mentions_materialization_targets:
        return False
    for pattern in _EXPLICIT_DEFER_MATERIALIZATION_RE:
        for source in (raw_prompt, normalized):
            for match in pattern.finditer(source):
                if materialization_negation_match_is_artifact_fulfillment_only(
                    source,
                    match.start(),
                    match.end(),
                ):
                    continue
                if _materialization_negation_match_is_cardinality_constraint(
                    source,
                    match.start(),
                    match.end(),
                ):
                    continue
                if materialization_negation_match_is_output_contrast(
                    source,
                    match.start(),
                    match.end(),
                ):
                    continue
                return True
    return False


def _latest_explicit_visual_defer_end(normalized_prompt: str) -> int:
    """Return the last visual defer match end in command text, or ``-1``."""

    prompt = str(normalized_prompt or '').strip()
    latest_end = -1
    for pattern in _EXPLICIT_DEFER_MATERIALIZATION_RE:
        for match in pattern.finditer(prompt):
            if materialization_negation_match_is_artifact_fulfillment_only(
                prompt,
                match.start(),
                match.end(),
            ):
                continue
            if _materialization_negation_match_is_cardinality_constraint(
                prompt,
                match.start(),
                match.end(),
            ):
                continue
            if materialization_negation_match_is_output_contrast(
                prompt,
                match.start(),
                match.end(),
            ):
                continue
            scope = _materialization_negation_scope(
                prompt,
                match.start(),
                match.end(),
            )
            if _VISUAL_ACTION_TARGET_RE.search(scope):
                contrast = _VISUAL_ACTION_POLARITY_RESET_RE.search(
                    prompt,
                    match.start(),
                    match.end(),
                )
                governing_end = contrast.start() if contrast else match.end()
                latest_end = max(latest_end, governing_end)
    return latest_end


def _is_text_revision_turn(
    prompt: str,
    normalized_prompt: str,
    *,
    direct_audio_materialization_request: bool,
) -> bool:
    raw_prompt = str(prompt or '').strip()
    normalized = str(normalized_prompt or '').strip()
    if not raw_prompt or not normalized or direct_audio_materialization_request:
        return False
    named_revision_intent = classify_named_text_revision_intent(raw_prompt)
    if (
        named_revision_intent.get('mutation_requested') is True
        and bool(named_revision_intent.get('named_targets'))
    ):
        return True
    has_revision_action = bool(_TEXT_REVISION_ACTION_RE.search(normalized))
    has_output_constraint = bool(_TEXT_REVISION_OUTPUT_RE.search(normalized))
    has_text_target = bool(_TEXT_REVISION_TARGET_RE.search(normalized))
    if has_revision_action and has_text_target:
        return True
    if has_output_constraint and has_text_target:
        return True
    return False


def analyze_prompt_intent(prompt: str) -> dict[str, Any]:
    normalized = normalize_intent_text(prompt)
    normalized_tts_source = resolve_explicit_tts_source(normalized)
    explicit_tts_source = (
        resolve_explicit_tts_source(str(prompt or ''))
        or normalized_tts_source
    )
    preliminary_answer_as_audio_delivery_spans, _, _ = _answer_as_audio_delivery_state(normalized)
    preliminary_tts_positive, _ = _score_rules(normalized, _TTS_POSITIVE_RULES)
    direct_spoken_payload_request = bool(
        normalized_tts_source
        or preliminary_tts_positive >= 4
        or preliminary_answer_as_audio_delivery_spans
    )
    command_text = (
        _mask_literal_payloads(normalized)
        if direct_spoken_payload_request
        else normalized
    )
    raw_command_text = (
        _mask_literal_payloads(str(prompt or ''))
        if direct_spoken_payload_request
        else str(prompt or '')
    )
    (
        answer_as_audio_delivery_spans,
        answer_as_audio_delivery_negated,
        answer_as_audio_delivery_deferred,
    ) = _answer_as_audio_delivery_state(command_text)
    answer_as_audio_delivery_request = bool(answer_as_audio_delivery_spans)
    tts_positive, tts_cues = _score_rules(command_text, _TTS_POSITIVE_RULES)
    if normalized_tts_source:
        tts_positive = max(tts_positive, 5)
        if 'direct_tts_source_contract' not in tts_cues:
            tts_cues.append('direct_tts_source_contract')
    tts_negative, tts_negative_cues = _score_rules(command_text, _TTS_NEGATIVE_RULES)
    visual_command_text = command_text
    image_positive, image_cues = _score_rules(visual_command_text, _IMAGE_POSITIVE_RULES)
    image_negative, image_negative_cues = _score_rules(visual_command_text, _IMAGE_NEGATIVE_RULES)
    vision_positive, vision_cues = _score_rules(visual_command_text, _VISION_POSITIVE_RULES)
    stt_positive, stt_cues = _score_rules(command_text, _STT_POSITIVE_RULES)
    (
        requested_audio_output_count,
        counted_audio_output_obligation,
        audio_output_count_exceeds_bound,
        requested_audio_output_count_raw,
    ) = _infer_requested_audio_output_count(
        command_text,
        answer_as_audio_delivery_spans=answer_as_audio_delivery_spans,
    )
    (
        visual_artifact_preservation_without_regeneration,
        visual_analysis_preservation_without_reanalysis,
        visual_preservation_cues,
    ) = _visual_preservation_flags(visual_command_text)
    (
        separate_visual_generation_request,
        separate_visual_analysis_request,
        separate_visual_work_cues,
    ) = _separate_visual_work_flags(visual_command_text)
    if answer_as_audio_delivery_request:
        tts_positive = max(tts_positive, 5)
        if 'answer_as_audio_delivery_request' not in tts_cues:
            tts_cues.append('answer_as_audio_delivery_request')
    visual_artifact_execution_suppressed_by_preservation = bool(
        visual_artifact_preservation_without_regeneration
        and not separate_visual_generation_request
    )
    visual_analysis_execution_suppressed_by_preservation = bool(
        visual_analysis_preservation_without_reanalysis
        and not separate_visual_analysis_request
    )
    if counted_audio_output_obligation:
        tts_positive = max(tts_positive, 5)
        if 'counted_audio_output_obligation' not in tts_cues:
            tts_cues.append('counted_audio_output_obligation')
    if separate_visual_generation_request:
        image_positive = max(image_positive, 5)
        for cue in separate_visual_work_cues:
            if cue not in image_cues and cue == 'separate_visual_generation_request':
                image_cues.append(cue)
    if separate_visual_analysis_request:
        vision_positive = max(vision_positive, 5)
        for cue in separate_visual_work_cues:
            if cue not in vision_cues and cue == 'separate_visual_analysis_request':
                vision_cues.append(cue)
    if visual_artifact_execution_suppressed_by_preservation:
        image_positive = 0
        if 'visual_artifact_preservation_without_regeneration' not in image_negative_cues:
            image_negative_cues.append('visual_artifact_preservation_without_regeneration')
    if visual_analysis_execution_suppressed_by_preservation:
        vision_positive = 0

    capability_scores = {
        CAPABILITY_TEXT_TO_SPEECH: max(0, tts_positive - tts_negative),
        CAPABILITY_IMAGE_GENERATION: max(0, image_positive - image_negative),
        CAPABILITY_VISION_ANALYSIS: max(0, vision_positive),
        CAPABILITY_SPEECH_TO_TEXT: max(0, stt_positive),
    }
    threshold_map = {
        CAPABILITY_TEXT_TO_SPEECH: 4,
        CAPABILITY_IMAGE_GENERATION: 4,
        CAPABILITY_VISION_ANALYSIS: 4,
        CAPABILITY_SPEECH_TO_TEXT: 4,
    }
    has_explicit_spoken_content = bool(normalized_tts_source)
    has_text_preparation_step = bool(
        answer_as_audio_delivery_request
        or any(pattern.search(command_text) for pattern in _TEXT_PREPARATION_HINTS)
    )
    source_followed_by_text_preparation = bool(
        normalized_tts_source
        and any(
            pattern.search(
                command_text[int(normalized_tts_source.get('end') or 0):]
            )
            for pattern in _TEXT_PREPARATION_HINTS
        )
    )
    has_audio_follow_up_request = bool(
        answer_as_audio_delivery_request
        or any(pattern.search(command_text) for pattern in _AUDIO_FOLLOW_UP_HINTS)
    )
    audio_output_negation_matches = [
        match
        for pattern in _AUDIO_OUTPUT_NEGATION_HINTS
        for match in pattern.finditer(command_text)
    ]
    audio_output_negation_matches = [
        match
        for match in audio_output_negation_matches
        if not materialization_negation_match_is_output_contrast(
            command_text,
            match.start(),
            match.end(),
        )
        and _AUDIO_NEGATION_TARGET_RE.search(match.group(0))
        and _audio_negation_match_governs_negated_action(command_text, match)
    ]
    if answer_as_audio_delivery_request:
        audio_output_negation_matches = []
    if audio_output_negation_matches and all(
        _materialization_negation_match_is_cardinality_constraint(
            command_text,
            match.start(),
            match.end(),
        )
        for match in audio_output_negation_matches
    ):
        audio_output_negation_matches = []
    affirmative_audio_materialization_action = _has_affirmative_audio_materialization_action(
        command_text
    )
    has_audio_output_negation = bool(
        audio_output_negation_matches
        or answer_as_audio_delivery_negated
    ) and not affirmative_audio_materialization_action
    has_visual_follow_up_request = any(
        pattern.search(visual_command_text)
        for pattern in _VISUAL_FOLLOW_UP_HINTS
    )
    has_visual_text_preparation_step = any(
        pattern.search(visual_command_text)
        for pattern in _VISUAL_TEXT_PREPARATION_HINTS
    )
    has_visual_creative_delegation = any(
        pattern.search(visual_command_text)
        for pattern in _VISUAL_CREATIVE_DELEGATION_HINTS
    )
    has_visual_descriptor_request = bool(
        re.search(
            r'\b(describe|depict|portray|beschreibe|schildere|zeichne)\b',
            visual_command_text,
        )
    )
    has_narration_script_request = bool(_NARRATION_SCRIPT_HINT_RE.search(command_text))
    requests_translation_output = any(pattern.search(command_text) for pattern in _TRANSLATION_OUTPUT_HINTS)
    meta_execution_explanation_request = _is_meta_execution_explanation_request(prompt, normalized)
    requested_visual_output_count = _infer_requested_visual_output_count(visual_command_text)
    local_visual_asset_cues = _local_visual_asset_requirement_cues(visual_command_text)
    local_visual_asset_requirement = bool(local_visual_asset_cues)
    inferred_visual_output_count_source = ''
    if local_visual_asset_requirement:
        local_visual_count, local_visual_count_source = _infer_local_visual_asset_output_count(
            visual_command_text
        )
        if local_visual_count > requested_visual_output_count:
            requested_visual_output_count = local_visual_count
            inferred_visual_output_count_source = local_visual_count_source
        elif requested_visual_output_count <= 0:
            requested_visual_output_count = 1
            inferred_visual_output_count_source = 'local_visual_asset_requirement_fallback'
        capability_scores[CAPABILITY_IMAGE_GENERATION] = max(
            capability_scores.get(CAPABILITY_IMAGE_GENERATION, 0),
            4,
        )
        for cue in local_visual_asset_cues:
            if cue not in image_cues:
                image_cues.append(cue)
    counted_visual_output_obligation = _has_counted_visual_output_obligation(
        visual_command_text,
        requested_visual_output_count,
    )
    if counted_visual_output_obligation:
        capability_scores[CAPABILITY_IMAGE_GENERATION] = max(
            capability_scores.get(CAPABILITY_IMAGE_GENERATION, 0),
            4,
        )
        if 'counted_visual_output_obligation' not in image_cues:
            image_cues.append('counted_visual_output_obligation')
    if visual_artifact_execution_suppressed_by_preservation:
        capability_scores[CAPABILITY_IMAGE_GENERATION] = 0
    if visual_analysis_execution_suppressed_by_preservation:
        capability_scores[CAPABILITY_VISION_ANALYSIS] = 0
    direct_audio_materialization_request = bool(
        {
            'direct_tts_request',
            'direct_tts_artifact_request',
            'direct_tts_phrase_request',
            'direct_tts_request_de',
            'direct_tts_request_es',
            'direct_tts_request_fr',
            'direct_audio_variant_replacement',
            'answer_as_audio_delivery_request',
        }
        & set(tts_cues)
        or (
            bool(explicit_tts_source)
            and capability_scores.get(CAPABILITY_TEXT_TO_SPEECH, 0) >= 4
        )
    )
    if (
        has_audio_output_negation
        and direct_audio_materialization_request
        and re.search(
            r'\b(?:kein(?:e|en|er|es)?|nicht|ohne|noch\s+kein)\b[^.;!?]{0,48}\b(?:bild(?:er)?|image(?:s)?|picture(?:s)?)\b'
            r'[^.;!?]{0,120}\b(?:lies|lese|sprich|read|speak|audio|vor)\b',
            command_text,
        )
    ):
        has_audio_output_negation = False
    if has_audio_output_negation:
        capability_scores[CAPABILITY_TEXT_TO_SPEECH] = 0
        has_audio_follow_up_request = False
        direct_audio_materialization_request = False
        requested_audio_output_count = 0
        counted_audio_output_obligation = False
        if 'negated_audio_output_request' not in tts_negative_cues:
            tts_negative_cues.append('negated_audio_output_request')
    mentions_materialization_targets = bool(
        capability_scores[CAPABILITY_TEXT_TO_SPEECH] > 0
        or capability_scores[CAPABILITY_IMAGE_GENERATION] > 0
        or has_audio_follow_up_request
        or has_visual_follow_up_request
        or _MATERIALIZATION_TARGET_RE.search(command_text)
    )
    explicit_defer_materialization = _has_explicit_materialization_deferal(
        raw_command_text,
        command_text,
        mentions_materialization_targets=mentions_materialization_targets,
    )
    if answer_as_audio_delivery_deferred:
        explicit_defer_materialization = True
    latest_affirmative_visual_action_start = (
        _latest_affirmative_visual_generation_action_start(visual_command_text)
    )
    latest_explicit_visual_defer_end = _latest_explicit_visual_defer_end(
        visual_command_text
    )
    affirmative_visual_action_overrides_defer = bool(
        _ADDITIVE_MATERIALIZATION_RE.search(visual_command_text)
        or (
            latest_affirmative_visual_action_start >= 0
            and latest_affirmative_visual_action_start >= latest_explicit_visual_defer_end
        )
    )
    explicit_visual_defer_materialization = bool(
        explicit_defer_materialization
        and not affirmative_visual_action_overrides_defer
        and re.search(
            r'\b(?:image(?:s)?|picture(?:s)?|photo(?:s)?|illustration(?:s)?|bild(?:er)?|foto(?:s)?|animation(?:en)?)\b',
            visual_command_text,
        )
    )
    target_scoped_visual_preservation = bool(
        visual_artifact_preservation_without_regeneration
        and separate_visual_generation_request
    )
    if target_scoped_visual_preservation:
        explicit_visual_defer_materialization = False
    targeted_audio_defer = bool(
        answer_as_audio_delivery_deferred
        or (
            explicit_defer_materialization
            and re.search(
                r'\b(?:do\s+not|don[\'’]?t|dont|not\s+yet|not\s+now|kein(?:e|en|er|es)?|nicht|ohne|noch\s+kein)\b'
                r'[^.;!?]{0,96}\b(?:audio|voice|voiceover|speech|tts|mp3|wav|stimme|sprachversion|horversion|hoerversion)\b',
                command_text,
            )
        )
    )
    if not targeted_audio_defer and explicit_defer_materialization:
        targeted_audio_defer = bool(
            re.search(
                r'\b(?:audio|voice|voiceover|speech|tts|mp3|wav|stimme|sprachversion|horversion|hoerversion|'
                r'hoerver(?:sion)?|hörver(?:sion)?)\b'
                r'[^.;!?]{0,140}\b(?:later|future|possible|hold|keep|reserve|zur(?:ue|ü|u)ck|spaeter|später|spater|'
                r'moegliche(?:n|r|s)?|mögliche(?:n|r|s)?|mogliche(?:n|r|s)?|option(?:en)?|schritt(?:e)?|phase(?:n)?)\b',
                command_text,
            )
        )
    if (
        targeted_audio_defer
        and explicit_visual_defer_materialization
        and direct_audio_materialization_request
        and re.search(
            r'\b(?:do\s+not|don[\'’]?t|dont|not\s+yet|not\s+now|kein(?:e|en|er|es)?|nicht|ohne|noch\s+kein)\b'
            r'[^.;!?]{0,64}\b(?:image(?:s)?|picture(?:s)?|photo(?:s)?|illustration(?:s)?|bild(?:er)?|foto(?:s)?)\b'
            r'[^.;!?]{0,140}\b(?:read|speak|lies|lese|sprich|audio|voice|vor)\b',
            command_text,
        )
    ):
        targeted_audio_defer = False
    if affirmative_audio_materialization_action and not answer_as_audio_delivery_deferred:
        targeted_audio_defer = False
    explicit_audio_defer_materialization = bool(
        has_audio_output_negation
        or targeted_audio_defer
    )
    if (
        target_scoped_visual_preservation
        and not explicit_visual_defer_materialization
        and not explicit_audio_defer_materialization
    ):
        explicit_defer_materialization = False
    named_text_revision_intent = classify_named_text_revision_intent(raw_command_text)
    text_revision_turn = _is_text_revision_turn(
        raw_command_text,
        command_text,
        direct_audio_materialization_request=direct_audio_materialization_request,
    )
    if text_revision_turn and not direct_audio_materialization_request:
        capability_scores[CAPABILITY_TEXT_TO_SPEECH] = 0
        requested_audio_output_count = 0
        counted_audio_output_obligation = False
    if explicit_audio_defer_materialization:
        capability_scores[CAPABILITY_TEXT_TO_SPEECH] = 0
        requested_audio_output_count = 0
        counted_audio_output_obligation = False
    if explicit_visual_defer_materialization:
        capability_scores[CAPABILITY_IMAGE_GENERATION] = 0
    candidates = [
        capability
        for capability, score in capability_scores.items()
        if score >= threshold_map.get(capability, 4)
    ]
    primary_capability = None
    if candidates:
        primary_capability = sorted(
            candidates,
            key=lambda capability: (capability_scores.get(capability, 0), capability),
            reverse=True,
        )[0]
    requests_audio_output = capability_scores[CAPABILITY_TEXT_TO_SPEECH] >= 4
    requests_visual_output = capability_scores[CAPABILITY_IMAGE_GENERATION] >= 4
    requests_speech_to_text_output = capability_scores[CAPABILITY_SPEECH_TO_TEXT] >= 4
    open_ended_visual_batch_preparation = bool(
        requests_visual_output
        and requested_visual_output_count > 1
        and has_visual_creative_delegation
    )
    text_preparation_before_audio_output = bool(
        not has_audio_output_negation
        and (
            answer_as_audio_delivery_request
            or (
                (not has_explicit_spoken_content or source_followed_by_text_preparation)
                and has_text_preparation_step
                and (
                    capability_scores[CAPABILITY_TEXT_TO_SPEECH] > 0
                    or has_audio_follow_up_request
                )
            )
        )
    )
    text_preparation_before_visual_output = bool(
        (
            has_text_preparation_step
            or has_visual_text_preparation_step
            or (has_visual_descriptor_request and has_visual_follow_up_request)
            or open_ended_visual_batch_preparation
        )
        and (
            capability_scores[CAPABILITY_IMAGE_GENERATION] > 0
            or has_visual_follow_up_request
        )
    )
    text_first_follow_up_capability = None
    if text_preparation_before_audio_output and has_audio_follow_up_request:
        text_first_follow_up_capability = CAPABILITY_TEXT_TO_SPEECH
    elif text_preparation_before_visual_output and (
        has_visual_follow_up_request
        or capability_scores[CAPABILITY_IMAGE_GENERATION] >= 4
    ):
        text_first_follow_up_capability = CAPABILITY_IMAGE_GENERATION
    downstream_follow_up_capabilities: list[str] = []
    if text_preparation_before_audio_output and has_audio_follow_up_request:
        downstream_follow_up_capabilities.append(CAPABILITY_TEXT_TO_SPEECH)
    if text_preparation_before_visual_output and (
        has_visual_follow_up_request
        or capability_scores[CAPABILITY_IMAGE_GENERATION] >= 4
    ):
        downstream_follow_up_capabilities.append(CAPABILITY_IMAGE_GENERATION)
    if not (
        requests_visual_output
        or has_visual_follow_up_request
        or text_preparation_before_visual_output
        or counted_visual_output_obligation
        or local_visual_asset_requirement
    ):
        requested_visual_output_count = 0
    if meta_execution_explanation_request:
        primary_capability = None
        requests_audio_output = False
        requests_visual_output = False
        requests_speech_to_text_output = False
        has_audio_follow_up_request = False
        has_visual_follow_up_request = False
        requested_visual_output_count = 0
        requested_audio_output_count = 0
        counted_audio_output_obligation = False
        local_visual_asset_requirement = False
        local_visual_asset_cues = []
        inferred_visual_output_count_source = ''
        text_preparation_before_audio_output = False
        text_preparation_before_visual_output = False
        text_first_follow_up_capability = None
        downstream_follow_up_capabilities = []
        visual_artifact_preservation_without_regeneration = False
        visual_analysis_preservation_without_reanalysis = False
        visual_preservation_cues = []
        separate_visual_generation_request = False
        separate_visual_analysis_request = False
        separate_visual_work_cues = []
        visual_artifact_execution_suppressed_by_preservation = False
        visual_analysis_execution_suppressed_by_preservation = False
    if text_revision_turn and not direct_audio_materialization_request:
        if primary_capability == CAPABILITY_TEXT_TO_SPEECH:
            primary_capability = None
        requests_audio_output = False
        requested_audio_output_count = 0
        counted_audio_output_obligation = False
        text_preparation_before_audio_output = False
        if text_first_follow_up_capability == CAPABILITY_TEXT_TO_SPEECH:
            text_first_follow_up_capability = None
        downstream_follow_up_capabilities = [
            capability
            for capability in downstream_follow_up_capabilities
            if capability != CAPABILITY_TEXT_TO_SPEECH
        ]
    if explicit_visual_defer_materialization or explicit_audio_defer_materialization:
        if explicit_audio_defer_materialization:
            requests_audio_output = False
            requested_audio_output_count = 0
            counted_audio_output_obligation = False
            has_audio_follow_up_request = False
            text_preparation_before_audio_output = False
            if text_first_follow_up_capability == CAPABILITY_TEXT_TO_SPEECH:
                text_first_follow_up_capability = None
        if explicit_visual_defer_materialization:
            requests_visual_output = False
            has_visual_follow_up_request = False
            requested_visual_output_count = 0
            local_visual_asset_requirement = False
            local_visual_asset_cues = []
            inferred_visual_output_count_source = ''
            counted_visual_output_obligation = False
            text_preparation_before_visual_output = False
            if text_first_follow_up_capability == CAPABILITY_IMAGE_GENERATION:
                text_first_follow_up_capability = None
        downstream_follow_up_capabilities = [
            capability
            for capability in downstream_follow_up_capabilities
            if not (
                (capability == CAPABILITY_TEXT_TO_SPEECH and explicit_audio_defer_materialization)
                or (capability == CAPABILITY_IMAGE_GENERATION and explicit_visual_defer_materialization)
            )
        ]
    
    temperament_hint, temperament_cues = infer_temperament_hint(prompt)
    return {
        'normalized_prompt': normalized,
        'capability_scores': capability_scores,
        'capability_cues': {
            CAPABILITY_TEXT_TO_SPEECH: tts_cues,
            CAPABILITY_IMAGE_GENERATION: image_cues,
            CAPABILITY_VISION_ANALYSIS: vision_cues,
            CAPABILITY_SPEECH_TO_TEXT: stt_cues,
        },
        'negative_cues': {
            CAPABILITY_TEXT_TO_SPEECH: tts_negative_cues,
            CAPABILITY_IMAGE_GENERATION: image_negative_cues,
        },
        'primary_capability': primary_capability,
        'language_codes': infer_prompt_languages(normalized),
        'voice_descriptors': infer_voice_descriptors(normalized),
        'audio_response_format': infer_audio_response_format(normalized),
        'image_aspect_ratio': infer_image_aspect_ratio(normalized),
        'meta_execution_explanation_request': meta_execution_explanation_request,
        'direct_audio_materialization_request': direct_audio_materialization_request,
        'negated_audio_output_request': has_audio_output_negation,
        'explicit_defer_materialization': explicit_defer_materialization,
        'explicit_visual_defer_materialization': explicit_visual_defer_materialization,
        'explicit_audio_defer_materialization': explicit_audio_defer_materialization,
        'visual_artifact_preservation_without_regeneration': (
            visual_artifact_preservation_without_regeneration
        ),
        'visual_analysis_preservation_without_reanalysis': (
            visual_analysis_preservation_without_reanalysis
        ),
        'visual_preservation_cues': visual_preservation_cues,
        'separate_visual_generation_request': separate_visual_generation_request,
        'separate_visual_analysis_request': separate_visual_analysis_request,
        'separate_visual_work_cues': separate_visual_work_cues,
        'visual_artifact_execution_suppressed_by_preservation': (
            visual_artifact_execution_suppressed_by_preservation
        ),
        'visual_analysis_execution_suppressed_by_preservation': (
            visual_analysis_execution_suppressed_by_preservation
        ),
        'text_revision_turn': text_revision_turn,
        'named_text_revision_intent': named_text_revision_intent,
        'requests_audio_output': requests_audio_output,
        'requests_visual_output': requests_visual_output,
        'requests_speech_to_text_output': requests_speech_to_text_output,
        'requests_translation_output': requests_translation_output,
        'counted_visual_output_obligation': counted_visual_output_obligation,
        'local_visual_asset_requirement': local_visual_asset_requirement,
        'local_visual_asset_cues': local_visual_asset_cues,
        'inferred_visual_output_count_source': inferred_visual_output_count_source,
        'requested_visual_output_count': requested_visual_output_count,
        'counted_audio_output_obligation': counted_audio_output_obligation,
        'requested_audio_output_count': requested_audio_output_count,
        'requested_audio_output_count_raw': requested_audio_output_count_raw,
        'audio_output_count_exceeds_bound': audio_output_count_exceeds_bound,
        'requested_audio_output_count_max': _MAX_REQUESTED_AUDIO_OUTPUT_COUNT,
        'has_audio_follow_up_request': has_audio_follow_up_request,
        'has_visual_follow_up_request': has_visual_follow_up_request,
        'text_preparation_before_audio_output': text_preparation_before_audio_output,
        'text_preparation_before_visual_output': text_preparation_before_visual_output,
        'text_first_follow_up_capability': text_first_follow_up_capability,
        'downstream_follow_up_capabilities': downstream_follow_up_capabilities,
        'temperament_hint': temperament_hint,
        'temperament_cues': temperament_cues,
    }


def prompt_has_self_contained_direct_tts_source(prompt: Any) -> bool:
    """Return whether current-turn text fully binds one direct TTS source."""

    if not resolve_explicit_tts_source(prompt):
        return False
    analysis = analyze_prompt_intent(str(prompt or ''))
    return bool(
        analysis.get('direct_audio_materialization_request')
        and analysis.get('requests_audio_output')
        and not analysis.get('text_preparation_before_audio_output')
        and not analysis.get('requests_visual_output')
        and not analysis.get('requests_speech_to_text_output')
    )
