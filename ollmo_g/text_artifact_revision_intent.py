"""Pure current-turn intent classification for named text-artifact revisions."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ollmo_core.transports import TEXT_ARTIFACT_EXTENSIONS


_NAMED_TEXT_FILENAME_RE = re.compile(
    r'(?<![A-Za-z0-9._-])'
    r'(?P<filename>[A-Za-z0-9][A-Za-z0-9._-]{0,120}\.(?:'
    + '|'.join(sorted(TEXT_ARTIFACT_EXTENSIONS, key=len, reverse=True))
    + r'))(?![A-Za-z0-9_-])',
    re.IGNORECASE,
)
_TEXT_MUTATION_ACTION_RE = re.compile(
    r'\b(?:chang(?:e|ed|es|ing)|modif(?:y|ied|ies|ying)|updat(?:e|ed|es|ing)|'
    r'edit(?:ed|s|ing)?|revis(?:e|ed|es|ing)|alter(?:ed|s|ing)?|'
    r'replac(?:e|ed|es|ing)|repair(?:ed|s|ing)?|fix(?:ed|es|ing)?|'
    r'adjust(?:ed|s|ing)?|tweak(?:ed|s|ing)?|upgrad(?:e|ed|es|ing)|'
    r'restyl(?:e|ed|es|ing)|wire(?:d|s|ing)?|link(?:ed|s|ing)?|'
    r'bind(?:s|ing|bound)?|embed(?:ded|s|ding)?|'
    r'add(?:ed|s|ing)?|append(?:ed|s|ing)?|insert(?:ed|s|ing)?|'
    r'includ(?:e|ed|es|ing)|remov(?:e|ed|es|ing)|delet(?:e|ed|es|ing)|'
    r'aendere|ändere|veraendere|verändere|aktualisier(?:e|en|t)?|'
    r'bearbeit(?:e|en|et)?|anpass(?:e|en|t)?|passe\s+an|'
    r'ersetz(?:e|en|t)?|reparier(?:e|en|t)?|korrigier(?:e|en|t)?|'
    r'fueg(?:e|en|t)?|füg(?:e|en|t)?|entfern(?:e|en|t)?|'
    r'loesch(?:e|en|t)?|lösch(?:e|en|t)?|verlink(?:e|en|t)?|'
    r'verkn(?:u|ü)pf(?:e|en|t)?|einbind(?:e|en|et)?|bind(?:e|en|et)?)\b',
    re.IGNORECASE,
)
_TEXT_CREATE_ACTION_RE = re.compile(
    r'\b(?:create|generate|make|produce|build|draft|write|materiali[sz]e|'
    r'erstell(?:e|en|t)?|generier(?:e|en|t)?|erzeug(?:e|en|t)?|'
    r'baue|bau(?:en|t)?|schreib(?:e|en|t)?|materialisier(?:e|en|t)?)\b',
    re.IGNORECASE,
)
_TEXT_SOURCE_ANCHOR_RE = re.compile(
    r'\b(?:use|using|reference|referencing|work\s+from|start\s+from|'
    r'current|existing|selected|preserve|keep|'
    r'verwende|verwendet|nutze|nutzt|referenziere|referenziert|'
    r'aktuell(?:e|en|er|es|em)?|bestehend(?:e|en|er|es|em)?|'
    r'ausgewählt(?:e|en|er|es|em)?)\b',
    re.IGNORECASE,
)
_TEXT_SEGMENT_RE = re.compile(
    r'[^\r\n;]+?(?:(?<=[.!?])(?=\s+[A-ZÄÖÜ0-9])|$)',
    re.MULTILINE,
)
_PRESERVATION_RE = re.compile(
    r'\b(?:preserve|retain|keep|leave)\b[^.;!?\n]{0,120}'
    r'\b(?:rest|other|everything|all|unchanged|intact|as\s+is)\b|'
    r'\b(?:preserve|retain|keep|leave)\s+(?:everything|all)\b|'
    r'\b(?:bewahre|behalte|lass(?:e)?|erhalte)\b[^.;!?\n]{0,120}'
    r'\b(?:rest|andere[nmrs]?|alles|unverändert|intakt)\b',
    re.IGNORECASE,
)
_FULL_REPLACEMENT_RE = re.compile(
    r'\b(?:replace|rewrite|rebuild|recreate)\b[^.;!?\n]{0,80}'
    r'\b(?:entire|whole|complete|from\s+scratch)\b|'
    r'\b(?:vollständig|komplett|ganz)\b[^.;!?\n]{0,80}'
    r'\b(?:ersetzen|neu\s+schreiben|neu\s+aufbauen)\b',
    re.IGNORECASE,
)


def classify_named_text_revision_intent(prompt: Any) -> dict[str, Any]:
    """Return bounded evidence for current-turn edits of explicitly named files.

    A filename is governed either by an edit action in the same sentence or by
    an explicit source-acquisition sentence immediately followed by a filename-
    free edit sentence. Creation verbs never authorize predecessor mutation.
    """

    text = str(prompt or '').strip()
    if not text:
        return {
            'mutation_requested': False,
            'source_acquisition_requested': False,
            'named_targets': [],
            'preservation_requested': False,
            'full_replacement_requested': False,
        }
    segments = [
        match.group(0)
        for match in _TEXT_SEGMENT_RE.finditer(text)
        if match.group(0).strip()
    ]
    targets: list[str] = []
    seen: set[str] = set()

    def append_target(value: Any) -> None:
        filename = Path(str(value or '')).name
        key = filename.lower()
        if filename and key not in seen:
            seen.add(key)
            targets.append(filename)

    for segment in segments:
        if not _TEXT_MUTATION_ACTION_RE.search(segment):
            continue
        actions = [
            (match.start(), 'edit')
            for match in _TEXT_MUTATION_ACTION_RE.finditer(segment)
        ] + [
            (match.start(), 'create')
            for match in _TEXT_CREATE_ACTION_RE.finditer(segment)
        ]
        actions.sort()
        for filename_match in _NAMED_TEXT_FILENAME_RE.finditer(segment):
            prior_actions = [item for item in actions if item[0] < filename_match.start()]
            governing_action = prior_actions[-1][1] if prior_actions else None
            if governing_action is None:
                following_actions = [item for item in actions if item[0] > filename_match.end()]
                governing_action = following_actions[0][1] if following_actions else None
            if governing_action == 'edit':
                append_target(filename_match.group('filename'))

    for index, source_segment in enumerate(segments[:-1]):
        source_filenames = [
            match.group('filename')
            for match in _NAMED_TEXT_FILENAME_RE.finditer(source_segment)
        ]
        if (
            not source_filenames
            or not _TEXT_SOURCE_ANCHOR_RE.search(source_segment)
            or _TEXT_MUTATION_ACTION_RE.search(source_segment)
            or _TEXT_CREATE_ACTION_RE.search(source_segment)
        ):
            continue
        edit_segment = segments[index + 1]
        if (
            not _TEXT_MUTATION_ACTION_RE.search(edit_segment)
            or _NAMED_TEXT_FILENAME_RE.search(edit_segment)
        ):
            continue
        for filename in source_filenames:
            append_target(filename)

    source_acquisition_requested = any(
        _TEXT_SOURCE_ANCHOR_RE.search(segment)
        and _NAMED_TEXT_FILENAME_RE.search(segment)
        for segment in segments
    )
    return {
        'mutation_requested': bool(_TEXT_MUTATION_ACTION_RE.search(text)),
        'source_acquisition_requested': bool(source_acquisition_requested),
        'named_targets': targets,
        'preservation_requested': bool(_PRESERVATION_RE.search(text)),
        'full_replacement_requested': bool(_FULL_REPLACEMENT_RE.search(text)),
    }
