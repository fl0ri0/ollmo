import json
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ollmo_webserver import app


REPO_ROOT = Path(__file__).resolve().parents[1]
OLLMO_CLI = REPO_ROOT / 'ollmo'
LANDING_SITE_ROOT = REPO_ROOT / 'site'
LANDING_PATH = LANDING_SITE_ROOT / 'index.html'
LANDING_CSS_PATH = LANDING_SITE_ROOT / 'landing.css'
LANDING_JS_PATH = LANDING_SITE_ROOT / 'landing.js'
DASHBOARD_CSS_PATH = REPO_ROOT / 'static' / 'ui' / 'ollmo.css'


def _run_node(script: str) -> dict:
    result = subprocess.run(
        ['node', '-e', textwrap.dedent(script)],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(result.stdout)


def _custom_property(css: str, name: str) -> str:
    match = re.search(rf'^\s*{re.escape(name)}:\s*(.+);$', css, re.MULTILINE)
    assert match is not None
    return match.group(1).strip()


def _keyframes_block(css: str, name: str) -> str:
    start = css.index(f'@keyframes {name}')
    opening = css.index('{', start)
    depth = 0
    for index in range(opening, len(css)):
        if css[index] == '{':
            depth += 1
        elif css[index] == '}':
            depth -= 1
            if depth == 0:
                return re.sub(r'\s+', ' ', css[start : index + 1]).strip()
    raise AssertionError(f'Unclosed keyframes block: {name}')


def test_landing_is_a_standalone_static_repo_page():
    html = LANDING_PATH.read_text(encoding='utf-8')

    assert 'data-page="ollmo-landing"' in html
    assert re.search(r'href="\./landing\.css(?:\?[^\"]+)?"', html)
    assert 'src="./landing.js"' in html
    assert '{{' not in html
    assert '{%' not in html
    assert 'url_for(' not in html
    assert 'Where local AI becomes durable work.' in html
    assert '<header class="site-header">' not in html
    assert 'class="wordmark"' not in html
    assert 'class="hero-actions"' not in html
    assert 'Open Ollmo' not in html
    assert 'Install locally' not in html
    assert 'class="repo-link" href="https://github.com/fl0ri0/ollmo"' in html
    setup = html.split(
        '<section id="setup" class="setup-scene scene"',
        1,
    )[1].split('</section>', 1)[0]
    assert setup.index('class="install-copy"') < setup.index('class="setup-path"')
    assert setup.index('class="setup-path"') < setup.index('class="repo-link"')
    assert 'href="#install"' not in html
    assert '<p class="eyebrow">Ollmo</p>' in html
    hero = html.split('<section id="intro" class="hero scene"', 1)[1].split(
        '</section>',
        1,
    )[0]
    assert '0.1 experimental' not in hero.lower()
    assert 'class="landing-ghost"' in html
    assert 'landing-ghost__body' in html
    assert 'landing-ghost__outline' in html
    assert 'id="landing-ghost-shape"' in html
    assert 'id="landing-ghost-body-clip"' in html
    assert 'id="landing-ghost-body-mask"' in html
    assert 'landing-ghost__eye-cutouts' in html
    assert html.count('href="#landing-ghost-shape"') == 3
    assert html.count('d="M128,24') == 1
    assert (
        'class="landing-ghost__body" href="#landing-ghost-shape" '
        'fill="#baff8a" mask="url(#landing-ghost-body-mask)"' in html
    )
    assert (
        'class="landing-ghost__outline" href="#landing-ghost-shape" '
        'fill="none" stroke="#baff8a" '
        'clip-path="url(#landing-ghost-body-clip)"' in html
    )
    assert 'landing-ghost__eyes' not in html
    assert 'landing-ghost__icon--stroke' not in html
    assert 'landing-ghost__icon--filled' not in html
    assert 'ambient ambient--' not in html
    assert 'fonts.googleapis.com' not in html
    assert 'cdnjs.cloudflare.com' not in html
    assert 'cdn.jsdelivr.net' not in html

    for relative_path in (
        Path('site/landing.css'),
        Path('site/landing.js'),
        Path('site/fonts/Montserrat-Black-latin.woff2'),
        Path('site/fonts/OFL.txt'),
    ):
        assert (REPO_ROOT / relative_path).is_file()

    assert html.count('class="scene"') == 0
    assert html.count(' scene"') == 5
    for scene_id in ('intro', 'principles', 'proof', 'about', 'setup'):
        assert f'id="{scene_id}"' in html
    assert html.count('tabindex="-1"') == 6
    assert 'class="scene-nav" aria-label="Page sections"' in html
    assert html.count('class="scene-nav__marker"') == 5
    for scene_id in ('intro', 'principles', 'proof', 'about', 'setup'):
        assert f'class="scene-nav__marker" href="#{scene_id}"' in html
    assert 'class="scene-nav__status visually-hidden"' in html
    assert '<span aria-hidden="true">»</span>' not in html
    beyond = html.split(
        '<section id="principles" class="beyond__overview scene"',
        1,
    )[1].split(
        '<section id="about" class="today scene"',
        1,
    )[0]
    overview = beyond.split(
        '<section id="proof" class="proof beyond__proof scene"',
        1,
    )[0]
    assert overview.count('class="principle-card__icon"') == 3
    assert 'class="principle-card__index">01' not in overview
    assert 'class="principle-card__index">02' not in overview
    assert 'class="principle-card__index">03' not in overview
    about = html.split(
        '<section id="about" class="today scene"',
        1,
    )[1].split(
        '<section id="setup" class="setup-scene scene"',
        1,
    )[0]
    assert about.count('class="principle-card__icon"') == 3
    assert 'class="principle-card__index"' not in about
    assert '<section id="proof" class="proof beyond__proof scene"' in beyond
    assert '<h2 id="proof-title">One request becomes connected work.</h2>' in beyond


def test_landing_leads_with_runtime_truth_before_current_product_body():
    html = LANDING_PATH.read_text(encoding='utf-8')
    hero = html.split('<section id="intro" class="hero scene"', 1)[1].split(
        '</section>',
        1,
    )[0]

    for term in ('Ollama', 'MLX', 'llama.cpp'):
        assert term in hero
    for term in ('outputs exist', 'work remains open', 'artifacts', 'runtime state'):
        assert term in hero
    for overpromise in (
        'whatever you have installed',
        'Universal Model Support',
        'Fully compatible',
        'Backend agnostic',
        'all of them at once',
    ):
        assert overpromise not in html

    assert 'ChatGPT' not in hero
    assert 'Backend packages and model weights are installed' in hero
    assert 'separately.' in hero
    assert 'Ollmo is a state-native runtime for local AI.' in hero
    assert 'A model can say' in hero
    assert '“done”; Ollmo records what the runtime can prove happened' in hero
    assert 'A response is more than an answer.' in html
    assert 'a graph of connected work' in html
    assert 'State becomes a response frame.' in html
    assert 'Completion still has to be earned.' in html
    assert '<p class="eyebrow">The runtime underneath</p>' in html
    assert (
        '<h2 id="today-title">Ollmo gives every<br>'
        'response a body.</h2>'
    ) in html
    assert (
        'Route text and image generation, voice input and transcription, '
        'text-to-speech, vision, and OCR work'
    ) in html
    visible_text = re.sub(r'<[^>]+>', ' ', html)
    assert 'Ghost' not in visible_text
    assert 'For supported workflows' not in visible_text
    assert 'For supported local workflows' not in visible_text
    assert 'In a supported workflow' not in visible_text
    assert 'The response preserves the work.' in visible_text
    assert html.index('A response is more than an answer.') < html.index(
        'One request becomes connected work.'
    )
    assert html.index('One request becomes connected work.') < html.index(
        '<h2 id="today-title">Ollmo gives every<br>'
    )
    assert '<strong>Third-party providers.</strong>' in html
    assert 'an explicitly enabled third-party provider (currently ChatGPT)' in html
    assert (
        'For those turns, the current prompt, only the context Ollmo promotes as '
        'relevant, and any explicitly selected files or Ollmo artifacts leave your device'
    ) in html
    assert 'by OpenAI for the current ChatGPT integration' in html
    assert "directly or through Ollmo's routing" in html
    assert (
        'Ollmo also offers an optional companion skill that helps ChatGPT inspect '
        'runtime truth, work with canonical outputs, and execute requests through Ollmo'
    ) in html
    assert 'Codex' not in visible_text
    assert (
        'Ollmo 0.1.0 is an experimental release candidate, tested on macOS on '
        'Apple Silicon with Python 3.11 or newer.'
    ) in html
    assert (
        'Output quality and factual accuracy still depend on the models and '
        'providers you choose.'
    ) in html
    for heading in (
        'Promotion turns possibility into obligation.',
        'Work becomes a graph.',
        'State becomes a response frame.',
        'Run compatible local models.',
        'Turn branches into outputs.',
        'Carry the work forward.',
    ):
        assert heading in html
    assert 'checked, not bound' in html
    assert 'Review and freeze.' in html
    for step_number in ('1', '2', '3'):
        assert f'class="proof-flow__index">{step_number}</span>' in html
    for padded_step_number in ('01', '02', '03'):
        assert f'class="proof-flow__index">{padded_step_number}</span>' not in html
    assert 'what is fulfilled, blocked, or still open' in html
    for overpromise in (
        'Ollmo is deterministic',
        'sees your entire conversation',
        'No more history-induced hallucinations',
        'transactional runtime',
        'snapshot of the entire system state',
    ):
        assert overpromise not in html
    assert 'An image prompt is not an image' in html
    assert 'Ask for one webpage with three different local images.' in html
    assert 'Create a webpage with three different local images.' in html
    assert 'One webpage branch, three local image branches' in html
    assert (
        'Ollmo checks each required output and link against runtime evidence and, '
        'where needed, reviews whether the finished set still fits the current intent '
        'before the frame can report fulfillment.'
    ) in html
    assert '<span>Open</span>' not in html
    assert 'Not yet fulfilled. Two images exist, or a placeholder remains.' not in html
    assert 'Continuing' not in html
    assert (
        '<strong class="proof-outcome__claim">The reply summarizes the moment. '
        'The response preserves the work.</strong>'
    ) in html
    assert 'class="proof-cadence"' not in html
    assert (
        'All four artifacts satisfy their branch contracts, exist as saved outputs, '
        'and their links resolve.'
    ) in html
    assert 'Arena sends one text prompt to two running local chat models' not in html
    assert 'compatible link-checked bundles' in html
    assert 'canonical Responses endpoints' in html
    assert 'Ollmo itself can start before a backend or model is available' in html
    assert 'neither the runtime nor its model weights are bundled' in html
    assert 'Choose a local runtime.' in html
    assert 'Start the control plane.' in html
    assert 'Add a compatible model.' in html
    assert 'add Hugging Face repositories or local GGUF files' in html
    assert 'remove local copies' in html
    assert 'Run Ollama, MLX, and llama.cpp side by side' in html
    assert 'multiple instances of the same model' in html
    assert 'start or stop each instance independently' in html
    assert html.index('Choose a local runtime.') < html.index('Add a compatible model.')
    assert html.index('Add a compatible model.') < html.index(
        'Start the control plane.'
    )
    assert 'Python 3.11 or newer' in html
    assert 'class="install"' not in html
    assert 'From the Ollmo checkout' not in html
    assert '<span class="install-terminal__label">Terminal</span>' in html
    assert 'data-copy-target="install-commands"' in html
    assert 'id="install-commands"' in html
    assert (
        '<code>python3 -m venv .venv\n'
        '.venv/bin/python -m pip install -r requirements.txt\n'
        './ollmo start</code>'
    ) in html
    assert html.index('Start the control plane.') < html.index(
        'class="install-terminal"'
    )
    assert '.venv/bin/python -m pip install -r requirements.txt' in html
    assert './ollmo start</code>' in html
    assert 'role="status"' in html
    assert 'aria-live="polite"' in html
    assert 'class="provider-note"' in html
    assert html.index('Carry the work forward.') < html.index('Third-party providers.')
    assert html.index('Third-party providers.') < html.index('Local setup')
    assert 'class="site-footer"' not in html
    assert 'Open Large Language Model Orchestrator' not in html


def test_root_renders_dashboard_without_redirect():
    app.config['TESTING'] = True
    response = app.test_client().get('/')

    assert response.status_code == 200
    assert not response.history
    html = response.get_data(as_text=True)
    assert 'data-page="ollmo-landing"' not in html
    assert 'id="models-list"' in html
    assert '/static/ui/ollmo.css' in html


def test_dashboard_alias_renders_same_workbench_without_redirect():
    app.config['TESTING'] = True
    client = app.test_client()
    root_response = client.get('/')
    alias_response = client.get('/dashboard')

    assert alias_response.status_code == 200
    assert not alias_response.history
    assert alias_response.data == root_response.data


def test_standalone_landing_has_canonical_local_preview_route():
    app.config['TESTING'] = True
    response = app.test_client().get('/site/')

    assert response.status_code == 200
    assert not response.history
    assert response.mimetype == 'text/html'
    html = response.get_data(as_text=True)
    assert 'data-page="ollmo-landing"' in html
    assert re.search(r'href="\./landing\.css(?:\?[^\"]+)?"', html)
    assert 'src="./landing.js"' in html
    assert 'id="models-list"' not in html


@pytest.mark.parametrize(
    ('route', 'expected_path'),
    (
        ('/site/landing.css', LANDING_CSS_PATH),
        ('/site/landing.js', LANDING_JS_PATH),
        (
            '/site/fonts/Montserrat-Black-latin.woff2',
            LANDING_SITE_ROOT / 'fonts' / 'Montserrat-Black-latin.woff2',
        ),
        ('/site/fonts/OFL.txt', LANDING_SITE_ROOT / 'fonts' / 'OFL.txt'),
    ),
)
def test_landing_site_serves_its_complete_local_dependency_tree(
    route: str,
    expected_path: Path,
):
    app.config['TESTING'] = True
    response = app.test_client().get(route)

    assert response.status_code == 200
    assert response.data == expected_path.read_bytes()


def test_landing_styles_are_local_responsive_and_motion_safe():
    css = LANDING_CSS_PATH.read_text(encoding='utf-8')
    assert '@media (max-width: 800px)' in css
    assert '@media (forced-colors: active)' in css
    assert '@media (prefers-reduced-motion: reduce)' in css
    assert 'grid-template-columns: minmax(12rem, 0.86fr) minmax(20rem, 1.14fr);' not in css
    assert 'max-width: 48rem;' in css
    assert '@font-face {' in css
    assert 'url("./fonts/Montserrat-Black-latin.woff2")' in css
    assert '--font-display: "Montserrat", "Arial Black"' in css
    assert '--eyebrow-title-gap: 1.1rem;' in css
    assert '--section-content-gap: clamp(1.75rem, 3.5vw, 2.75rem);' in css
    assert '--scene-padding: clamp(2rem, 5svh, 4.25rem);' in css
    lede_rule = css.split('.lede {', 1)[1].split('}', 1)[0]
    assert 'margin: 2rem 0 0;' in lede_rule
    assert 'calc(2rem + 12px)' not in lede_rule
    assert (
        '--surface-accent-card: color-mix(in srgb, '
        'var(--surface-panel-soft) 90%, var(--color-text));'
    ) in css
    assert '--accent-card-ink: var(--text);' in css
    assert '--surface-accent-card-start:' not in css
    assert '--surface-accent-card-end:' not in css
    eyebrow_rule = css.split('.eyebrow {', 1)[1].split('}', 1)[0]
    assert 'margin: 0 0 var(--eyebrow-title-gap);' in eyebrow_rule
    assert '.section-heading .eyebrow {' not in css
    section_heading_rule = css.split('.section-heading h2 {', 1)[1].split('}', 1)[0]
    assert 'font-family: var(--font-display);' in section_heading_rule
    assert 'font-weight: 900;' in section_heading_rule
    assert 'letter-spacing: 0;' in section_heading_rule
    assert 'line-height: 1.04;' in section_heading_rule
    expanded_heading_rule = css.split(
        '.beyond__overview .section-heading h2,\n.today .section-heading h2 {',
        1,
    )[1].split(
        '}',
        1,
    )[0]
    assert 'width: 100%;' in expanded_heading_rule
    assert 'max-width: none;' in expanded_heading_rule
    assert 'padding-bottom: 0.12em;' in expanded_heading_rule
    assert 'margin-bottom: -0.12em;' in expanded_heading_rule
    assert 'font-size: clamp(1.45rem, 7vw, 4.4rem);' in expanded_heading_rule
    assert 'line-height: 1.08;' in expanded_heading_rule
    today_heading_rule = css.rsplit('.today .section-heading h2 {', 1)[1].split(
        '}',
        1,
    )[0]
    assert 'white-space: nowrap;' in today_heading_rule
    proof_heading_rule = css.split('.proof-copy h2 {', 1)[1].split('}', 1)[0]
    assert 'font-family: var(--font-display);' in proof_heading_rule
    assert 'font-weight: 900;' in proof_heading_rule
    assert 'letter-spacing: 0;' in proof_heading_rule
    proof_lede_rule = css.split(
        '.proof-copy > p:not(.eyebrow) {',
        1,
    )[1].split('}', 1)[0]
    assert 'font-size: clamp(0.98rem, 1.5vw, 1.1rem);' in proof_lede_rule
    assert 'line-height: 1.72;' in proof_lede_rule
    proof_axiom_rule = css.split('.proof-copy .proof-axiom {', 1)[1].split('}', 1)[0]
    assert 'padding: 0.575rem 0 0.575rem 1.15rem;' in proof_axiom_rule
    assert 'border-left: 4px solid var(--status-success-vivid);' in proof_axiom_rule
    install_lede_rule = css.split('.install-copy .install-lede {', 1)[1].split(
        '}',
        1,
    )[0]
    assert 'font-size: clamp(0.98rem, 1.5vw, 1.1rem);' in install_lede_rule
    assert 'line-height: 1.72;' in install_lede_rule
    install_meta_rule = css.split('.install-copy .install-meta {', 1)[1].split(
        '}',
        1,
    )[0]
    assert 'color: var(--faint);' in install_meta_rule
    assert 'font-family: var(--font-mono);' in install_meta_rule
    assert 'font-size: 0.72rem;' in install_meta_rule
    assert 'line-height: 1.6;' in install_meta_rule
    card_heading_rule = css.split('.principle-card h3 {', 1)[1].split('}', 1)[0]
    assert 'font-family: var(--font-display);' in card_heading_rule
    assert 'font-weight: 900;' in card_heading_rule
    assert 'letter-spacing: 0;' in card_heading_rule
    for selector in (
        '.principle-card {',
        '.proof-outcome {',
        '.setup-step {',
    ):
        card_rule = css.split(selector, 1)[1].split('}', 1)[0]
        assert 'min-height:' not in card_rule
    assert '.principle-card--compact {' not in css
    overview_rule = css.split('.today {', 1)[1].split('}', 1)[0]
    assert 'gap: var(--section-content-gap);' in overview_rule
    assert 'align-content: safe center;' in overview_rule
    setup_scene_rule = css.split('.setup-scene {', 1)[1].split('}', 1)[0]
    assert (
        'grid-template-columns: minmax(15rem, 0.82fr) '
        'minmax(22rem, 1.18fr);'
    ) in setup_scene_rule
    assert 'grid-template-areas: "copy path";' in setup_scene_rule
    assert 'grid-template-rows: auto;' in setup_scene_rule
    assert 'gap: var(--section-content-gap);' in setup_scene_rule
    repo_rule = css.split('.repo-link {', 1)[1].split('}', 1)[0]
    assert 'grid-column: 1;' in repo_rule
    assert 'grid-row: 1;' in repo_rule
    assert 'align-self: end;' in repo_rule
    assert 'justify-self: start;' in repo_rule
    narrow_rule = css.split('@media (max-width: 800px) {', 1)[1]
    narrow_setup_rule = narrow_rule.split('.setup-scene {', 1)[1].split('}', 1)[0]
    assert '"copy"\n            "path"\n            "repo";' in narrow_setup_rule
    assert 'grid-template-rows: auto;' in narrow_setup_rule
    narrow_repo_rule = narrow_rule.split('.repo-link {', 1)[1].split('}', 1)[0]
    assert 'grid-area: repo;' in narrow_repo_rule
    assert 'grid-column:' not in narrow_repo_rule
    assert 'grid-row:' not in narrow_repo_rule
    assert 'align-self: start;' in narrow_repo_rule
    assert 'justify-self: center;' in narrow_repo_rule
    assert 'animation-iteration-count: 1 !important' in css
    assert 'animation: ghostBlink 6.4s linear infinite;' in css
    assert '@keyframes ghostBlink' in css
    assert '@keyframes ghostSymbolPeek' in css
    assert '@keyframes ghostStrokeCounterPeek' in css
    shared_aura_animation = (
        'animation: ghostAuraBloomA var(--ghost-aura-duration) '
        'ease-in-out infinite alternate;'
    )
    assert css.count(shared_aura_animation) == 2
    assert (
        'animation: ghostAuraBloomB var(--ghost-aura-duration) '
        'ease-in-out infinite alternate;' in css
    )
    assert '--ghost-aura-duration: 21s;' in css
    assert '--hero-paint-bleed: clamp(3rem, 6vw, 6rem);' in css
    assert 'width: calc(100% + (2 * var(--hero-paint-bleed)));' in css
    assert 'margin-inline: calc(-1 * var(--hero-paint-bleed));' in css
    assert 'padding-inline: var(--hero-paint-bleed);' in css
    assert 'transition: opacity 0.24s ease;' in css
    assert '.ghost-stage--returning' not in css
    assert 'animation-play-state: paused;' not in css
    assert '-5.2s' not in css
    assert '-3.9s' not in css
    assert '@keyframes ghostSolidPhase' not in css
    assert '@keyframes ghostOutlinePhase' not in css
    assert '@keyframes ghostPresence' not in css
    assert 'landing-ghost__eye-cutouts' in css
    assert '.ghost-stage--vanished' in css
    assert '.site-header' not in css
    assert '.wordmark' not in css
    assert '.header-link' not in css
    assert 'min-height: 100svh;' in css
    assert 'scroll-snap-type:' not in css
    assert 'font-size: clamp(3.3rem, 6.3vw, 5.25rem);' in css
    assert 'max-width: 14ch;' in css
    assert 'linear-gradient(' in css
    assert '135deg' in css
    assert 'var(--status-success-vivid) 28%' in css
    assert 'var(--status-success-vivid) 68%' in css
    assert (
        'h1,\n'
        '    .section-heading h2,\n'
        '    .proof-copy h2,\n'
        '    .install-copy h2 {'
    ) in css
    assert 'h1,\n    h2,\n    h3 {\n        color: transparent;' not in css
    assert '-webkit-text-fill-color: transparent;' in css
    assert '-webkit-text-fill-color: CanvasText;' in css
    assert '.scene {' in css
    scene_rule = css.split('.scene {', 1)[1].split('}', 1)[0]
    assert 'min-height:' not in scene_rule
    assert 'padding-block: var(--scene-padding);' in scene_rule
    assert 'position: relative;' in scene_rule
    pager_scene_rule = css.split(
        'html.scene-pager-ready .scene {',
        1,
    )[1].split('}', 1)[0]
    assert 'height: 100svh;' in pager_scene_rule
    assert 'min-height: 100svh;' in pager_scene_rule
    assert 'overflow-y: auto;' in pager_scene_rule
    assert 'overscroll-behavior-y: contain;' in pager_scene_rule
    assert 'padding-block: var(--scene-padding);' in pager_scene_rule
    assert 'html.scene-pager-ready .scene:not(.scene--active) {' in css
    assert 'display: none !important;' in css
    assert '.scene-nav {' in css
    assert 'position: fixed;' in css
    assert '--pattern-grid-step: 24px;' in css
    assert '--pattern-grid-origin: 1px;' in css
    assert '--pattern-grid-nav-offset: 17px;' in css
    assert 'background-size: var(--pattern-grid-step) var(--pattern-grid-step);' in css
    assert (
        'background-position: calc(env(safe-area-inset-left, 0px) + '
        'var(--pattern-grid-nav-offset)) 0;'
    ) in css
    assert (
        'inset-inline-start: calc(env(safe-area-inset-left, 0px) + '
        'var(--pattern-grid-nav-offset));'
    ) in css
    assert 'inset-block-start: var(--scene-nav-grid-center, 50%);' in css
    assert 'height: var(--pattern-grid-step);' in css
    assert 'gap: 0;' in css.split('.scene-nav {', 1)[1].split('}', 1)[0]
    assert 'margin-inline-start: var(--pattern-grid-origin);' in css
    assert 'translate: -50% 0;' in css
    assert 'transform: translateY(-50%);' in css
    assert '.scene-nav__marker[aria-current="step"] span {' in css
    active_marker_rule = css.split(
        '.scene-nav__marker[aria-current="step"] span {', 1
    )[1].split('}', 1)[0]
    assert 'background: var(--text);' in active_marker_rule
    assert 'box-shadow' not in active_marker_rule
    assert '.scene-nav__marker:focus-visible {' in css
    assert '@keyframes sceneEnterFromAbove' in css
    assert '@keyframes sceneEnterFromBelow' in css
    assert '.beyond__overview,' in css
    assert '.beyond__proof {' in css
    proof_rule = css.split('.beyond__proof {', 1)[1].split('}', 1)[0]
    assert 'align-items: start;' in proof_rule
    assert 'border-top:' not in proof_rule
    assert 'border-radius:' not in proof_rule
    assert 'box-shadow:' not in proof_rule
    assert 'background:' not in proof_rule
    assert '.beyond__proof::before' not in css
    proof_example_rule = css.split('.proof-example {', 1)[1].split('}', 1)[0]
    assert 'color: var(--accent-card-ink);' in proof_example_rule
    assert 'display: grid;' in proof_example_rule
    assert 'border: 0;' in proof_example_rule
    assert 'background: transparent;' in proof_example_rule
    assert 'box-shadow: none;' in proof_example_rule
    assert '.proof-request,\n.proof-flow {' in css
    proof_card_rule = css.split('.proof-request,\n.proof-flow {', 1)[1].split('}', 1)[0]
    assert 'border: 1px solid var(--line);' in proof_card_rule
    assert 'border-radius: 1rem;' in proof_card_rule
    assert 'background: var(--surface-accent-card);' in proof_card_rule
    assert '.proof-flow li:last-child {' in css
    assert '.proof-outcome--open {' not in css
    proof_outcomes_rule = css.split('.proof-outcomes {', 1)[1].split('}', 1)[0]
    assert 'grid-template-columns: minmax(0, 1fr);' in proof_outcomes_rule
    proof_done_rule = css.split('.proof-outcome--done {', 1)[1].split('}', 1)[0]
    assert 'border-color: var(--line);' in proof_done_rule
    assert 'var(--surface-accent-card) 78%, var(--color-text)' in proof_done_rule
    proof_claim_rule = css.split('.proof-outcome__claim {', 1)[1].split('}', 1)[0]
    assert 'font-family: var(--font-display);' in proof_claim_rule
    assert 'font-weight: 900;' in proof_claim_rule
    assert '.proof-cadence {' not in css
    proof_request_heading_rule = css.split('.proof-request p {', 1)[1].split('}', 1)[0]
    assert 'font-family: var(--font-display);' in proof_request_heading_rule
    assert 'font-weight: 900;' in proof_request_heading_rule
    assert '.principle-card__icon {' in css
    assert '.principle-card__icon svg {' in css
    assert '.setup-path {' in css
    setup_path_rule = css.split('.setup-path {', 1)[1].split('}', 1)[0]
    assert 'grid-template-columns: minmax(0, 1fr);' in setup_path_rule
    setup_step_rule = css.split('.setup-step {', 1)[1].split('}', 1)[0]
    assert 'color: var(--accent-card-ink);' in setup_step_rule
    assert 'border: 1px solid var(--line);' in setup_step_rule
    assert 'background: var(--surface-accent-card);' in setup_step_rule
    assert 'box-shadow: none;' in setup_step_rule
    setup_step_heading_rule = css.split('.setup-step h3 {', 1)[1].split('}', 1)[0]
    assert 'color: var(--accent-card-ink);' in setup_step_heading_rule
    assert '.setup-step--launch {' in css
    launch_step_rule = css.split('.setup-step--launch {', 1)[1].split('}', 1)[0]
    assert 'grid-template-columns: minmax(0, 1fr);' in launch_step_rule
    assert 'grid-column:' not in launch_step_rule
    assert '.install-terminal {' in css
    install_terminal_rule = css.split('\n.install-terminal {', 1)[1].split('}', 1)[0]
    assert 'background: #151a17;' in install_terminal_rule
    assert 'scrollbar-gutter: stable;' in css
    assert '.install-commands code {' in css
    assert 'min-block-size: 5.55em;' in css
    assert '.install {' not in css
    assert '.install-copy-button:focus-visible {' in css
    assert 'min-width: 2.25rem;' in css
    assert 'min-height: 2.25rem;' in css
    assert '.hero-actions' not in css
    assert '.button--primary' not in css
    assert '.button--quiet' not in css
    assert 'mix-blend-mode: screen' not in css
    assert 'filter: blur(1.5rem);' in css
    ghost_body_rule = re.search(
        r'\.landing-ghost__body\s*\{(?P<body>.*?)\}',
        css,
        re.DOTALL,
    )
    assert ghost_body_rule is not None
    assert 'fill: var(--status-success-vivid, #baff8a);' in ghost_body_rule.group(
        'body',
    )
    assert 'filter:' not in ghost_body_rule.group('body')
    for selector in (
        '.landing-ghost',
        '.landing-ghost__icon',
        '.landing-ghost__outline',
    ):
        rule = re.search(
            rf'{re.escape(selector)}\s*\{{(?P<body>.*?)\}}',
            css,
            re.DOTALL,
        )
        assert rule is not None
        assert 'filter:' not in rule.group('body')
    heading_rule = re.search(r'^h1\s*\{(?P<body>.*?)\}', css, re.DOTALL | re.MULTILINE)
    assert heading_rule is not None
    assert 'font-family: var(--font-display);' in heading_rule.group('body')
    assert 'font-weight: 900;' in heading_rule.group('body')
    assert 'letter-spacing: 0;' in heading_rule.group('body')
    install_heading_rule = re.search(
        r'^\.install-copy h2\s*\{(?P<body>.*?)\}',
        css,
        re.DOTALL | re.MULTILINE,
    )
    assert install_heading_rule is not None
    assert 'font-family: var(--font-display);' in install_heading_rule.group('body')
    assert 'max-width: 11ch;' in install_heading_rule.group('body')
    assert 'font-size: clamp(2.3rem, 4.6vw, 4.2rem);' in install_heading_rule.group(
        'body'
    )
    assert 'font-weight: 900;' in install_heading_rule.group('body')
    assert 'letter-spacing: 0;' in install_heading_rule.group('body')
    assert 'line-height: 0.98;' in install_heading_rule.group('body')
    assert 'url(http' not in css


def test_landing_setup_copy_control_reads_the_rendered_command():
    html = LANDING_PATH.read_text(encoding='utf-8')
    css = LANDING_CSS_PATH.read_text(encoding='utf-8')
    script = LANDING_JS_PATH.read_text(encoding='utf-8')

    assert 'From the Ollmo checkout' not in html
    assert 'install-terminal__dots' not in html
    assert '<button class="install-copy-button" type="button"' in html
    assert 'aria-label="Copy the Ollmo installation commands"' in html
    assert 'title="Copy the Ollmo installation commands"' in html
    assert '<svg viewBox="0 0 24 24" aria-hidden="true"' in html
    assert 'data-copy-target="install-commands"' in html
    assert 'id="install-commands"' in html
    assert 'class="install-copy-status" role="status"' in html

    assert '.install-terminal__actions {' in css
    assert 'margin-left: auto;' in css
    assert '.install-copy-button:focus-visible {' in css
    assert 'outline: 2px solid var(--status-success-vivid);' in css
    assert '@media (forced-colors: active)' in css
    assert 'outline-color: Highlight;' in css
    reduced_motion_rule = css.split(
        '@media (prefers-reduced-motion: reduce) {',
        1,
    )[1]
    assert 'animation-duration: 0.01ms !important;' in reduced_motion_rule

    assert "copyButton.dataset.copyTarget" in script
    assert "document.getElementById(targetId)" in script
    assert "target.textContent.trim()" in script
    assert "navigator.clipboard.writeText(text)" in script
    assert "document.execCommand('copy')" in script
    assert "setCopyFeedback('Copied', 'Installation commands copied')" in script
    assert "setCopyFeedback('Select + copy', 'Command selected; copy it manually')" in script


def test_landing_scene_pager_keeps_one_hash_addressable_scene_active():
    html = LANDING_PATH.read_text(encoding='utf-8')
    css = LANDING_CSS_PATH.read_text(encoding='utf-8')
    script = LANDING_JS_PATH.read_text(encoding='utf-8')

    assert '<div class="beyond">' not in html
    assert '<section class="beyond"' not in html
    assert '<section id="principles" class="beyond__overview scene"' in html
    assert '<h2 id="proof-title">One request becomes connected work.</h2>' in html
    assert 'class="scene-next"' not in html

    assert '.scene[hidden] {' in css
    assert 'html.scene-pager-ready .scene:not(.scene--active) {' in css
    assert 'html.scene-pager-ready .scene--active {' in css
    assert 'position: fixed;' in css.split('.scene-nav {', 1)[1].split('}', 1)[0]
    assert css.split('.scene-nav {', 1)[1].split('}', 1)[0].count('flex-direction: column;') == 1
    assert '@media print {' in css
    assert 'html.scene-pager-ready .scene[hidden] {' in css
    assert '@media (min-width: 801px) and (max-height: 760px)' in css
    assert '--section-content-gap: 2.5rem;' in css

    assert "Array.from(document.querySelectorAll('.scene'))" in script
    assert "document.querySelector('.scene-nav')" in script
    assert "navigation.querySelectorAll('.scene-nav__marker')" in script
    assert 'scene.hidden = !active;' in script
    assert "scene.classList.toggle('scene--active', active)" in script
    assert 'document.body.dataset.activeScene = activeScene.id;' in script
    assert "window.history.pushState({ scene: activeScene.id }, '', targetHash)" in script
    assert "window.addEventListener('hashchange', syncSceneFromLocation)" in script
    assert "window.addEventListener('popstate', syncSceneFromLocation)" in script
    assert "window.addEventListener('wheel'" in script
    assert 'const PATTERN_GRID_STEP = 24;' in script
    assert 'const PATTERN_GRID_ORIGIN = 1;' in script
    assert 'Math.floor(' in script
    assert "navigation.style.setProperty('--scene-nav-grid-center'" in script
    assert "window.addEventListener('resize', alignNavigationToGrid)" in script
    assert 'alignNavigationToGrid();' in script
    assert "main.addEventListener('wheel'" not in script
    assert "main.addEventListener('touchstart'" in script
    assert "main.addEventListener('touchend'" in script
    assert "main.addEventListener('touchcancel'" in script
    assert 'const WHEEL_HOP_THRESHOLD = 1;' in script
    assert 'const WHEEL_GESTURE_GAP_MS = 80;' in script
    assert 'const WHEEL_NOTCH_REARM_MS = 48;' in script
    assert 'const WHEEL_PIXEL_NOTCH_GAP_MS = 36;' in script
    assert 'const WHEEL_REIMPULSE_DELAY_MS = 140;' in script
    assert "wheelPhase = 'consumed';" in script
    assert 'consumedWheelCanRearm' in script
    assert 'activeScene.scrollTop += delta;' in script
    assert 'Math.abs(event.deltaX) >= Math.abs(event.deltaY)' in script
    assert 'keepNavigationLockedUntilWheelIdle' not in script
    assert 'navigationLocked' not in script
    assert 'WHEEL_INTENT_IDLE_MS' not in script
    assert 'const discreteStep = event.deltaMode !== 0' in script
    assert 'wheelDistance < WHEEL_HOP_THRESHOLD' in script
    assert 'wheelDistance < 56' not in script
    assert 'window.getComputedStyle(scene).paddingBottom' in script
    assert 'const tolerance = sceneEdgeTolerance(scene);' in script
    assert 'Math.abs(distance) < 52' in script
    assert "document.addEventListener('keydown'" in script
    assert "link.setAttribute('aria-current', 'step')" in script
    assert "link.removeAttribute('aria-current')" in script
    assert "window.history.scrollRestoration = 'manual'" in script
    assert 'const scrollingElement = document.scrollingElement || document.documentElement;' in script
    assert 'window.requestAnimationFrame(resetRootScroll);' in script
    assert "document.documentElement.classList.add('scene-pager-ready')" in script


@pytest.mark.skipif(shutil.which('node') is None, reason='Node.js is required')
def test_landing_scene_pager_handles_wheel_and_trackpad_gestures_once():
    result = _run_node(
        r"""
        const fs = require('fs');
        const vm = require('vm');

        const source = fs.readFileSync('site/landing.js', 'utf8');
        const nextModule = source.indexOf('\n\n(() => {', 1);
        const pagerSource = source.slice(0, nextModule);
        let clock = 0;

        function makeClassList() {
          const values = new Set();
          return {
            add(...names) {
              names.forEach((name) => values.add(name));
            },
            remove(...names) {
              names.forEach((name) => values.delete(name));
            },
            toggle(name, force) {
              if (force) values.add(name);
              else values.delete(name);
            },
          };
        }

        const sceneIds = ['intro', 'principles', 'proof', 'about', 'setup'];
        const headings = {};
        const scenes = sceneIds.map((id) => {
          headings[`${id}-title`] = { textContent: `${id} title` };
          return {
            id,
            hidden: false,
            scrollTop: 0,
            clientHeight: 700,
            scrollHeight: id === 'setup' ? 1000 : id === 'principles' ? 724 : 700,
            classList: makeClassList(),
            getAttribute(name) {
              return name === 'aria-labelledby' ? `${id}-title` : null;
            },
            focus() {},
          };
        });

        const linkHandlers = [];
        const links = sceneIds.map((id, index) => ({
          dataset: {},
          attributes: {},
          addEventListener(type, handler) {
            if (type === 'click') linkHandlers[index] = handler;
          },
          setAttribute(name, value) {
            this.attributes[name] = value;
          },
          removeAttribute(name) {
            delete this.attributes[name];
          },
        }));
        const mainHandlers = {};
        const main = {
          addEventListener(type, handler) {
            mainHandlers[type] = handler;
          },
        };
        const navigation = {
          dataset: {},
          style: {
            setProperty(name, value) {
              navigation.dataset[name] = value;
            },
          },
          querySelectorAll(selector) {
            return selector === '.scene-nav__marker' ? links : [];
          },
        };
        const status = { textContent: '' };
        const windowHandlers = {};
        const documentHandlers = {};
        const location = { hash: '#intro' };
        const history = {
          scrollRestoration: 'auto',
          pushState(_state, _title, hash) {
            location.hash = hash;
          },
        };
        const documentElement = { scrollTop: 0, classList: makeClassList() };
        const body = { scrollTop: 0, dataset: {} };
        const document = {
          body,
          documentElement,
          scrollingElement: documentElement,
          querySelectorAll(selector) {
            return selector === '.scene' ? scenes : [];
          },
          querySelector(selector) {
            if (selector === '.scene-nav') return navigation;
            if (selector === '.scene-nav__status') return status;
            return null;
          },
          getElementById(id) {
            if (id === 'main') return main;
            return headings[id] || null;
          },
          addEventListener(type, handler) {
            documentHandlers[type] = handler;
          },
        };
        const window = {
          innerHeight: 700,
          history,
          location,
          performance: { now: () => clock },
          requestAnimationFrame(callback) {
            callback();
          },
          getComputedStyle() {
            return { paddingBottom: '32px' };
          },
          addEventListener(type, handler) {
            windowHandlers[type] = handler;
          },
        };

        vm.runInNewContext(pagerSource, {
          document,
          window,
          Element: function Element() {},
        });

        function activeScene() {
          return body.dataset.activeScene;
        }

        function clickScene(index) {
          linkHandlers[index]({ preventDefault() {} });
        }

        function wheel({
          deltaY,
          deltaX = 0,
          deltaMode = 0,
          advance = 16,
          inside = true,
        }) {
          clock += advance;
          let prevented = false;
          const current = scenes.find((scene) => scene.id === activeScene());
          windowHandlers.wheel({
            ctrlKey: false,
            deltaY,
            deltaX,
            deltaMode,
            composedPath() {
              return inside ? [current, main, document, window] : [body, document, window];
            },
            preventDefault() {
              prevented = true;
            },
          });
          return prevented;
        }

        const observations = {};

        observations.initialGridCenter = navigation.dataset['--scene-nav-grid-center'];
        window.innerHeight = 844;
        windowHandlers.resize();
        observations.resizedGridCenter = navigation.dataset['--scene-nav-grid-center'];
        window.innerHeight = 700;
        windowHandlers.resize();

        observations.discretePrevented = wheel({
          deltaY: 3,
          deltaMode: 1,
          inside: false,
          advance: 200,
        });
        observations.afterDiscreteNotch = activeScene();
        wheel({ deltaY: 3, deltaMode: 1, inside: false, advance: 50 });
        observations.afterRapidDiscreteNotch = activeScene();

        clickScene(0);
        wheel({ deltaY: 0.4, advance: 200 });
        observations.afterGentleStart = activeScene();
        wheel({ deltaY: 0.6 });
        observations.afterGentleBurst = activeScene();

        [20, 12, 7, 4, 2].forEach((deltaY) => wheel({ deltaY }));
        observations.afterMomentumTail = activeScene();
        wheel({ deltaY: 8, advance: 100 });
        observations.afterRenewedStart = activeScene();
        wheel({ deltaY: 14 });
        observations.afterRenewedBurst = activeScene();

        clickScene(4);
        observations.outsideLongScenePrevented = wheel({
          deltaY: 40,
          inside: false,
          advance: 200,
        });
        observations.afterOutsideLongScene = activeScene();
        observations.longSceneScrollTop = scenes[4].scrollTop;

        clickScene(0);
        observations.horizontalPrevented = wheel({
          deltaY: 40,
          deltaX: 50,
          advance: 200,
        });
        observations.afterHorizontalGesture = activeScene();

        console.log(JSON.stringify(observations));
        """
    )

    assert result == {
        'initialGridCenter': '337px',
        'resizedGridCenter': '409px',
        'discretePrevented': True,
        'afterDiscreteNotch': 'principles',
        'afterRapidDiscreteNotch': 'proof',
        'afterGentleStart': 'intro',
        'afterGentleBurst': 'principles',
        'afterMomentumTail': 'principles',
        'afterRenewedStart': 'proof',
        'afterRenewedBurst': 'proof',
        'outsideLongScenePrevented': True,
        'afterOutsideLongScene': 'setup',
        'longSceneScrollTop': 40,
        'horizontalPrevented': False,
        'afterHorizontalGesture': 'intro',
    }


def test_landing_serves_bundled_montserrat_black_and_license():
    font_path = LANDING_SITE_ROOT / 'fonts' / 'Montserrat-Black-latin.woff2'
    license_path = LANDING_SITE_ROOT / 'fonts' / 'OFL.txt'

    assert font_path.is_file()
    assert len(font_path.read_bytes()) > 10_000
    assert 'SIL OPEN FONT LICENSE Version 1.1' in license_path.read_text(
        encoding='utf-8',
    )


def test_landing_ambient_vanish_uses_dashboard_timing():
    script = LANDING_JS_PATH.read_text(encoding='utf-8')
    dashboard_html = (REPO_ROOT / 'ollmo_webUI.html').read_text(encoding='utf-8')
    shared_timing = (
        '78000 + Math.floor(Math.random() * 54000)',
        'Math.random() < 0.1',
        '2000 + Math.floor(Math.random() * 1500)',
    )
    for expression in shared_timing:
        assert expression in script
        assert expression in dashboard_html

    assert "document.querySelector('.ghost-stage')" in script
    assert "ghostStage.classList.add('ghost-stage--vanished')" in script
    assert 'ghost-stage--returning' not in script
    assert '18000 + Math.floor(Math.random() * 6001)' in script
    assert "'--ghost-aura-duration'" in script
    assert 'AURA_FADE_OUT_MS' not in script
    assert 'AURA_FADE_IN_MS' not in script
    assert "matchMedia('(prefers-reduced-motion: reduce)')" in script
    assert "window.addEventListener('pagehide', stopAmbientVanish)" in script
    assert "window.addEventListener('pageshow'" in script
    assert "reducedMotion.addEventListener('change', startAmbientVanish)" in script
    assert 'if (event.persisted) startAmbientVanish();' in script


def test_landing_inherits_dashboard_surface_and_ghost_language():
    landing_css = LANDING_CSS_PATH.read_text(encoding='utf-8')
    dashboard_css = DASHBOARD_CSS_PATH.read_text(encoding='utf-8')

    shared_tokens = (
        '--color-bg: #111315;',
        '--surface-page: var(--color-bg);',
        '--status-success-vivid: #baff8a;',
    )
    for token in shared_tokens:
        assert token in landing_css
        assert token in dashboard_css

    assert 'background-image: var(--noise);' in landing_css
    assert 'background-image: var(--pattern-grid);' in landing_css
    assert '--pattern-grid-step: 24px;' in landing_css
    assert 'background-size: var(--pattern-grid-step) var(--pattern-grid-step);' in landing_css
    assert 'mix-blend-mode: multiply;' in landing_css
    assert 'color: var(--status-success-vivid);' in landing_css
    assert _custom_property(landing_css, '--noise') == _custom_property(
        dashboard_css,
        '--noise',
    )
    assert _custom_property(landing_css, '--pattern-grid') == _custom_property(
        dashboard_css,
        '--pattern-grid',
    )

    shared_ghost_animations = (
        'ghostFloat',
        'ghostAuraBloomA',
        'ghostAuraBloomB',
        'ghostSymbolPeek',
        'ghostStrokeCounterPeek',
    )
    for animation in shared_ghost_animations:
        assert f'@keyframes {animation}' in landing_css
        assert f'@keyframes {animation}' in dashboard_css

    for animation in ('ghostSymbolPeek', 'ghostStrokeCounterPeek'):
        assert _keyframes_block(landing_css, animation) == _keyframes_block(
            dashboard_css,
            animation,
        )


def test_route_split_keeps_canonical_api_routes_registered():
    routes = {rule.rule for rule in app.url_map.iter_rules()}

    assert '/site/' in routes
    assert '/site/<path:filename>' in routes
    assert '/ollmo_landing.html' not in routes
    assert '/api/responses' in routes
    assert '/v1/responses' in routes
    assert '/api/runtime_manifest' in routes


def test_dashboard_cli_opens_direct_dashboard_route():
    result = subprocess.run(
        [str(OLLMO_CLI), 'dashboard'],
        cwd=REPO_ROOT,
        env={'PATH': '/usr/bin:/bin', 'OLLMO_DRY_RUN_OPEN': '1'},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == (
        'DRY RUN: would open dashboard at http://127.0.0.1:5001/'
    )


def test_version_cli_uses_single_python_version_source():
    expected = subprocess.run(
        [
            sys.executable,
            '-c',
            'from ollmo_core.version import __version__; print(f"Ollmo {__version__}")',
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    for command in ('version', '--version'):
        result = subprocess.run(
            [str(OLLMO_CLI), command],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert result.stdout.strip() == expected
