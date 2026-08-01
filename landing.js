(() => {
  'use strict';

  const scenes = Array.from(document.querySelectorAll('.scene'));
  const main = document.getElementById('main');
  const navigation = document.querySelector('.scene-nav');
  const sceneLinks = navigation
    ? Array.from(navigation.querySelectorAll('.scene-nav__marker'))
    : [];
  const status = document.querySelector('.scene-nav__status');
  if (!scenes.length || !main || !navigation || sceneLinks.length !== scenes.length) {
    return;
  }

  let activeIndex = 0;
  let wheelPhase = 'idle';
  let wheelDirection = 0;
  let wheelDistance = 0;
  let wheelLastAt = 0;
  let wheelLastMagnitude = 0;
  let wheelConsumedAt = 0;
  let wheelTailDecayed = false;
  let touchIntent = null;
  const WHEEL_HOP_THRESHOLD = 1;
  const WHEEL_GESTURE_GAP_MS = 80;
  const WHEEL_NOTCH_REARM_MS = 48;
  const WHEEL_PIXEL_NOTCH_GAP_MS = 36;
  const WHEEL_PIXEL_NOTCH_FLOOR = 32;
  const WHEEL_REIMPULSE_DELAY_MS = 140;
  const WHEEL_REIMPULSE_FLOOR = 6;
  const WHEEL_TAIL_DECAY_MAX = 4;
  const WHEEL_REIMPULSE_GROWTH = 1.5;
  const PATTERN_GRID_STEP = 24;
  const PATTERN_GRID_ORIGIN = 1;

  if ('scrollRestoration' in window.history) {
    window.history.scrollRestoration = 'manual';
  }

  function sceneIndexFromHash() {
    const hash = window.location.hash.slice(1);
    const index = scenes.findIndex((scene) => scene.id === hash);
    return index >= 0 ? index : 0;
  }

  function sceneName(scene) {
    const headingId = scene.getAttribute('aria-labelledby');
    const heading = headingId ? document.getElementById(headingId) : null;
    return heading ? heading.textContent.trim() : scene.id;
  }

  function resetWheelIntent() {
    wheelPhase = 'idle';
    wheelDirection = 0;
    wheelDistance = 0;
    wheelLastAt = 0;
    wheelLastMagnitude = 0;
    wheelConsumedAt = 0;
    wheelTailDecayed = false;
  }

  function consumeWheelIntent(direction, magnitude, timestamp) {
    wheelPhase = 'consumed';
    wheelDirection = direction;
    wheelDistance = 0;
    wheelLastAt = timestamp;
    wheelLastMagnitude = magnitude;
    wheelConsumedAt = timestamp;
    wheelTailDecayed = false;
  }

  function wheelDeltaInPixels(event, scene) {
    const modeFactor = event.deltaMode === 1
      ? 16
      : event.deltaMode === 2
        ? scene.clientHeight
        : 1;
    return event.deltaY * modeFactor;
  }

  function wheelTargetsScene(event, scene) {
    return typeof event.composedPath === 'function'
      && event.composedPath().includes(scene);
  }

  function consumedWheelCanRearm(
    direction,
    magnitude,
    discreteStep,
    pixelNotch,
    timestamp,
  ) {
    const gap = timestamp - wheelLastAt;
    const elapsed = timestamp - wheelConsumedAt;
    if (direction !== wheelDirection) return true;
    if (gap >= WHEEL_GESTURE_GAP_MS) return true;
    if (discreteStep && elapsed >= WHEEL_NOTCH_REARM_MS) return true;
    if (
      pixelNotch
      && gap >= WHEEL_PIXEL_NOTCH_GAP_MS
      && elapsed >= WHEEL_NOTCH_REARM_MS
    ) return true;
    return wheelTailDecayed
      && elapsed >= WHEEL_REIMPULSE_DELAY_MS
      && magnitude >= WHEEL_REIMPULSE_FLOOR
      && magnitude >= Math.max(
        WHEEL_REIMPULSE_FLOOR,
        wheelLastMagnitude * WHEEL_REIMPULSE_GROWTH,
      );
  }

  function resetRootScroll() {
    const scrollingElement = document.scrollingElement || document.documentElement;
    scrollingElement.scrollTop = 0;
    document.body.scrollTop = 0;
  }

  function alignNavigationToGrid() {
    const viewportCenter = window.innerHeight / 2;
    const row = Math.floor(
      (viewportCenter - PATTERN_GRID_ORIGIN) / PATTERN_GRID_STEP,
    );
    const snappedCenter = PATTERN_GRID_ORIGIN + (row * PATTERN_GRID_STEP);
    navigation.style.setProperty('--scene-nav-grid-center', `${snappedCenter}px`);
  }

  function activateScene(index, options = {}) {
    if (index < 0 || index >= scenes.length) return false;

    const direction = Math.sign(index - activeIndex);
    activeIndex = index;
    scenes.forEach((scene, sceneIndex) => {
      const active = sceneIndex === activeIndex;
      scene.hidden = !active;
      scene.classList.toggle('scene--active', active);
      scene.classList.remove('scene--from-above', 'scene--from-below');
      if (active && direction < 0) scene.classList.add('scene--from-above');
      if (active && direction > 0) scene.classList.add('scene--from-below');
    });

    const activeScene = scenes[activeIndex];
    resetRootScroll();
    window.requestAnimationFrame(resetRootScroll);
    activeScene.scrollTop = 0;
    document.body.dataset.activeScene = activeScene.id;
    navigation.dataset.scene = `${activeIndex + 1}`;
    sceneLinks.forEach((link, sceneIndex) => {
      const current = sceneIndex === activeIndex;
      const position = sceneIndex < activeIndex
        ? 'previous'
        : sceneIndex > activeIndex
          ? 'next'
          : 'current';
      link.dataset.position = position;
      link.setAttribute(
        'aria-label',
        `Section ${sceneIndex + 1} of ${scenes.length}: ${sceneName(scenes[sceneIndex])}`,
      );
      if (current) {
        link.setAttribute('aria-current', 'step');
      } else {
        link.removeAttribute('aria-current');
      }
    });

    if (status) {
      status.textContent = `Section ${activeIndex + 1} of ${scenes.length}: ${sceneName(activeScene)}`;
    }

    if (options.updateHistory) {
      const targetHash = `#${activeScene.id}`;
      if (window.location.hash !== targetHash) {
        window.history.pushState({ scene: activeScene.id }, '', targetHash);
      }
    }

    if (options.focus) {
      window.requestAnimationFrame(() => activeScene.focus({ preventScroll: true }));
    }

    resetWheelIntent();
    return true;
  }

  function moveScene(direction, options = {}) {
    return activateScene(activeIndex + direction, options);
  }

  function sceneEdgeTolerance(scene) {
    const paddingBottom = Number.parseFloat(
      window.getComputedStyle(scene).paddingBottom,
    );
    return Math.max(2, Number.isFinite(paddingBottom) ? paddingBottom : 0);
  }

  function sceneAtBoundary(scene, direction) {
    const tolerance = sceneEdgeTolerance(scene);
    if (direction < 0) return scene.scrollTop <= tolerance;
    return scene.scrollTop + scene.clientHeight >= scene.scrollHeight - tolerance;
  }

  sceneLinks.forEach((link, sceneIndex) => {
    link.addEventListener('click', (event) => {
      event.preventDefault();
      activateScene(sceneIndex, { updateHistory: true, focus: true });
    });
  });

  window.addEventListener('wheel', (event) => {
    if (
      event.ctrlKey
      || event.deltaY === 0
      || Math.abs(event.deltaX) >= Math.abs(event.deltaY)
    ) {
      return;
    }

    const activeScene = scenes[activeIndex];
    const delta = wheelDeltaInPixels(event, activeScene);
    const direction = Math.sign(delta);
    const magnitude = Math.abs(delta);
    const timestamp = window.performance.now();
    const discreteStep = event.deltaMode !== 0 && Math.abs(event.deltaY) >= 1;
    const pixelNotch = event.deltaMode === 0
      && magnitude >= WHEEL_PIXEL_NOTCH_FLOOR;

    if (wheelPhase === 'consumed') {
      if (!consumedWheelCanRearm(
        direction,
        magnitude,
        discreteStep,
        pixelNotch,
        timestamp,
      )) {
        event.preventDefault();
        if (magnitude <= WHEEL_TAIL_DECAY_MAX) wheelTailDecayed = true;
        wheelLastAt = timestamp;
        wheelLastMagnitude = magnitude;
        return;
      }
      resetWheelIntent();
    }

    if (!sceneAtBoundary(activeScene, direction)) {
      resetWheelIntent();
      if (!wheelTargetsScene(event, activeScene)) {
        event.preventDefault();
        activeScene.scrollTop += delta;
      }
      return;
    }

    event.preventDefault();
    const targetIndex = activeIndex + direction;
    if (targetIndex < 0 || targetIndex >= scenes.length) {
      return;
    }

    const gestureGap = timestamp - wheelLastAt;
    if (
      wheelPhase !== 'tracking'
      || wheelDirection !== direction
      || gestureGap >= WHEEL_GESTURE_GAP_MS
    ) {
      wheelPhase = 'tracking';
      wheelDirection = direction;
      wheelDistance = 0;
    }

    wheelDistance += magnitude;
    wheelLastAt = timestamp;
    wheelLastMagnitude = magnitude;
    if (!discreteStep && wheelDistance < WHEEL_HOP_THRESHOLD) return;

    if (moveScene(direction, { updateHistory: true, focus: false })) {
      consumeWheelIntent(direction, magnitude, timestamp);
    }
  }, { passive: false });

  main.addEventListener('touchstart', (event) => {
    if (event.touches.length !== 1) {
      touchIntent = null;
      return;
    }

    const activeScene = scenes[activeIndex];
    touchIntent = {
      index: activeIndex,
      startY: event.touches[0].clientY,
      canMovePrevious: sceneAtBoundary(activeScene, -1),
      canMoveNext: sceneAtBoundary(activeScene, 1),
    };
  }, { passive: true });

  main.addEventListener('touchend', (event) => {
    if (!touchIntent || event.changedTouches.length !== 1) {
      touchIntent = null;
      return;
    }

    const distance = event.changedTouches[0].clientY - touchIntent.startY;
    const direction = distance < 0 ? 1 : -1;
    const allowed = direction > 0
      ? touchIntent.canMoveNext
      : touchIntent.canMovePrevious;
    const sameScene = touchIntent.index === activeIndex;
    touchIntent = null;

    if (Math.abs(distance) < 52 || !allowed || !sameScene) return;
    moveScene(direction, { updateHistory: true, focus: false });
  }, { passive: true });

  main.addEventListener('touchcancel', () => {
    touchIntent = null;
  }, { passive: true });

  document.addEventListener('keydown', (event) => {
    const eventTarget = event.target instanceof Element ? event.target : null;
    const interactive = eventTarget?.closest(
      'a, button, input, textarea, select, summary, [contenteditable="true"]',
    );
    if (interactive) return;

    const direction = event.key === 'ArrowDown' || event.key === 'PageDown'
      ? 1
      : event.key === 'ArrowUp' || event.key === 'PageUp'
        ? -1
        : 0;
    if (!direction) return;

    const activeScene = scenes[activeIndex];
    if (!sceneAtBoundary(activeScene, direction)) return;
    if (!moveScene(direction, { updateHistory: true, focus: true })) return;
    event.preventDefault();
  });

  function syncSceneFromLocation() {
    activateScene(sceneIndexFromHash(), { updateHistory: false, focus: false });
  }

  window.addEventListener('hashchange', syncSceneFromLocation);
  window.addEventListener('popstate', syncSceneFromLocation);
  window.addEventListener('pageshow', syncSceneFromLocation);
  window.addEventListener('resize', alignNavigationToGrid);

  alignNavigationToGrid();
  activateScene(sceneIndexFromHash(), { updateHistory: false, focus: false });
  document.documentElement.classList.add('scene-pager-ready');
})();

(() => {
  'use strict';

  const copyButton = document.querySelector('.install-copy-button');
  if (!copyButton) return;

  const label = copyButton.querySelector('.install-copy-button__label');
  const status = document.querySelector('.install-copy-status');
  const targetId = copyButton.dataset.copyTarget;
  const target = targetId ? document.getElementById(targetId) : null;
  let resetTimer = null;

  function setCopyFeedback(message, accessibleMessage) {
    if (label) label.textContent = message;
    if (status) status.textContent = accessibleMessage;
    copyButton.setAttribute('aria-label', accessibleMessage);
    copyButton.title = accessibleMessage;
    if (resetTimer !== null) window.clearTimeout(resetTimer);
    resetTimer = window.setTimeout(() => {
      if (label) label.textContent = 'Copy';
      if (status) status.textContent = '';
      copyButton.setAttribute('aria-label', 'Copy the Ollmo installation commands');
      copyButton.title = 'Copy the Ollmo installation commands';
      resetTimer = null;
    }, 1800);
  }

  function fallbackCopy(text) {
    const previousFocus = document.activeElement;
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.setAttribute('readonly', '');
    textarea.setAttribute('aria-hidden', 'true');
    textarea.tabIndex = -1;
    textarea.style.position = 'fixed';
    textarea.style.left = '-9999px';
    document.body.appendChild(textarea);
    textarea.focus();
    textarea.select();
    textarea.setSelectionRange(0, textarea.value.length);

    let copied = false;
    try {
      copied = document.execCommand('copy');
    } finally {
      textarea.remove();
      if (previousFocus && typeof previousFocus.focus === 'function') {
        previousFocus.focus();
      } else {
        copyButton.focus();
      }
    }
    if (!copied) throw new Error('Copy command was rejected.');
  }

  async function copyCommands() {
    if (!target) {
      setCopyFeedback('Unavailable', 'Installation commands unavailable');
      return;
    }

    const text = target.textContent.trim();
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
      } else {
        fallbackCopy(text);
      }
      setCopyFeedback('Copied', 'Installation commands copied');
    } catch (_error) {
      try {
        fallbackCopy(text);
        setCopyFeedback('Copied', 'Installation commands copied');
      } catch (_fallbackError) {
        const selection = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents(target);
        selection.removeAllRanges();
        selection.addRange(range);
        setCopyFeedback('Select + copy', 'Command selected; copy it manually');
      }
    }
  }

  copyButton.addEventListener('click', copyCommands);
})();

(() => {
  'use strict';

  const ghostStage = document.querySelector('.ghost-stage');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  if (!ghostStage) return;

  let ambientTimer = null;
  let vanishTimer = null;

  function setRandomAuraDuration() {
    const auraDurationMs = 18000 + Math.floor(Math.random() * 6001);
    ghostStage.style.setProperty(
      '--ghost-aura-duration',
      `${auraDurationMs}ms`,
    );
  }

  function clearVanishState() {
    if (vanishTimer !== null) {
      window.clearTimeout(vanishTimer);
      vanishTimer = null;
    }
    ghostStage.classList.remove('ghost-stage--vanished');
  }

  function stopAmbientVanish() {
    if (ambientTimer !== null) {
      window.clearTimeout(ambientTimer);
      ambientTimer = null;
    }
    clearVanishState();
  }

  function scheduleAmbientVanish() {
    if (reducedMotion.matches) {
      stopAmbientVanish();
      return;
    }

    if (ambientTimer !== null) {
      window.clearTimeout(ambientTimer);
    }

    const delayMs = 78000 + Math.floor(Math.random() * 54000);
    ambientTimer = window.setTimeout(() => {
      ambientTimer = null;
      if (document.hidden) {
        scheduleAmbientVanish();
        return;
      }

      if (Math.random() < 0.1) {
        const hideDurationMs = 2000 + Math.floor(Math.random() * 1500);
        clearVanishState();
        ghostStage.classList.add('ghost-stage--vanished');
        vanishTimer = window.setTimeout(() => {
          clearVanishState();
          scheduleAmbientVanish();
        }, hideDurationMs);
        return;
      }

      scheduleAmbientVanish();
    }, delayMs);
  }

  function startAmbientVanish() {
    stopAmbientVanish();
    if (reducedMotion.matches) return;
    setRandomAuraDuration();
    scheduleAmbientVanish();
  }

  window.addEventListener('pagehide', stopAmbientVanish);
  window.addEventListener('pageshow', (event) => {
    if (event.persisted) startAmbientVanish();
  });
  reducedMotion.addEventListener('change', startAmbientVanish);

  startAmbientVanish();
})();
