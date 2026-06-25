/* Persistent page zoom.
   Browsers don't remember their native zoom for local file:// pages, so each
   chapter opens back at 100%. This replaces the native zoom shortcuts with a
   CSS `zoom` we control and store in localStorage, then re-apply on every page
   load — so the zoom level follows you as you jump between chapters.

   Loaded as a plain (non-defer) script in <head> so the saved zoom is applied
   before the body paints, avoiding a flash of unscaled content. */
(function () {
  var KEY = 'vpbook-zoom';
  var MIN = 0.5, MAX = 3, STEP = 0.1;
  var root = document.documentElement;

  function clamp(v) {
    v = Math.round(v * 100) / 100;          // kill float drift (e.g. 1.0000001)
    return Math.min(MAX, Math.max(MIN, v));
  }
  function read() {
    try {
      var v = parseFloat(localStorage.getItem(KEY));
      return isFinite(v) && v > 0 ? clamp(v) : 1;
    } catch (e) { return 1; }
  }
  function save(v) {
    try { localStorage.setItem(KEY, String(v)); } catch (e) {}
  }

  var zoom = read();
  root.style.zoom = zoom;                    // apply immediately, pre-paint

  var badge;
  function showBadge() {
    if (!document.body) return;              // body not parsed yet (initial load)
    if (!badge) {
      badge = document.createElement('div');
      badge.className = 'zoom-badge';
      document.body.appendChild(badge);
    }
    badge.textContent = Math.round(zoom * 100) + '%';
    badge.classList.add('show');
    clearTimeout(showBadge._t);
    showBadge._t = setTimeout(function () { badge.classList.remove('show'); }, 900);
  }

  function set(v) {
    zoom = clamp(v);
    root.style.zoom = zoom;
    save(zoom);
    showBadge();
  }

  // Take over the native zoom shortcuts: Ctrl/Cmd with +, -, or 0.
  window.addEventListener('keydown', function (e) {
    if (!(e.ctrlKey || e.metaKey) || e.altKey || e.shiftKey) return;
    switch (e.key) {
      case '=': case '+': e.preventDefault(); set(zoom + STEP); break;
      case '-': case '_': e.preventDefault(); set(zoom - STEP); break;
      case '0':           e.preventDefault(); set(1);          break;
    }
  }, { passive: false });

  // Ctrl/Cmd + mouse wheel (pinch-zoom on trackpads sends this too).
  window.addEventListener('wheel', function (e) {
    if (!(e.ctrlKey || e.metaKey)) return;
    e.preventDefault();
    set(zoom + (e.deltaY < 0 ? STEP : -STEP));
  }, { passive: false });
})();
