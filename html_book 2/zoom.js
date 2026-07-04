/* Persistent page zoom.
   A web page can't read or change the browser's *native* zoom, so this provides
   its own zoom: it overrides the Ctrl/Cmd +/- shortcuts and scales the page with
   the CSS `zoom` property, then carries the chosen level from page to page.

   Persistence is the tricky part on local file:// pages. Safari (and others to
   varying degrees) treats every file:// URL as a SEPARATE origin, so
   localStorage is NOT shared between chapters — the zoom would reset on each
   navigation. To survive that we primarily use window.name, which is preserved
   across same-tab navigations in every browser regardless of origin or scheme.
   localStorage is kept as a secondary store (works across tabs and when the
   book is served over http/https).

   Loaded as a plain (non-defer) script in <head> so the saved zoom is applied
   before the body paints, avoiding a flash of unscaled content. */
(function () {
  var KEY = 'vpbook-zoom';                  // localStorage key
  var TAG = 'vpzoom';                        // window.name marker: "vpzoom=1.2;"
  var MIN = 0.5, MAX = 3, STEP = 0.1;
  var root = document.documentElement;

  function clamp(v) {
    v = Math.round(v * 100) / 100;           // kill float drift (e.g. 1.0000001)
    return Math.min(MAX, Math.max(MIN, v));
  }

  // --- window.name store (survives same-tab navigation, even across file://) ---
  function readName() {
    try {
      var m = new RegExp('\\b' + TAG + '=([0-9.]+)').exec(window.name || '');
      return m ? parseFloat(m[1]) : NaN;
    } catch (e) { return NaN; }
  }
  function writeName(v) {
    try {
      var rest = (window.name || '').replace(new RegExp(TAG + '=[0-9.]+;?'), '');
      window.name = TAG + '=' + v + ';' + rest;
    } catch (e) {}
  }

  // --- localStorage store (cross-tab / when served over http) ---
  function readStore() {
    try {
      var v = parseFloat(localStorage.getItem(KEY));
      return isFinite(v) ? v : NaN;
    } catch (e) { return NaN; }
  }
  function writeStore(v) {
    try { localStorage.setItem(KEY, String(v)); } catch (e) {}
  }

  // Prefer window.name (most reliable across pages), then localStorage, else 100%.
  var initial = readName();
  if (!(isFinite(initial) && initial > 0)) initial = readStore();
  var zoom = isFinite(initial) && initial > 0 ? clamp(initial) : 1;
  root.style.zoom = zoom;                     // apply immediately, pre-paint
  writeName(zoom);                            // make sure it's seeded for next page

  var badge;
  function showBadge() {
    if (!document.body) return;               // body not parsed yet (initial load)
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
    writeName(zoom);
    writeStore(zoom);
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
