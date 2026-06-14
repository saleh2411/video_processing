/* Adds a copy-LaTeX icon button to every .eq block.
   Runs as a deferred script so it executes after the DOM is parsed
   but before MathJax (loaded async from CDN) has typeset the math,
   which means each .eq still contains the raw "$$ ... $$" source. */
(function () {
  var ICON_COPY =
    '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" ' +
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
    '<rect x="9" y="9" width="11" height="11" rx="2"></rect>' +
    '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>' +
    '</svg>';
  var ICON_CHECK =
    '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" ' +
    'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
    '<polyline points="5 12 10 17 19 7"></polyline>' +
    '</svg>';
  var ICON_X =
    '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" ' +
    'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
    '<line x1="6" y1="6" x2="18" y2="18"></line>' +
    '<line x1="18" y1="6" x2="6" y2="18"></line>' +
    '</svg>';

  function stripDelims(src) {
    var s = src.trim();
    if (s.indexOf('$$') === 0) s = s.slice(2);
    if (s.lastIndexOf('$$') === s.length - 2) s = s.slice(0, -2);
    return s.trim();
  }

  function copyToClipboard(text, btn) {
    var done = function () {
      btn.classList.add('copied');
      btn.innerHTML = ICON_CHECK;
      btn.setAttribute('title', 'Copied');
      setTimeout(function () {
        btn.classList.remove('copied');
        btn.innerHTML = ICON_COPY;
        btn.setAttribute('title', 'Copy LaTeX source');
      }, 1200);
    };
    var fail = function () {
      btn.classList.add('failed');
      btn.innerHTML = ICON_X;
      btn.setAttribute('title', 'Copy failed — press Ctrl+C');
      setTimeout(function () {
        btn.classList.remove('failed');
        btn.innerHTML = ICON_COPY;
        btn.setAttribute('title', 'Copy LaTeX source');
      }, 1500);
    };
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(done, function () {
        legacyCopy(text) ? done() : fail();
      });
    } else {
      legacyCopy(text) ? done() : fail();
    }
  }

  function legacyCopy(text) {
    try {
      var ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.top = '-9999px';
      document.body.appendChild(ta);
      ta.select();
      var ok = document.execCommand('copy');
      document.body.removeChild(ta);
      return ok;
    } catch (e) {
      return false;
    }
  }

  function decorate(el) {
    if (el.dataset.eqDecorated === '1') return;
    var raw = stripDelims(el.textContent || '');
    if (!raw) return;
    el.dataset.latex = raw;
    el.dataset.eqDecorated = '1';
    el.classList.add('eq--has-copy');

    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'eq-copy-btn';
    btn.setAttribute('aria-label', 'Copy LaTeX source');
    btn.setAttribute('title', 'Copy LaTeX source');
    btn.innerHTML = ICON_COPY;
    btn.addEventListener('click', function (e) {
      e.preventDefault();
      e.stopPropagation();
      copyToClipboard(el.dataset.latex || '', btn);
    });
    el.appendChild(btn);
  }

  function run() {
    var nodes = document.querySelectorAll('.eq');
    for (var i = 0; i < nodes.length; i++) decorate(nodes[i]);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', run);
  } else {
    run();
  }
})();
