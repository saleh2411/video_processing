/* Adds a "copy LaTeX" button to every .eq block.
   Runs as a deferred script so it executes after the DOM is parsed
   but before MathJax (loaded async from CDN) has typeset the math,
   which means each .eq still contains the raw "$$ ... $$" source. */
(function () {
  function stripDelims(src) {
    var s = src.trim();
    if (s.indexOf('$$') === 0) s = s.slice(2);
    if (s.lastIndexOf('$$') === s.length - 2) s = s.slice(0, -2);
    return s.trim();
  }

  function copyToClipboard(text, btn) {
    var done = function () {
      var old = btn.getAttribute('data-label') || btn.textContent;
      btn.classList.add('copied');
      btn.textContent = 'Copied';
      setTimeout(function () {
        btn.classList.remove('copied');
        btn.textContent = old;
      }, 1200);
    };
    var fail = function () {
      btn.classList.add('failed');
      btn.textContent = 'Press Ctrl+C';
      setTimeout(function () {
        btn.classList.remove('failed');
        btn.textContent = btn.getAttribute('data-label') || 'Copy LaTeX';
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
    btn.setAttribute('data-label', 'Copy LaTeX');
    btn.textContent = 'Copy LaTeX';
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
