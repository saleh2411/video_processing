/* ===========================================================
   Video Processing — interactive practice-exam engine
   Content is authored as inert <template class="q"> blocks inside
   #quiz-src (so raw LaTeX like $\nabla u$ survives verbatim).
   This script turns them into a retakeable quiz with per-question
   Check + Hint buttons, hidden hint panels holding the matching
   course material, a running score, and a global Reset / Check-all.
   =========================================================== */
(function () {
  "use strict";

  function el(tag, cls, html) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html != null) n.innerHTML = html;
    return n;
  }
  function typeset(node) {
    if (window.MathJax && MathJax.typesetPromise) {
      MathJax.typesetPromise(node ? [node] : undefined).catch(function () {});
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    var mount = document.getElementById("quiz");
    var src = document.getElementById("quiz-src");
    if (!mount || !src) return;

    var templates = Array.prototype.slice.call(src.querySelectorAll("template.q"));
    if (!templates.length) return;

    var chapHref = mount.getAttribute("data-chapter-href");
    var chapLabel = mount.getAttribute("data-chapter-label") || "the chapter";

    // parse each template into a plain data record
    var data = templates.map(function (tpl) {
      var c = tpl.content;
      var textEl = c.querySelector(".q-text");
      var figEl = c.querySelector(".q-figure");
      var hintEl = c.querySelector(".q-hint");
      var opts = Array.prototype.slice.call(c.querySelectorAll(".q-options > li"))
        .map(function (li) { return li.innerHTML; });
      return {
        q: textEl ? textEl.innerHTML : "",
        figure: figEl ? figEl.outerHTML : "",
        hint: hintEl ? hintEl.innerHTML : "",
        options: opts,
        correct: parseInt(tpl.getAttribute("data-correct"), 10)
      };
    });

    var cards = [];

    /* ---------- sticky score / control bar ---------- */
    var bar = el("div", "quiz-bar");
    var score = el("div", "score", 'Score <b>0</b> / ' + data.length +
      ' <span class="pct">— pick an answer, then press Check</span>');
    var spacer = el("div", "spacer");
    var btnCheckAll = el("button", "act primary", "Check all");
    var btnReset = el("button", "act", "Reset / Retake");
    var meter = el("div", "meter");
    var meterFill = el("i"); meter.appendChild(meterFill);
    bar.appendChild(score); bar.appendChild(spacer);
    bar.appendChild(btnCheckAll); bar.appendChild(btnReset); bar.appendChild(meter);
    mount.appendChild(bar);

    /* ---------- build each question ---------- */
    data.forEach(function (item, qi) {
      var card = el("div", "qcard"); card.id = "q" + (qi + 1);

      var head = el("div", "qhead");
      head.appendChild(el("span", "qnum", "" + (qi + 1)));
      head.appendChild(el("span", "qtag", "Question " + (qi + 1)));
      head.appendChild(el("span", "qstate"));
      card.appendChild(head);

      var body = el("div", "qbody");
      body.appendChild(el("div", "qtext", item.q));
      if (item.figure) body.insertAdjacentHTML("beforeend", item.figure);

      var opts = el("ul", "opts");
      var inputs = [];
      item.options.forEach(function (optHtml, oi) {
        var li = document.createElement("li");
        var label = el("label", "opt");
        var input = document.createElement("input");
        input.type = "radio"; input.name = "q" + qi; input.value = "" + oi;
        var txt = el("span", "opt-txt", optHtml);
        var mark = el("span", "mark");
        label.appendChild(input); label.appendChild(txt); label.appendChild(mark);
        li.appendChild(label); opts.appendChild(li);
        inputs.push({ input: input, label: label });
        input.addEventListener("change", function () { ctrl.clear(); updateScore(); });
      });
      body.appendChild(opts);

      var actions = el("div", "qactions");
      var bCheck = el("button", "act primary", "Check");
      var bHint = el("button", "act hint", "💡 Hint");
      var feedback = el("div", "qfeedback");
      actions.appendChild(bCheck); actions.appendChild(bHint);
      body.appendChild(actions); body.appendChild(feedback);

      var hint = el("div", "hintpanel");
      hint.appendChild(el("span", "hint-src", "From the course material — explanation of this answer"));
      hint.insertAdjacentHTML("beforeend", item.hint || "<p>No hint available.</p>");
      if (chapHref) {
        var link = el("a", "chap-link", "Read the full section in " + chapLabel + " ›");
        link.href = chapHref; hint.appendChild(link);
      }
      body.appendChild(hint);

      card.appendChild(body);
      mount.appendChild(card);

      var ctrl = {
        checked: false, correct: false,
        selected: function () {
          for (var i = 0; i < inputs.length; i++) if (inputs[i].input.checked) return i;
          return -1;
        },
        clear: function () {
          card.classList.remove("answered-correct", "answered-wrong");
          feedback.className = "qfeedback"; feedback.innerHTML = "";
          inputs.forEach(function (o) { o.label.classList.remove("correct", "wrong", "reveal"); });
          ctrl.checked = false; ctrl.correct = false;
        },
        check: function () {
          ctrl.clear();
          var sel = ctrl.selected();
          if (sel < 0) {
            feedback.className = "qfeedback no show";
            feedback.innerHTML = "Pick an answer first.";
            return;
          }
          ctrl.checked = true;
          inputs.forEach(function (o, oi) { if (oi === item.correct) o.label.classList.add("correct"); });
          if (sel === item.correct) {
            ctrl.correct = true;
            card.classList.add("answered-correct");
            feedback.className = "qfeedback ok show";
            feedback.innerHTML = "Correct.";
          } else {
            inputs[sel].label.classList.add("wrong");
            card.classList.add("answered-wrong");
            feedback.className = "qfeedback no show";
            feedback.innerHTML = 'Not quite. Correct answer: <span class="ans">' +
              item.options[item.correct] + "</span>";
            typeset(feedback);
          }
          updateScore();
        }
      };

      bCheck.addEventListener("click", ctrl.check);
      bHint.addEventListener("click", function () {
        var open = hint.classList.toggle("open");
        bHint.innerHTML = open ? "💡 Hide hint" : "💡 Hint";
      });
      cards.push(ctrl);
    });

    var done = el("div", "quiz-done");
    mount.appendChild(done);

    function updateScore(showBanner) {
      var n = data.length, right = 0, checked = 0;
      cards.forEach(function (c) { if (c.checked) checked++; if (c.correct) right++; });
      var pct = n ? Math.round((right / n) * 100) : 0;
      score.innerHTML = 'Score <b>' + right + '</b> / ' + n +
        ' <span class="pct">' + (checked ? "(" + pct + "%)" : "— pick an answer, then press Check") + '</span>';
      meterFill.style.width = pct + "%";
      if (showBanner) {
        done.classList.add("show");
        var msg = right === n ? "Perfect score — every answer correct. 🎉"
          : right >= Math.ceil(n * 0.6) ? "Nice work. Review the ones you missed with the 💡 Hint buttons."
          : "Keep going — open the 💡 Hint on each miss to see the exact course material, then Reset and retake.";
        done.innerHTML = "<h3>You scored " + right + " / " + n + " (" + pct + "%)</h3><p>" + msg + "</p>";
        done.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
    }

    btnCheckAll.addEventListener("click", function () {
      cards.forEach(function (c) { c.check(); });
      updateScore(true);
    });
    btnReset.addEventListener("click", function () {
      cards.forEach(function (c) { c.clear(); });
      mount.querySelectorAll("input[type=radio]").forEach(function (r) { r.checked = false; });
      mount.querySelectorAll(".hintpanel.open").forEach(function (h) {
        h.classList.remove("open");
        var b = h.parentNode.querySelector(".act.hint"); if (b) b.innerHTML = "💡 Hint";
      });
      done.classList.remove("show");
      updateScore();
      mount.scrollIntoView({ behavior: "smooth", block: "start" });
    });

    typeset(mount);
  });
})();
