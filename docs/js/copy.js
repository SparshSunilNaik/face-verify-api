(function () {
  document.addEventListener("click", function (e) {
    var b = e.target.closest("[data-copy]");
    if (!b) return;
    var text = document.getElementById(b.getAttribute("data-copy")).textContent;
    function done(ok) { b.textContent = ok ? "Copied" : "Press Ctrl+C"; setTimeout(function () { b.textContent = "Copy"; }, 1400); }
    if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(text).then(function () { done(true); }, function () { done(false); });
    else {
      var r = document.createRange(); r.selectNodeContents(document.getElementById(b.getAttribute("data-copy")));
      var s = getSelection(); s.removeAllRanges(); s.addRange(r); done(false);
    }
  });
})();
