/* Threshold explorer: genuine vs impostor distance distributions (illustrative, normal-shaped). */
(function () {
  var el = document.getElementById("thr");
  var G = { mu: 0.19, sd: 0.07 }, I = { mu: 0.68, sd: 0.11 }; // same person / different people
  var state = { t: 0.30 };

  function pdf(x, d) { return Math.exp(-0.5 * Math.pow((x - d.mu) / d.sd, 2)) / (d.sd * Math.sqrt(2 * Math.PI)); }
  function cdf(x, d) { // Abramowitz-Stegun erf
    var z = (x - d.mu) / (d.sd * Math.SQRT2), t = 1 / (1 + 0.3275911 * Math.abs(z));
    var y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-z * z);
    return 0.5 * (1 + (z >= 0 ? y : -y));
  }
  function css(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }

  el.innerHTML =
    '<div><svg id="thr-svg" viewBox="0 0 560 250" role="img" aria-label="Distance distributions for the same person and different people, with the threshold"></svg>' +
    '<div class="slider-row" style="margin-top:10px"><label for="thr-range">Threshold</label><input id="thr-range" type="range" min="0.05" max="0.95" step="0.01" value="0.30"><output id="thr-val">0.30</output></div></div>' +
    '<div style="display:flex;flex-direction:column;gap:14px">' +
    '<div class="thr-stats"><div class="thr-stat far"><b id="far">0%</b><span>strangers let in (false accepts)</span></div><div class="thr-stat frr"><b id="frr">0%</b><span>owners turned away (false rejects)</span></div></div>' +
    '<div class="legend"><span><i style="background:var(--match)"></i>same person</span><span><i style="background:var(--nomatch)"></i>different people</span></div>' +
    '<p class="note" id="thr-say" style="margin:0;font-size:15px;color:var(--ink-2)"></p>' +
    '<div class="presets"><button class="btn small" type="button" data-t="0.30">0.30 (app.py to app2.py)</button><button class="btn small" type="button" data-t="0.35">0.35 (main.py)</button></div>' +
    '<p class="note" style="margin:0">Curves are illustrative; real ones depend on the model and the photos. Tune with <code>/verify-live-file-debug</code>, which returns the raw distance.</p></div>';

  var svg = el.querySelector("#thr-svg"), range = el.querySelector("#thr-range");
  var W = 560, H = 250, L = 30, R = 540, T = 20, B = 210;
  function X(v) { return L + v * (R - L); }
  var maxY = pdf(G.mu, G);
  function Y(p) { return B - p / maxY * (B - T - 10); }

  function curve(d, from, to) {
    var s = "M" + X(from) + " " + B, steps = 120;
    for (var i = 0; i <= steps; i++) { var x = from + (to - from) * i / steps; s += " L" + X(x).toFixed(1) + " " + Y(pdf(x, d)).toFixed(1); }
    return s + " L" + X(to) + " " + B + "Z";
  }
  function line(d) {
    var s = "", steps = 200;
    for (var i = 0; i <= steps; i++) { var x = i / steps; s += (i ? " L" : "M") + X(x).toFixed(1) + " " + Y(pdf(x, d)).toFixed(1); }
    return s;
  }

  function draw() {
    var t = state.t, m = css("--match"), n = css("--nomatch"), ink = css("--ink"), ink3 = css("--ink-3"), line2 = css("--line-2");
    var ticks = "";
    for (var v = 0; v <= 1.0001; v += 0.1) ticks += '<line x1="' + X(v) + '" x2="' + X(v) + '" y1="' + B + '" y2="' + (B + 5) + '" stroke="' + line2 + '"/><text x="' + X(v) + '" y="' + (B + 18) + '" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="10" fill="' + ink3 + '">' + v.toFixed(1) + "</text>";
    svg.innerHTML =
      '<line x1="' + L + '" x2="' + R + '" y1="' + B + '" y2="' + B + '" stroke="' + line2 + '"/>' + ticks +
      '<text x="' + R + '" y="' + (B + 36) + '" text-anchor="end" font-family="IBM Plex Mono, monospace" font-size="10" fill="' + ink3 + '">cosine distance</text>' +
      '<path d="' + curve(G, 0, 1) + '" fill="' + m + '" opacity=".14"/><path d="' + curve(I, 0, 1) + '" fill="' + n + '" opacity=".12"/>' +
      '<path d="' + curve(I, 0, t) + '" fill="' + n + '" opacity=".55"/>' +
      '<path d="' + curve(G, t, 1) + '" fill="' + css("--warn") + '" opacity=".6"/>' +
      '<path d="' + line(G) + '" fill="none" stroke="' + m + '" stroke-width="2"/><path d="' + line(I) + '" fill="none" stroke="' + n + '" stroke-width="2"/>' +
      '<line x1="' + X(t) + '" x2="' + X(t) + '" y1="' + (T - 6) + '" y2="' + B + '" stroke="' + ink + '" stroke-width="2" stroke-dasharray="4 3"/>' +
      '<text x="' + (X(t) + 6) + '" y="' + (T + 4) + '" font-family="IBM Plex Mono, monospace" font-size="11" font-weight="600" fill="' + ink + '">' + t.toFixed(2) + "</text>" +
      '<text x="' + (X(t) - 6) + '" y="' + (T + 22) + '" text-anchor="end" font-family="IBM Plex Mono, monospace" font-size="10" fill="' + m + '">Match</text>' +
      '<text x="' + (X(t) + 6) + '" y="' + (T + 22) + '" font-family="IBM Plex Mono, monospace" font-size="10" fill="' + n + '">No Match</text>';
    var far = cdf(t, I) * 100, frr = (1 - cdf(t, G)) * 100;
    el.querySelector("#far").textContent = pct(far);
    el.querySelector("#frr").textContent = pct(frr);
    el.querySelector("#thr-val").textContent = t.toFixed(2);
    el.querySelector("#thr-say").textContent = t < 0.2 ? "Very strict: almost no stranger gets through, but the owner is often turned away and has to retry."
      : t <= 0.4 ? "A balanced setting for unlocking: strangers rarely pass and owners usually get in first time."
      : t <= 0.55 ? "Loose: owners always get in, and the odd stranger starts slipping through."
      : "Far too loose: a large share of strangers would be accepted.";
  }
  function pct(v) { return v < 0.01 ? "<0.01%" : v < 1 ? v.toFixed(2) + "%" : v.toFixed(1) + "%"; }

  range.addEventListener("input", function () { state.t = Number(range.value); draw(); });
  el.querySelectorAll("[data-t]").forEach(function (b) { b.addEventListener("click", function () { state.t = Number(b.dataset.t); range.value = state.t; draw(); }); });
  document.addEventListener("themechange", draw);
  draw();
})();
