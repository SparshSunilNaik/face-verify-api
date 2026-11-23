/* Pipeline playground + hero rig. Mirrors app2.py: _detect_best_face_with_rotations -> _to_embedding -> _verify_emb. */
(function () {
  var F = window.Faces;
  var DEFAULT_THRESHOLD = 0.30, MIN_FACE_PROB = 0.95;
  var reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------- deterministic embeddings with an exact cosine distance ---------- */
  function rng(seed) {
    var h = 2166136261;
    for (var i = 0; i < seed.length; i++) { h ^= seed.charCodeAt(i); h = Math.imul(h, 16777619); }
    return function () { h ^= h << 13; h ^= h >>> 17; h ^= h << 5; return ((h >>> 0) % 1e6) / 1e6; };
  }
  function gauss(r) { var u = r() || 1e-6, v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  function unit(v) { var n = Math.hypot.apply(null, v); return v.map(function (x) { return x / n; }); }
  function dot(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  function randomUnit(seed) { var r = rng(seed), v = []; for (var i = 0; i < 512; i++) v.push(gauss(r)); return unit(v); }
  var REF = randomUnit("asha-reference");
  var cache = {};
  function embedding(c) {
    if (cache[c.id]) return cache[c.id];
    var u = randomUnit("live-" + c.id), d = dot(u, REF);
    u = unit(u.map(function (x, i) { return x - d * REF[i]; })); // orthogonal to the reference
    var cs = 1 - c.distance, sn = Math.sqrt(1 - cs * cs);
    return (cache[c.id] = REF.map(function (x, i) { return cs * x + sn * u[i]; }));
  }

  /* ---------- drawing helpers ---------- */
  function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function barcode(canvas, v) {
    var dpr = Math.min(window.devicePixelRatio || 1, 2), w = canvas.clientWidth || 400, h = canvas.clientHeight || 64;
    canvas.width = w * dpr; canvas.height = h * dpr;
    var g = canvas.getContext("2d"); g.scale(dpr, dpr);
    g.fillStyle = css("--code-bg"); g.fillRect(0, 0, w, h);
    var pos = css("--scan"), neg = css("--warn"), bw = w / v.length;
    for (var i = 0; i < v.length; i++) {
      var a = Math.min(1, Math.abs(v[i]) * 14);
      g.globalAlpha = 0.15 + a * 0.85;
      g.fillStyle = v[i] >= 0 ? pos : neg;
      g.fillRect(i * bw, 0, Math.max(1, bw), h);
    }
    g.globalAlpha = 1;
  }
  function fmt(x, n) { return x == null ? "none" : x.toFixed(n == null ? 3 : n); }
  function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;"); }
  function json(obj) {
    return esc(JSON.stringify(obj, null, 2))
      .replace(/"([^"]+)":/g, '<span class="k">"$1"</span>:')
      .replace(/: "([^"]*)"/g, ': <span class="s">"$1"</span>')
      .replace(/: (-?[\d.]+|true|false)/g, ': <span class="n">$1</span>');
  }

  /* ---------- state ---------- */
  var STAGES = [
    { key: "read", name: "Read" },
    { key: "rotate", name: "Rotate" },
    { key: "check", name: "Check" },
    { key: "align", name: "Align" },
    { key: "embed", name: "Embed" },
    { key: "decide", name: "Decide" },
  ];
  var state = { c: F.CASES[0], stage: 0, threshold: DEFAULT_THRESHOLD, running: null };
  function failIndex(c) {
    if (!c.error) return -1;
    return c.best == null ? 1 : 2; // no face at any rotation fails while rotating; the rest fail the checks
  }

  /* ---------- cast ---------- */
  var cast = document.getElementById("cast");
  cast.innerHTML = '<h3>Enrolled reference</h3><p class="note" style="margin:0">Loaded once at startup from <code>REF_IMAGE_PATH</code></p>' +
    '<div class="ref-row"><div class="thumb">' + F.photo(F.REFERENCE) + '</div><div><b>Asha</b><span>me.jpg, embedding cached in <code>REF_EMB</code></span></div></div>' +
    '<h3>Live photo to check</h3><p class="note" style="margin:0 0 10px">What arrives in <code>live_file</code></p><div class="cases" role="group" aria-label="Live photo">' +
    F.CASES.map(function (c) {
      return '<button class="case" type="button" data-id="' + c.id + '"><span class="thumb">' + F.photo(c) + '</span><b>' + c.name + '</b><span>' + c.sub + "</span></button>";
    }).join("") + "</div>";
  cast.querySelectorAll(".case").forEach(function (b) {
    b.addEventListener("click", function () {
      state.c = F.CASES.find(function (c) { return c.id === b.dataset.id; });
      run();
    });
  });

  /* ---------- bench ---------- */
  var bench = document.getElementById("bench");
  bench.innerHTML =
    '<div class="stages" role="tablist" aria-label="Pipeline steps">' + STAGES.map(function (s, i) {
      return '<button class="stage" type="button" role="tab" data-i="' + i + '"><span class="n">0' + (i + 1) + '</span><b>' + s.name + "</b></button>";
    }).join("") + "</div>" +
    '<div class="bench-body"><div class="viewer"><div class="frame" id="v-frame"></div><div class="status-line" id="v-status"></div></div>' +
    '<div class="explain" id="v-explain" aria-live="polite"></div></div>' +
    '<div class="bench-foot"><span class="verdict idle" id="v-verdict">Waiting</span><div style="display:flex;gap:8px">' +
    '<button class="btn small" type="button" id="v-prev">Back</button><button class="btn small" type="button" id="v-next">Next step</button>' +
    '<button class="btn small primary" type="button" id="v-run">Run all steps</button></div></div>';
  var stageBtns = bench.querySelectorAll(".stage");
  var frame = bench.querySelector("#v-frame"), explain = bench.querySelector("#v-explain"), statusEl = bench.querySelector("#v-status"), verdictEl = bench.querySelector("#v-verdict");

  stageBtns.forEach(function (b) { b.addEventListener("click", function () { stop(); go(Number(b.dataset.i)); }); });
  bench.querySelector("#v-prev").addEventListener("click", function () { stop(); go(state.stage - 1); });
  bench.querySelector("#v-next").addEventListener("click", function () { stop(); go(state.stage + 1); });
  bench.querySelector("#v-run").addEventListener("click", run);

  function stop() { clearTimeout(state.running); state.running = null; }
  function run() {
    stop();
    cast.querySelectorAll(".case").forEach(function (b) { b.setAttribute("aria-pressed", String(b.dataset.id === state.c.id)); });
    var last = failIndex(state.c) >= 0 ? failIndex(state.c) : STAGES.length - 1;
    var i = 0;
    (function tick() {
      go(i);
      if (i < last) { i++; state.running = setTimeout(tick, reduce ? 0 : 900); }
    })();
  }

  function overlayBoxes(c, failed) {
    var dets = F.detection(c), col = failed ? css("--nomatch") : css("--match");
    if (c.best == null) return "";
    return '<svg class="overlay" viewBox="0 0 200 250">' + dets.map(function (d) {
      var b = d.box;
      return '<rect x="' + b.x + '" y="' + b.y + '" width="' + b.w + '" height="' + b.h + '" fill="none" stroke="' + col + '" stroke-width="1.6" rx="3"/>' +
        '<rect x="' + b.x + '" y="' + (b.y - 13) + '" width="' + Math.max(40, b.w * .6) + '" height="12" rx="2" fill="' + col + '"/>' +
        '<text x="' + (b.x + 4) + '" y="' + (b.y - 4) + '" font-family="IBM Plex Mono, monospace" font-size="8.5" font-weight="600" fill="#fff">p=' + fmt(d.prob) + "</text>" +
        d.lm.map(function (p) { return '<circle cx="' + p[0] + '" cy="' + p[1] + '" r="2.2" fill="' + css("--scan") + '" stroke="#fff" stroke-width=".8"/>'; }).join("");
    }).join("") + "</svg>";
  }
  function upright(c) { return c.sideways ? Object.assign({}, c, { sideways: false }) : c; }
  function cropSvg(c) {
    // MTCNN(image_size=160, margin=20): a square around the box, resized to 160 x 160
    var s = F.photo(upright(c)).replace('viewBox="0 0 200 250"', 'viewBox="34 49 132 132"');
    return '<div style="position:absolute;inset:0;display:grid;place-items:center;background:var(--paper-2)"><div style="width:min(82%,260px);aspect-ratio:1;position:relative;border-radius:8px;overflow:hidden;outline:2px solid var(--scan);outline-offset:3px">' + s + '</div></div>';
  }

  function go(i) {
    var c = state.c, fi = failIndex(c);
    i = Math.max(0, Math.min(fi >= 0 ? fi : STAGES.length - 1, i));
    state.stage = i;
    stageBtns.forEach(function (b, k) {
      b.classList.toggle("done", k < i);
      b.classList.toggle("fail", fi >= 0 && k === fi && i === fi);
      b.disabled = fi >= 0 && k > fi;
      if (k === i) b.setAttribute("aria-current", "step"); else b.removeAttribute("aria-current");
    });
    bench.querySelector("#v-prev").disabled = i === 0;
    bench.querySelector("#v-next").disabled = i >= (fi >= 0 ? fi : STAGES.length - 1);
    render(STAGES[i].key, c, fi === i);
  }

  function setVerdict(kind, text) { verdictEl.className = "verdict " + kind; verdictEl.textContent = text; }

  function errorBlock(c) {
    return '<span class="label">Response, HTTP 400</span><pre class="resp">' + json({ error: c.error }) + '</pre><p class="note">The text endpoint returns the same message as plain text with status 400.</p>';
  }

  function render(key, c, failed) {
    var html = "";
    var withTag = '<span class="tag">live_file</span>';
    statusEl.textContent = "";
    setVerdict("idle", "Not decided yet");
    if (key === "read") {
      frame.innerHTML = withTag + F.photo(c);
      html = "<h3>Read the upload</h3><p>The bytes go through Pillow. <code>ImageOps.exif_transpose</code> applies the camera's rotation tag, the image becomes RGB, and anything larger than <code>MAX_SIDE</code> is shrunk so detection stays fast.</p>" +
        '<dl class="readout"><dt>exif_transpose</dt><dd>applied</dd><dt>mode</dt><dd>RGB</dd><dt>MAX_SIDE</dt><dd>1600 px (longest side)</dd><dt>then</dt><dd>cv2 BGR array for the detector</dd></dl>';
      statusEl.textContent = "_read_bgr_from_bytes_with_meta()";
    } else if (key === "rotate") {
      frame.innerHTML = withTag + F.photo(c) + '<div class="scanline"></div>';
      frame.classList.remove("scanning"); void frame.offsetWidth; frame.classList.add("scanning");
      var degs = [0, 90, 270, 180];
      var bestI = c.best == null ? -1 : c.rot.indexOf(Math.max.apply(null, c.rot.filter(function (x) { return x != null; })));
      html = "<h3>Try four rotations</h3><p>Some phones save photos sideways without a rotation tag, so MTCNN runs on the image at 0, 90, 270 and 180 degrees and keeps whichever orientation gives the most confident face.</p>" +
        '<div class="rot-grid">' + degs.map(function (d, k) {
          var p = c.rot[k];
          return '<div class="rot' + (k === bestI ? " best" : "") + '"><div class="mini"><div style="width:100%;height:100%;transform:rotate(' + (-d) + 'deg) scale(.78)">' + F.photo(c) + "</div></div>" + d + "°<br>" + (p == null ? "no face" : "p=" + fmt(p)) + "</div>";
        }).join("") + "</div>" +
        (failed ? errorBlock(c) : '<dl class="readout"><dt>best rotation</dt><dd>' + degs[bestI] + "°</dd><dt>best_prob</dt><dd>" + fmt(c.best) + "</dd></dl>");
      statusEl.textContent = "DETECTOR.detect(img) for deg in [0, 90, 270, 180]";
    } else if (key === "check") {
      frame.innerHTML = withTag + F.photo(upright(c)) + overlayBoxes(upright(c), failed);
      var probs = c.group ? c.probs : [c.best];
      html = "<h3>Check the face is usable</h3><p>Two guards stop bad photos before any comparison. The best face must reach <code>MIN_FACE_PROB</code>, and with <code>REJECT_MULTI_FACE</code> on, only one face may clear that bar so the API can't compare against the wrong person.</p>" +
        '<dl class="readout"><dt>best_prob</dt><dd>' + fmt(c.best) + (c.best >= MIN_FACE_PROB ? " ≥ " : " < ") + MIN_FACE_PROB + "</dd><dt>strong_faces</dt><dd>" + c.strong + " (faces with p ≥ 0.95: " + probs.map(function (p) { return fmt(p); }).join(", ") + ")</dd></dl>" +
        (failed ? errorBlock(c) : '<p class="note">Passed. The five dots are MTCNN’s landmarks: eyes, nose and mouth corners.</p>');
      statusEl.textContent = c.sideways ? "using the 270° rotation" : "MIN_FACE_PROB = 0.95, REJECT_MULTI_FACE = 1";
    } else if (key === "align") {
      frame.innerHTML = '<span class="tag">160 × 160</span>' + cropSvg(c);
      html = "<h3>Crop and align</h3><p>MTCNN cuts a square around the face with a 20 pixel margin, resizes it to 160 by 160 and standardises the pixel values. Every face reaches the network at the same size and position.</p>" +
        '<dl class="readout"><dt>MTCNN</dt><dd>image_size=160, margin=20, post_process=True</dd><dt>select_largest</dt><dd>True</dd><dt>tensor</dt><dd>[1, 3, 160, 160]</dd></dl>';
      statusEl.textContent = "aligned = DETECTOR(best_img)";
    } else if (key === "embed") {
      var v = embedding(c);
      frame.innerHTML = '<span class="tag">160 × 160</span>' + cropSvg(c);
      html = "<h3>Turn the face into 512 numbers</h3><p><code>InceptionResnetV1</code>, pretrained on VGGFace2, maps the crop to 512 numbers and the API scales them to length 1. Faces of the same person land close together in this space. Each bar below is one number: blue is positive, amber negative.</p>" +
        '<span class="label">live embedding</span><canvas class="barcode" id="bc-live"></canvas><span class="label">reference embedding (Asha)</span><canvas class="barcode" id="bc-ref"></canvas>' +
        '<dl class="readout"><dt>first values</dt><dd>[' + v.slice(0, 5).map(function (x) { return x.toFixed(4); }).join(", ") + ", …]</dd><dt>norm</dt><dd>" + Math.hypot.apply(null, v).toFixed(4) + "</dd></dl>";
      statusEl.textContent = "EMB_MODEL(aligned); emb / norm(emb)";
    } else if (key === "decide") {
      frame.innerHTML = '<span class="tag">160 × 160</span>' + cropSvg(c);
      html = "<h3>Compare and decide</h3><p>Cosine distance is one minus the dot product of the two embeddings. At or under the threshold means the same person. Drag the threshold to see where this pair flips.</p>" +
        '<svg class="gauge" id="gauge" viewBox="0 0 400 74" aria-label="Distance gauge"></svg>' +
        '<div class="slider-row"><label for="thr-in">Threshold</label><input id="thr-in" type="range" min="0.05" max="0.9" step="0.01" value="' + state.threshold + '"><output id="thr-out">' + state.threshold.toFixed(2) + "</output></div>" +
        '<div class="presets"><button class="btn small" type="button" data-t="0.30">0.30 FaceNet apps</button><button class="btn small" type="button" data-t="0.35">0.35 main.py</button></div>' +
        '<span class="label" id="resp-label"></span><pre class="resp" id="resp"></pre>';
      statusEl.textContent = "dist = 1 - dot(ref_emb, live_emb)";
    }
    explain.innerHTML = html;
    if (failed) setVerdict("error", "HTTP 400");
    if (key === "embed") {
      requestAnimationFrame(function () { barcode(explain.querySelector("#bc-live"), embedding(c)); barcode(explain.querySelector("#bc-ref"), REF); });
    }
    if (key === "decide") {
      var input = explain.querySelector("#thr-in");
      input.addEventListener("input", function () { state.threshold = Number(input.value); decide(c); });
      explain.querySelectorAll("[data-t]").forEach(function (b) { b.addEventListener("click", function () { state.threshold = Number(b.dataset.t); input.value = state.threshold; decide(c); }); });
      decide(c);
    }
  }

  function decide(c) {
    var t = state.threshold, d = c.distance, ok = d <= t;
    explain.querySelector("#thr-out").textContent = t.toFixed(2);
    var g = explain.querySelector("#gauge"), X = function (v) { return 14 + v * 372; };
    g.innerHTML =
      '<rect x="14" y="24" width="' + (X(t) - 14) + '" height="14" rx="4" fill="' + css("--match") + '" opacity=".25"/>' +
      '<rect x="' + X(t) + '" y="24" width="' + (386 - X(t)) + '" height="14" rx="4" fill="' + css("--nomatch") + '" opacity=".2"/>' +
      [0, 0.25, 0.5, 0.75, 1].map(function (v) { return '<text x="' + X(v) + '" y="58" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="10" fill="' + css("--ink-3") + '">' + v.toFixed(2) + "</text>"; }).join("") +
      '<line x1="' + X(t) + '" x2="' + X(t) + '" y1="16" y2="46" stroke="' + css("--ink") + '" stroke-width="2" stroke-dasharray="3 2"/>' +
      '<text x="' + X(t) + '" y="12" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="10" font-weight="600" fill="' + css("--ink") + '">threshold</text>' +
      '<circle cx="' + X(d) + '" cy="31" r="8" fill="' + (ok ? css("--match") : css("--nomatch")) + '" stroke="' + css("--paper") + '" stroke-width="2.5"/>' +
      '<text x="' + Math.min(370, Math.max(30, X(d))) + '" y="72" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="10.5" font-weight="700" fill="' + (ok ? css("--match") : css("--nomatch")) + '">d=' + d.toFixed(3) + "</text>";
    setVerdict(ok ? "match" : "nomatch", ok ? "Match" : "No Match");
    var custom = Math.abs(t - DEFAULT_THRESHOLD) > 1e-9;
    explain.querySelector("#resp-label").textContent = "POST /verify-live-file, HTTP 200";
    explain.querySelector("#resp").innerHTML = json({
      who: "me", verified: ok, distance: d, similarity: +(1 - d).toFixed(3), threshold: t, threshold_source: custom ? "custom" : "default",
      metric: "cosine(1 - dot)", backbone: "InceptionResnetV1(vggface2)", detector: "MTCNN(160x160)", min_face_prob: MIN_FACE_PROB, ref_path: "me.jpg",
    });
  }

  document.addEventListener("themechange", function () { go(state.stage); });
  cast.querySelector(".case").setAttribute("aria-pressed", "true");
  go(0);

  /* ---------- hero rig: loops through a few cases ---------- */
  var rig = document.getElementById("rig");
  var loop = [F.CASES[0], F.CASES[4], F.CASES[1], F.CASES[3]];
  rig.innerHTML = '<div class="rig-faces"><div class="frame"><span class="tag">reference</span>' + F.photo(F.REFERENCE) + '</div>' +
    '<div class="rig-mid"><span class="label">distance</span><div class="dist" id="rig-d">0.000</div><span class="verdict idle" id="rig-v">Scanning</span></div>' +
    '<div class="frame" id="rig-live"></div></div>' +
    '<div class="rig-foot"><span class="note" id="rig-cap"></span><span class="note">threshold 0.30</span></div>';
  var k = 0, raf;
  function rigStep() {
    var c = loop[k % loop.length];
    var live = rig.querySelector("#rig-live");
    live.innerHTML = '<span class="tag">live_file</span>' + F.photo(c) + '<div class="scanline"></div>';
    live.classList.remove("scanning"); void live.offsetWidth; live.classList.add("scanning");
    rig.querySelector("#rig-cap").textContent = c.name + ": " + c.sub.toLowerCase();
    var v = rig.querySelector("#rig-v"), dEl = rig.querySelector("#rig-d");
    v.className = "verdict idle"; v.textContent = "Scanning";
    var t0 = performance.now(), dur = reduce ? 0 : 1400;
    cancelAnimationFrame(raf);
    (function count(now) {
      var p = dur ? Math.min(1, (now - t0) / dur) : 1;
      dEl.textContent = (c.distance * (1 - Math.pow(1 - p, 3))).toFixed(3);
      if (p < 1) raf = requestAnimationFrame(count);
      else { var ok = c.distance <= DEFAULT_THRESHOLD; v.className = "verdict " + (ok ? "match" : "nomatch"); v.textContent = ok ? "Match" : "No Match"; }
    })(t0);
    k++;
  }
  rigStep();
  if (!reduce) setInterval(function () { if (!document.hidden) rigStep(); }, 4200);
})();
