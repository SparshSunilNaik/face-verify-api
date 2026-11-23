/* Versions, API explorer, run commands and issues. Everything here is taken from the Python files. */
(function () {
  function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }

  var FACENET = { verified: true, distance: 0.142, threshold: 0.3, metric: "cosine(1 - dot)", backbone: "InceptionResnetV1(vggface2)", detector: "MTCNN(160x160)", min_face_prob: 0.95, ref_path: "me.jpg" };
  var FACENET3 = { verified: true, distance: 0.142, similarity: 0.858, threshold: 0.3, threshold_source: "default", metric: "cosine(1 - dot)", backbone: "InceptionResnetV1(vggface2)", detector: "MTCNN(160x160)", min_face_prob: 0.95, ref_path: "me.jpg" };
  var FACENET4 = Object.assign({ who: "me" }, FACENET3);
  var LIVE = { name: "live_file", type: "file", note: "photo to check" };
  var THR = { name: "custom_threshold", type: "float, optional", note: "overrides the default" };
  var WHO = { name: "who", type: "string, default me", note: "me, amya or rishi" };

  var VERSIONS = [
    {
      key: "main", file: "main.py", ver: "1.0.0", port: 8000, title: "Face Verify API (InsightFace/ONNX)",
      stack: "InsightFace buffalo_l: SCRFD detector + ArcFace", threshold: "0.35",
      changes: ["Compares any two photos you send", "Uploads, base64 or server paths", "Plain-text answer endpoint", "No authentication"],
      auth: "none",
      routes: [
        { m: "GET", p: "/health", open: true, desc: "Liveness check.", fields: [], res: "ok", text: true },
        { m: "POST", p: "/verify-file", open: true, desc: "Two uploaded photos, JSON verdict.", fields: [{ name: "ref_file", type: "file", note: "reference photo" }, { name: "live_file", type: "file", note: "photo to check" }, THR],
          res: { verified: true, distance: 0.21, threshold: 0.35, metric: "cosine(1 - dot)", model_pack: "buffalo_l" } },
        { m: "POST", p: "/verify-b64", open: true, json: true, desc: "Two base64 images (data URLs work) in a JSON body.", fields: [{ name: "ref_image_b64", type: "string", note: "base64 or data URL" }, { name: "live_image_b64", type: "string", note: "base64 or data URL" }, THR],
          res: { verified: true, distance: 0.21, threshold: 0.35, metric: "cosine(1 - dot)", model_pack: "buffalo_l" } },
        { m: "POST", p: "/verify-path", open: true, json: true, desc: "Two file paths on the server.", fields: [{ name: "ref_image_path", type: "string", note: "path on the server" }, { name: "live_image_path", type: "string", note: "path on the server" }, THR],
          res: { verified: false, distance: 0.71, threshold: 0.35, metric: "cosine(1 - dot)", model_pack: "buffalo_l" } },
        { m: "POST", p: "/verify-text", open: true, desc: "Two uploads, answers Match or No Match as plain text.", fields: [{ name: "ref_file", type: "file", note: "reference photo" }, { name: "live_file", type: "file", note: "photo to check" }, THR], res: "Match", text: true },
      ],
    },
    {
      key: "app", file: "app.py", ver: "2.4.0", port: 8000, title: "Face Verify API (FaceNet/PyTorch)",
      stack: "facenet-pytorch: MTCNN + InceptionResnetV1 (VGGFace2)", threshold: "0.30",
      changes: ["One reference photo held on the server (me.jpg)", "EXIF fix, 1600 px cap and rotation fallback", "Rejects weak or multiple faces", "Webcam /ui and phone /mobile pages"],
      auth: "x-api-key header, default key written in the code",
      routes: [
        { m: "GET", p: "/health", open: true, desc: "Liveness plus settings.", fields: [], res: "ok (ref_loaded=True, thr=0.3, min_prob=0.95, max_side=1600)", text: true },
        { m: "POST", p: "/verify-live-text", open: true, desc: "Live photo against me.jpg, plain-text verdict. Left open so the phone page works without a key.", fields: [LIVE, THR], res: "Match", text: true },
        { m: "POST", p: "/verify-live-file", desc: "Live photo against me.jpg, JSON verdict.", fields: [LIVE, THR], res: FACENET },
        { m: "POST", p: "/verify-live-file-debug", desc: "Raw numbers for tuning the threshold.", fields: [LIVE, THR], res: { distance: 0.142, threshold: 0.3, verified: true, best_prob: 0.999, strong_faces: 1 } },
        { m: "POST", p: "/reload-ref-file", desc: "Replace the reference with an uploaded photo.", fields: [{ name: "ref_file", type: "file", note: "new reference" }], res: { ok: true, ref_path: "uploaded:me.jpg" } },
        { m: "POST", p: "/reload-ref-path", desc: "Replace the reference from a server path.", fields: [{ name: "ref_image_path", type: "string, form", note: "path on the server" }], res: { ok: true, ref_path: "me.jpg" } },
        { m: "POST", p: "/verify-file", desc: "Disabled so callers can't bring their own reference.", fields: [{ name: "ref_file", type: "file", note: "" }, LIVE], res: { detail: "Pairwise verification disabled. Use /verify-live-file." }, status: 403 },
        { m: "GET", p: "/mobile", open: true, desc: "Phone page: take a selfie, verify, auto-close on Match (?autoclose=10, ?close_url=...), lock with Retry on No Match.", fields: [], res: "<html> phone camera page", text: true },
      ],
    },
    {
      key: "app1", file: "app1.py", ver: "3.2.0", port: 8001, title: "Face Verify API (Backend)",
      stack: "Same FaceNet pipeline, settings from environment variables", threshold: "0.30",
      changes: ["API key read from API_KEY; nothing protected works if unset", "Every route except /health, /ui, /mobile needs the key", "Base64 live endpoint", "Timing logs: decode, detect, embed, total", "Adds similarity and threshold_source"],
      auth: "x-api-key header from the API_KEY variable",
      routes: [
        { m: "GET", p: "/health", open: true, desc: "Liveness plus settings.", fields: [], res: "ok (ref_loaded=True, thr=0.3, min_prob=0.95, max_side=1600)", text: true },
        { m: "POST", p: "/verify-live-text", desc: "Live photo against the reference, plain-text verdict.", fields: [LIVE, THR], res: "Match", text: true },
        { m: "POST", p: "/verify-live-file", desc: "JSON verdict with similarity.", fields: [LIVE, THR], res: FACENET3 },
        { m: "POST", p: "/verify-live-b64", json: true, desc: "Base64 live photo in a JSON body.", fields: [{ name: "live_image_b64", type: "string", note: "base64 or data URL" }, THR], res: FACENET3 },
        { m: "POST", p: "/reload-ref-file", desc: "Replace the reference with an upload.", fields: [{ name: "ref_file", type: "file", note: "new reference" }], res: { ok: true, ref_path: "uploaded:me.jpg" } },
        { m: "POST", p: "/reload-ref-path", desc: "Replace the reference from a server path.", fields: [{ name: "ref_image_path", type: "string, form", note: "path on the server" }], res: { ok: true, ref_path: "me.jpg" } },
        { m: "GET", p: "/ui", open: true, desc: "Desktop webcam page with an API key box (or ?k=...).", fields: [], res: "<html> webcam test page", text: true },
      ],
    },
    {
      key: "app2", file: "app2.py", ver: "4.0.1", port: 8002, title: "Face Verify API (Multi-Identity)",
      stack: "Same FaceNet pipeline, one reference per person", threshold: "0.30",
      changes: ["Three enrolled people: me, amya, rishi", "who field picks the reference", "GET /who lists what's loaded", "Verdict also sent as X-Who, X-Verified, X-Distance headers"],
      auth: "x-api-key header from the API_KEY variable",
      routes: [
        { m: "GET", p: "/health", open: true, desc: "Liveness plus loaded references.", fields: [], res: "ok (thr=0.3, min_prob=0.95, loaded={'me': 'me.jpg', 'amya': 'amya.jpg', 'rishi': 'rishi.jpg'})", text: true },
        { m: "GET", p: "/who", open: true, desc: "Which identities are loaded.", fields: [], res: { me: { path: "me.jpg", loaded: true }, amya: { path: "amya.jpg", loaded: true }, rishi: { path: "rishi.jpg", loaded: true } } },
        { m: "POST", p: "/verify-live-text", desc: "Plain-text verdict, with the numbers in X- headers.", fields: [LIVE, WHO, THR], res: "Match", text: true, headers: "X-Who: me\nX-Verified: true\nX-Distance: 0.142000\nX-Threshold: 0.300000\nX-Similarity: 0.858000" },
        { m: "POST", p: "/verify-live-file", desc: "JSON verdict for the chosen person.", fields: [LIVE, WHO, THR], res: FACENET4 },
        { m: "POST", p: "/verify-live-b64", json: true, desc: "Base64 photo plus who in a JSON body.", fields: [{ name: "live_image_b64", type: "string", note: "base64 or data URL" }, WHO, THR], res: FACENET4 },
        { m: "POST", p: "/reload-ref-file", desc: "Enrol or replace one person from an upload.", fields: [{ name: "ref_file", type: "file", note: "their photo" }, { name: "who", type: "string, required", note: "name to store under" }], res: { ok: true, who: "amya", ref_path: "uploaded:amya.jpg" } },
        { m: "POST", p: "/reload-ref-path", desc: "Enrol or replace one person from a server path.", fields: [{ name: "ref_image_path", type: "string, form", note: "path on the server" }, { name: "who", type: "string, required", note: "name to store under" }], res: { ok: true, who: "rishi", ref_path: "rishi.jpg" } },
      ],
    },
  ];

  /* ---------- timeline ---------- */
  document.getElementById("timeline").innerHTML = VERSIONS.map(function (v, i) {
    return '<article class="card ver"><div class="ver-top"><span class="num">v' + v.ver + '</span><span class="pill">port ' + v.port + "</span></div>" +
      "<h3>" + v.file + '</h3><div class="stack">' + esc(v.stack) + "</div><ul>" + v.changes.map(function (c) { return "<li>" + esc(c) + "</li>"; }).join("") + "</ul>" +
      '<div class="foot"><span class="pill">threshold ' + v.threshold + '</span><a class="pill" href="#api" data-ver="' + v.key + '" style="text-decoration:none">' + v.routes.length + " endpoints</a></div></article>";
  }).join("");
  document.getElementById("helpers").innerHTML = [
    ["verify.py", "Command-line check with DeepFace (Facenet512 + MTCNN): opens the webcam, press c to capture, prints MATCH or NO MATCH with the distance. This is what requirements.txt installs for."],
    ["adas.py", "On-device demo with OpenCV's Haar cascade: draws a box around every face in the webcam feed. No recognition, just detection."],
    ["hhh.txt", "A copy of adas.py saved as text."],
  ].map(function (h) { return '<div class="card helper"><h4>' + h[0] + "</h4><p>" + esc(h[1]) + "</p></div>"; }).join("");

  /* ---------- API explorer ---------- */
  var side = document.getElementById("api-side"), main = document.getElementById("api-main");
  var cur = { v: VERSIONS[3], r: 3 };
  function sideHtml() {
    return '<div class="ver-tabs" role="group" aria-label="Version">' + VERSIONS.map(function (v) {
      return '<button type="button" data-v="' + v.key + '" aria-pressed="' + (v === cur.v) + '">' + v.file.replace(".py", "") + "</button>";
    }).join("") + '</div><div class="routes" role="group" aria-label="Endpoints">' + cur.v.routes.map(function (r, i) {
      return '<button class="route" type="button" data-r="' + i + '" aria-pressed="' + (i === cur.r) + '"><span class="verb ' + r.m.toLowerCase() + '">' + r.m + "</span><span>" + r.p + '</span><span class="lock' + (r.open || cur.v.key === "main" ? "" : " key") + '">' + (r.open || cur.v.key === "main" ? "open" : "key") + "</span></button>";
    }).join("") + "</div>";
  }
  function curl(v, r) {
    var url = "http://localhost:" + v.port + r.p;
    var parts = ["curl"];
    if (r.m === "POST") parts.push("-X POST");
    var needKey = !r.open && v.key !== "main";
    if (needKey) parts.push('-H "x-api-key: $API_KEY"');
    if (r.json) {
      var body = {};
      r.fields.forEach(function (f) {
        if (f.name === "custom_threshold") return;
        body[f.name] = f.name.indexOf("b64") >= 0 ? "data:image/jpeg;base64,/9j/4AAQ..." : f.name.indexOf("path") >= 0 ? (f.name.indexOf("ref") >= 0 ? "me.jpg" : "live.jpg") : "me";
      });
      parts.push('-H "Content-Type: application/json"');
      parts.push("-d '" + JSON.stringify(body) + "'");
    } else {
      r.fields.forEach(function (f) {
        if (f.name === "custom_threshold") return;
        if (f.type === "file") parts.push("-F " + f.name + "=@" + (f.name.indexOf("ref") === 0 ? "me.jpg" : "live.jpg"));
        else if (f.name === "who") parts.push("-F who=me");
        else parts.push("-F " + f.name + "=me.jpg");
      });
    }
    if (r.headers) parts.splice(1, 0, "-i");
    parts.push('"' + url + '"');
    return parts.join(" \\\n  ");
  }
  function mainHtml() {
    var v = cur.v, r = v.routes[cur.r], needKey = !r.open && v.key !== "main";
    var res = r.text ? esc(r.res) : esc(JSON.stringify(r.res, null, 2));
    return '<div><span class="label">' + v.file + " v" + v.ver + "</span><h3>" + r.m + " " + r.p + "</h3></div><p>" + esc(r.desc) + "</p>" +
      (r.fields.length ? '<table class="fields"><thead><tr><th>' + (r.json ? "JSON body" : "Form field") + "</th><th>Type</th><th>Notes</th></tr></thead><tbody>" +
        r.fields.map(function (f) { return "<tr><td>" + f.name + "</td><td>" + esc(f.type) + "</td><td>" + esc(f.note) + "</td></tr>"; }).join("") + "</tbody></table>" : "") +
      '<div class="callout' + (needKey ? " warn" : "") + '">' + (v.key === "main" ? "No authentication in this version." : needKey ? "Needs the <code>x-api-key</code> header. Without it the middleware answers 403 <code>{\"error\": \"Unauthorized\"}</code>." : "Open: listed in <code>OPEN_PATHS</code>, so no key is needed.") + "</div>" +
      '<div class="two-col"><div><span class="label">Request</span><div class="cmd"><pre id="curl-cmd">' + esc(curl(v, r)) + '</pre><button class="copy" type="button" data-copy="curl-cmd">Copy</button></div></div>' +
      '<div><span class="label">Response, HTTP ' + (r.status || 200) + '</span><div class="cmd"><pre>' + (r.headers ? esc(r.headers) + "\n\n" : "") + res + "</pre></div></div></div>" +
      (r.m === "POST" && r.p.indexOf("verify") >= 0 && !r.status ? '<p class="note">Errors (no face, low confidence, several faces, missing reference) come back as HTTP 400 with <code>{"error": "..."}</code>, or the message as plain text on the text endpoint.</p>' : "");
  }
  function draw() {
    side.innerHTML = sideHtml(); main.innerHTML = mainHtml();
    side.querySelectorAll("[data-v]").forEach(function (b) { b.addEventListener("click", function () { cur.v = VERSIONS.find(function (v) { return v.key === b.dataset.v; }); cur.r = Math.min(1, cur.v.routes.length - 1); draw(); }); });
    side.querySelectorAll("[data-r]").forEach(function (b) { b.addEventListener("click", function () { cur.r = Number(b.dataset.r); draw(); }); });
  }
  document.querySelectorAll("[data-ver]").forEach(function (a) {
    a.addEventListener("click", function () { cur.v = VERSIONS.find(function (v) { return v.key === a.dataset.ver; }); cur.r = Math.min(1, cur.v.routes.length - 1); draw(); });
  });
  draw();

  /* ---------- run it ---------- */
  var FACENET_PIP = "pip install fastapi \"uvicorn[standard]\" python-multipart \\\n  torch facenet-pytorch opencv-python pillow numpy";
  var RUNS = [
    ["app2.py", "Multi-person API on port 8002. Put me.jpg, amya.jpg and rishi.jpg next to it, or point REF_ME, REF_AMYA and REF_RISHI at your own photos.",
      "git clone https://github.com/SparshSunilNaik/face-verify-api.git\ncd face-verify-api\npython -m venv .venv && source .venv/bin/activate\n" + FACENET_PIP + "\nexport API_KEY=change-me\npython app2.py"],
    ["app1.py", "Single-reference API on port 8001. REF_IMAGE_PATH, DEFAULT_THRESHOLD, MIN_FACE_PROB, REJECT_MULTI_FACE and MAX_SIDE are all environment variables.",
      FACENET_PIP + "\nexport API_KEY=change-me\nexport REF_IMAGE_PATH=me.jpg\npython app1.py"],
    ["main.py", "Two-photo API with InsightFace on port 8000. The buffalo_l model pack downloads on first start.",
      "pip install fastapi \"uvicorn[standard]\" python-multipart \\\n  insightface onnxruntime opencv-python pillow numpy\nuvicorn main:app --port 8000"],
    ["verify.py", "Webcam check from the terminal, no server. This is the one requirements.txt is for.",
      "pip install -r requirements.txt\npython verify.py --ref me.jpg --custom_threshold 0.4"],
  ];
  document.getElementById("runs").innerHTML = RUNS.map(function (r, i) {
    return '<div class="card run"><h3><code>' + r[0] + "</code></h3><p>" + esc(r[1]) + '</p><div class="cmd"><pre id="run-' + i + '">' + esc(r[2]) + '</pre><button class="copy" type="button" data-copy="run-' + i + '">Copy</button></div></div>';
  }).join("");

  /* ---------- issues ---------- */
  var ISSUES = [
    ["A key is written into app.py", "app.py falls back to a real-looking API key in the source when API_KEY is unset. Rotate it and read it only from the environment, as app1.py and app2.py do."],
    ["requirements.txt installs the wrong stack", "It lists TensorFlow and DeepFace, which only verify.py uses. The servers need torch and facenet-pytorch, and main.py needs insightface and onnxruntime. None of those are listed."],
    ["The phone endpoint skips the key", "app.py leaves /verify-live-text open so the phone page works without a key, which means anyone who finds the server can query it."],
    ["CORS is wide open", "Every version allows all origins with credentials. Fine for a demo, but lock allow_origins to your front end before going live."],
    ["Face photos sit in the repo", "me.jpg, amya.jpg, rishi.jpg and photo.png are real portraits in a public repository. Keep reference photos out of git and load them from private storage."],
    ["No README", "Nothing says which of the four files to run or which libraries each needs. This page can serve as the start of one."],
  ];
  document.getElementById("issue-list").innerHTML = ISSUES.map(function (x) { return '<div class="card issue"><h3>' + esc(x[0]) + "</h3><p>" + esc(x[1]) + "</p></div>"; }).join("");
})();
