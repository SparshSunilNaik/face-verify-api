/* Illustrated faces (no real photos) and the test cases used across the page. */
(function () {
  // A face is drawn in a 200 x 250 frame. Feature positions are fixed so overlays line up.
  var PEOPLE = {
    asha:  { skin: "#c68a62", shade: "#a8704c", hair: "#1e1714", style: "long", brow: "#2a1d17", lip: "#9a4e45", shirt: "#3f6fd8", iris: "#3a2418" },
    meera: { skin: "#c48862", shade: "#a66e4b", hair: "#2a1c15", style: "bob",  brow: "#2a1d17", lip: "#a0524a", shirt: "#c2477a", iris: "#3a2418" },
    kiran: { skin: "#8a5a3c", shade: "#6f452c", hair: "#121012", style: "curly", brow: "#151012", lip: "#6b3a2e", shirt: "#2e9b6a", iris: "#1b120d", beard: true },
    dev:   { skin: "#e2b391", shade: "#c99474", hair: "#6b4a2f", style: "short", brow: "#5a3d27", lip: "#b3675b", shirt: "#e08a2e", iris: "#4d6b3a" },
  };
  var LANDMARKS = [[82, 108], [118, 108], [100, 127], [88, 146], [112, 146]]; // eyes, nose, mouth corners
  var BOX = { x: 50, y: 52, w: 100, h: 126 };

  function hairBack(p) {
    if (p.style === "long") return '<path d="M48 104c-6 40-4 86 6 116h92c10-30 12-76 6-116-8-46-96-46-104 0Z" fill="' + p.hair + '"/>';
    if (p.style === "bob") return '<path d="M46 100c-4 26-2 52 8 70h92c10-18 12-44 8-70-8-44-100-44-108 0Z" fill="' + p.hair + '"/>';
    return "";
  }
  function hairFront(p) {
    switch (p.style) {
      case "long": return '<path d="M52 104c0-40 22-58 48-58s48 18 48 58c-10-18-24-28-40-30-10 10-30 18-56 30Z" fill="' + p.hair + '"/>';
      case "bob": return '<path d="M50 102c0-38 22-56 50-56s50 18 50 56c-18-8-28-22-34-34-16 18-40 28-66 34Z" fill="' + p.hair + '"/>';
      case "curly":
        var s = "";
        for (var i = 0; i < 11; i++) { var a = Math.PI + i / 10 * Math.PI; s += '<circle cx="' + (100 + Math.cos(a) * 46).toFixed(1) + '" cy="' + (86 + Math.sin(a) * 34).toFixed(1) + '" r="13" fill="' + p.hair + '"/>'; }
        return s + '<path d="M58 92c10-22 74-22 84 0-20-8-64-8-84 0Z" fill="' + p.hair + '"/>';
      case "short": return '<path d="M52 98c0-34 20-50 48-50s48 16 48 50c-6-10-14-18-24-20-14 8-46 10-72 20Z" fill="' + p.hair + '"/>';
      default: return "";
    }
  }

  // one head + shoulders, drawn at the origin frame
  function person(p, o) {
    o = o || {};
    var smile = o.smile ? 'M88 145c6 7 18 7 24 0' : 'M89 146c6 3 16 3 22 0';
    var s = "";
    s += '<path d="M28 250c4-44 34-62 72-62s68 18 72 62Z" fill="' + p.shirt + '"/>';
    s += '<path d="M86 160h28v34c-8 6-20 6-28 0Z" fill="' + p.shade + '"/>';
    s += hairBack(p);
    s += '<ellipse cx="52" cy="114" rx="7" ry="11" fill="' + p.shade + '"/><ellipse cx="148" cy="114" rx="7" ry="11" fill="' + p.shade + '"/>';
    s += '<ellipse cx="100" cy="112" rx="48" ry="58" fill="' + p.skin + '"/>';
    if (o.back) {
      // back of the head: no features at all
      return s + '<ellipse cx="100" cy="104" rx="50" ry="60" fill="' + p.hair + '"/>';
    }
    s += '<path d="M60 128c4 26 20 40 40 40s36-14 40-40" fill="none" stroke="' + p.shade + '" stroke-width="2" opacity=".35"/>';
    s += '<path d="M72 96c6-5 14-6 20-3M108 93c6-3 14-2 20 3" stroke="' + p.brow + '" stroke-width="4" stroke-linecap="round" fill="none"/>';
    s += '<ellipse cx="82" cy="108" rx="8" ry="5.5" fill="#fff"/><ellipse cx="118" cy="108" rx="8" ry="5.5" fill="#fff"/>';
    s += '<circle cx="' + (82 + (o.look || 0)) + '" cy="108" r="4" fill="' + p.iris + '"/><circle cx="' + (118 + (o.look || 0)) + '" cy="108" r="4" fill="' + p.iris + '"/>';
    s += '<path d="M100 112c-2 8-5 13-7 16 3 2 8 2 11 0" fill="none" stroke="' + p.shade + '" stroke-width="2.4" stroke-linecap="round"/>';
    s += '<path d="' + smile + '" fill="none" stroke="' + p.lip + '" stroke-width="3.4" stroke-linecap="round"/>';
    if (p.beard) s += '<path d="M56 124c2 34 22 50 44 50s42-16 44-50c-6 14-14 18-22 16-6 10-38 10-44 0-8 2-16-2-22-16Z" fill="' + p.hair + '" opacity=".9"/><path d="M86 140c8-4 20-4 28 0" stroke="' + p.hair + '" stroke-width="5" stroke-linecap="round"/>';
    s += hairFront(p);
    if (o.glasses) {
      s += '<g fill="none" stroke="#1a1d24" stroke-width="3"><rect x="67" y="98" width="30" height="21" rx="7"/><rect x="103" y="98" width="30" height="21" rx="7"/><path d="M97 106h6M67 104l-14-3M133 104l14-3"/></g>';
      s += '<rect x="69" y="100" width="26" height="17" rx="6" fill="#bfe6ff" opacity=".18"/><rect x="105" y="100" width="26" height="17" rx="6" fill="#bfe6ff" opacity=".18"/>';
    }
    return s;
  }

  function background(bg) {
    return '<rect width="200" height="250" fill="' + bg[0] + '"/><rect y="170" width="200" height="80" fill="' + bg[1] + '" opacity=".6"/>';
  }

  // full "photo" for a case
  function photo(c, extraStyle) {
    var inner = background(c.bg);
    if (c.group) {
      inner += '<g transform="translate(6 46) scale(.62)">' + person(PEOPLE.dev, { smile: true }) + "</g>";
      inner += '<g transform="translate(70 40) scale(.66)">' + person(PEOPLE[c.who], { smile: true }) + "</g>";
    } else {
      var body = person(PEOPLE[c.who], c.opts);
      if (c.sideways) body = '<g transform="translate(100 125) rotate(90) scale(.8) translate(-100 -125)">' + body + "</g>";
      inner += body;
    }
    if (c.dim) inner += '<rect width="200" height="250" fill="#05070c" opacity=".62"/>';
    return '<svg class="face" viewBox="0 0 200 250" role="img" aria-label="' + c.alt + '"' + (extraStyle ? ' style="' + extraStyle + '"' : "") + ">" + inner + "</svg>";
  }

  // where the detector would put its box and the five landmarks, in frame coordinates
  function detection(c) {
    if (c.group) {
      return [
        { box: tx(BOX, 6, 46, .62), lm: LANDMARKS.map(function (p) { return [6 + p[0] * .62, 46 + p[1] * .62]; }), prob: c.probs[1] },
        { box: tx(BOX, 70, 40, .66), lm: LANDMARKS.map(function (p) { return [70 + p[0] * .66, 40 + p[1] * .66]; }), prob: c.probs[0] },
      ];
    }
    return [{ box: BOX, lm: LANDMARKS, prob: c.best }];
  }
  function tx(b, dx, dy, k) { return { x: dx + b.x * k, y: dy + b.y * k, w: b.w * k, h: b.h * k }; }

  var REFERENCE = { id: "ref", who: "asha", bg: ["#dfe6ee", "#c9d3df"], opts: { smile: false }, alt: "Illustrated reference portrait of Asha", name: "Asha", file: "me.jpg" };

  // distance = cosine distance (1 - dot) between the live and reference embeddings; illustrative values
  var CASES = [
    { id: "same", who: "asha", bg: ["#e9e2d6", "#d6cbb9"], opts: { smile: true, look: 1 }, alt: "Asha smiling, new photo", name: "Same person", sub: "New day, new smile", best: 0.999, rot: [0.999, 0.41, 0.38, null], strong: 1, distance: 0.142 },
    { id: "glasses", who: "asha", bg: ["#e4ecf3", "#cfdae6"], opts: { glasses: true }, alt: "Asha wearing glasses", name: "With glasses", sub: "Same person, frames on", best: 0.998, rot: [0.998, 0.33, 0.29, null], strong: 1, distance: 0.236 },
    { id: "sideways", who: "asha", bg: ["#e8e4ef", "#d4cee0"], sideways: true, opts: { smile: true }, alt: "Asha, photo stored sideways", name: "Sideways photo", sub: "Phone saved it rotated", best: 0.997, rot: [0.64, 0.12, 0.997, null], strong: 1, distance: 0.171, bestDeg: 270 },
    { id: "lookalike", who: "meera", bg: ["#efe6e3", "#e0d1cc"], opts: { smile: true }, alt: "Meera, Asha's sister", name: "Look-alike sister", sub: "Close, but someone else", best: 0.998, rot: [0.998, 0.36, 0.31, null], strong: 1, distance: 0.327 },
    { id: "stranger", who: "kiran", bg: ["#e2ece6", "#ccdcd2"], opts: { smile: true }, alt: "Kiran, a different person", name: "Different person", sub: "Clearly not Asha", best: 0.999, rot: [0.999, 0.44, 0.35, null], strong: 1, distance: 0.734 },
    { id: "group", who: "asha", bg: ["#f0ebe0", "#ddd3c0"], group: true, alt: "Asha with a friend in the frame", name: "Friend in frame", sub: "Two faces in one photo", best: 0.998, probs: [0.998, 0.986], rot: [0.998, 0.29, 0.31, null], strong: 2,
      error: "Multiple faces detected (2); submit a single-person image.", failStage: "detect" },
    { id: "dim", who: "asha", bg: ["#d9dee6", "#c3cad5"], dim: true, opts: {}, alt: "Asha in a very dark room", name: "Too dark", sub: "Taken with the lights off", best: 0.912, rot: [0.912, 0.21, 0.18, null], strong: 0,
      error: "Low face confidence: 0.912 (< 0.95)", failStage: "detect" },
    { id: "back", who: "asha", bg: ["#e6e6e6", "#d0d0d0"], opts: { back: true }, alt: "The back of Asha's head", name: "Wrong way round", sub: "Back of the head", best: null, rot: [null, null, null, null], strong: 0,
      error: "No face detected", failStage: "detect" },
  ];

  window.Faces = { PEOPLE: PEOPLE, REFERENCE: REFERENCE, CASES: CASES, photo: photo, detection: detection, LANDMARKS: LANDMARKS, BOX: BOX };
})();
