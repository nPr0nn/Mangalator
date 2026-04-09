/* ═══════════════════════════════════════════════════════════════════════════
   Manga Translator — frontend  (4-step pipeline)

   Bubble data shape (mirrors API):
     id                     unique int
     bx, by, bw, bh         full balloon box  (inpaint mask, drawn red/dashed)
     tx, ty, tw, th         tight text box    (OCR crop + render target, blue/solid)
     has_text_box           bool
     japanese               string

   Steps:
     1 → upload  → POST /detect
     2 → review  → edit boxes + OCR text  → POST /inpaint
     3 → inpaint preview  → adjust text boxes  → POST /render
     4 → result
   ═══════════════════════════════════════════════════════════════════════════ */

const API = 'http://localhost:8000';

/* ── state ───────────────────────────────────────────────────────────────── */
let originalFile = null;
let originalB64  = null;
let inpaintedB64 = null;
let imgW = 0, imgH = 0;

let bubbles    = [];     // [{id, bx,by,bw,bh, tx,ty,tw,th, has_text_box, japanese}]
let selectedId = null;
let editMode   = 'bubble';   // 'bubble' | 'text'  — which sub-box is active in step 2

let drawStart  = null;   // {cx, cy} image-coords, set on pointerdown on empty space
let dragState  = null;   // {type, bubId, field, handle?, startCx, startCy, origBox}

const HANDLE_R = 7;      // hit-test radius for resize handles

/* ── canvas elements ─────────────────────────────────────────────────────── */
const reviewCanvas  = document.getElementById('review-canvas');
const rCtx          = reviewCanvas.getContext('2d');
const inpaintCanvas = document.getElementById('inpaint-canvas');
const iCtx          = inpaintCanvas.getContext('2d');

// HTMLImageElement caches — one per canvas
let reviewImg    = null;
let inpaintImg   = null;
let reviewScale  = 1;
let inpaintScale = 1;

/* ═══════════════════════════════════════════════════════════════════════════
   IMAGE LOADING
   loadImage() is promise-based, adds a cache-bust param so the browser
   never serves a stale decode for re-uploaded files with the same name,
   and rejects on error instead of silently failing.
   ═══════════════════════════════════════════════════════════════════════════ */
function loadImage(b64) {
  return new Promise((resolve, reject) => {
    const img    = new Image();
    img.onload   = () => resolve(img);
    img.onerror  = () => reject(new Error('Falha ao decodificar imagem'));
    // Append a random query-param to defeat any browser image-decode cache
    img.src = 'data:image/png;base64,' + b64 + '#' + Math.random();
  });
}

/* ═══════════════════════════════════════════════════════════════════════════
   CANVAS SIZING
   fitToParent() must be called *after* the canvas container is visible
   (i.e. after setStep() + requestAnimationFrame) so clientWidth is non-zero.
   Returns the scale factor applied.
   ═══════════════════════════════════════════════════════════════════════════ */
function fitToParent(canvas, img) {
  const maxW = canvas.parentElement.clientWidth || img.naturalWidth;
  const s    = Math.min(1, maxW / img.naturalWidth);
  canvas.width  = Math.round(img.naturalWidth  * s);
  canvas.height = Math.round(img.naturalHeight * s);
  return s;
}

/* ═══════════════════════════════════════════════════════════════════════════
   DRAWING — step 2 (review)
   ═══════════════════════════════════════════════════════════════════════════ */
function redrawReview() {
  if (!reviewImg) return;
  const s = reviewScale;
  rCtx.clearRect(0, 0, reviewCanvas.width, reviewCanvas.height);
  rCtx.drawImage(reviewImg, 0, 0, reviewCanvas.width, reviewCanvas.height);

  bubbles.forEach(b => {
    const sel   = b.id === selectedId;
    const num   = String(b.id + 1);

    // ── balloon box — red, dashed ─────────────────────────────────────────
    const bAlpha  = (sel && editMode === 'bubble') ? 1 : 0.8;
    rCtx.strokeStyle = `rgba(226,75,74,${bAlpha})`;
    rCtx.lineWidth   = (sel && editMode === 'bubble') ? 2.5 : 1.5;
    rCtx.setLineDash([5, 3]);
    rCtx.strokeRect(b.bx * s, b.by * s, b.bw * s, b.bh * s);
    rCtx.setLineDash([]);

    // ── text box — blue, solid ────────────────────────────────────────────
    const tAlpha  = (sel && editMode === 'text') ? 1 : 0.8;
    rCtx.strokeStyle = `rgba(55,138,221,${tAlpha})`;
    rCtx.lineWidth   = (sel && editMode === 'text') ? 2.5 : 1.5;
    rCtx.strokeRect(b.tx * s, b.ty * s, b.tw * s, b.th * s);

    // ── label above balloon box ───────────────────────────────────────────
    rCtx.font = 'bold 11px sans-serif';
    const lw  = rCtx.measureText(num).width;
    rCtx.fillStyle = sel ? '#1a1917' : '#555';
    rCtx.fillRect(b.bx * s, b.by * s - 18, lw + 10, 18);
    rCtx.fillStyle = '#fff';
    rCtx.fillText(num, b.bx * s + 5, b.by * s - 5);

    // ── resize handles for the active sub-box ─────────────────────────────
    if (sel) {
      const hbox  = editMode === 'bubble'
        ? { x: b.bx, y: b.by, w: b.bw, h: b.bh }
        : { x: b.tx, y: b.ty, w: b.tw, h: b.th };
      const color = editMode === 'bubble' ? '#E24B4A' : '#378ADD';
      getHandles(hbox, s).forEach(h => drawHandle(rCtx, h.px, h.py, color));
    }
  });
}

/* ── step 3 (inpaint preview) — text boxes only ──────────────────────────── */
function redrawInpaint() {
  if (!inpaintImg) return;
  const s = inpaintScale;
  iCtx.clearRect(0, 0, inpaintCanvas.width, inpaintCanvas.height);
  iCtx.drawImage(inpaintImg, 0, 0, inpaintCanvas.width, inpaintCanvas.height);

  bubbles.forEach(b => {
    const sel = b.id === selectedId;
    iCtx.strokeStyle = sel ? '#378ADD' : 'rgba(55,138,221,0.7)';
    iCtx.lineWidth   = sel ? 2.5 : 1.5;
    iCtx.setLineDash([]);
    iCtx.strokeRect(b.tx * s, b.ty * s, b.tw * s, b.th * s);

    const num = String(b.id + 1);
    iCtx.font = 'bold 11px sans-serif';
    const lw  = iCtx.measureText(num).width;
    iCtx.fillStyle = sel ? '#378ADD' : 'rgba(55,138,221,0.8)';
    iCtx.fillRect(b.tx * s, b.ty * s - 18, lw + 10, 18);
    iCtx.fillStyle = '#fff';
    iCtx.fillText(num, b.tx * s + 5, b.ty * s - 5);

    if (sel) {
      getHandles({ x: b.tx, y: b.ty, w: b.tw, h: b.th }, s)
        .forEach(h => drawHandle(iCtx, h.px, h.py, '#378ADD'));
    }
  });
}

/* ── handle geometry helpers ─────────────────────────────────────────────── */
function getHandles(box, s) {
  const { x, y, w, h } = box;
  const cx = x * s, cy = y * s, cw = w * s, ch = h * s;
  return [
    { name: 'tl', px: cx,       py: cy        },
    { name: 'tm', px: cx + cw/2, py: cy       },
    { name: 'tr', px: cx + cw,  py: cy        },
    { name: 'ml', px: cx,       py: cy + ch/2 },
    { name: 'mr', px: cx + cw,  py: cy + ch/2 },
    { name: 'bl', px: cx,       py: cy + ch   },
    { name: 'bm', px: cx + cw/2, py: cy + ch  },
    { name: 'br', px: cx + cw,  py: cy + ch   },
  ];
}

function drawHandle(ctx, px, py, color) {
  ctx.beginPath();
  ctx.arc(px, py, HANDLE_R, 0, Math.PI * 2);
  ctx.fillStyle   = '#fff';
  ctx.fill();
  ctx.strokeStyle = color;
  ctx.lineWidth   = 2;
  ctx.stroke();
}

function hitHandle(box, s, px, py) {
  return getHandles(box, s).find(
    h => Math.hypot(h.px - px, h.py - py) <= HANDLE_R + 2
  ) || null;
}

function ptInBox(bbl, field, s, px, py) {
  const x = bbl[field[0] + 'x'] * s;
  const y = bbl[field[0] + 'y'] * s;
  const w = bbl[field[0] + 'w'] * s;
  const h = bbl[field[0] + 'h'] * s;
  return px >= x && px <= x + w && py >= y && py <= y + h;
}

/* ── drag application ────────────────────────────────────────────────────── */
function applyDrag(origBox, dragState, dx, dy) {
  const o = origBox;
  if (dragState.type === 'move') {
    return { x: Math.round(Math.max(0, o.x + dx)), y: Math.round(Math.max(0, o.y + dy)), w: o.w, h: o.h };
  }
  let { x, y, w, h } = o;
  const n = dragState.handle;
  if (n.includes('l')) { x = o.x + dx; w = o.w - dx; }
  if (n.includes('r')) { w = o.w + dx; }
  if (n.includes('t')) { y = o.y + dy; h = o.h - dy; }
  if (n.includes('b')) { h = o.h + dy; }
  return { x: Math.round(x), y: Math.round(y), w: Math.round(w), h: Math.round(h) };
}

function writeBox(b, field, val) {
  if (val.w <= 10 || val.h <= 10) return;
  const p = field[0];  // 'b' or 't'
  b[p + 'x'] = val.x; b[p + 'y'] = val.y;
  b[p + 'w'] = val.w; b[p + 'h'] = val.h;
}

/* ═══════════════════════════════════════════════════════════════════════════
   POINTER HELPERS
   ═══════════════════════════════════════════════════════════════════════════ */
function canvasXY(canvas, e) {
  const r = canvas.getBoundingClientRect();
  return {
    px: (e.touches ? e.touches[0].clientX : e.clientX) - r.left,
    py: (e.touches ? e.touches[0].clientY : e.clientY) - r.top,
  };
}

/* ═══════════════════════════════════════════════════════════════════════════
   CANVAS EVENTS — step 2 (review)
   ═══════════════════════════════════════════════════════════════════════════ */
reviewCanvas.addEventListener('pointerdown', e => {
  e.preventDefault();
  reviewCanvas.setPointerCapture(e.pointerId);
  const { px, py } = canvasXY(reviewCanvas, e);
  const s = reviewScale;

  // 1. Resize handle of active sub-box of the selected bubble
  if (selectedId !== null) {
    const sel = bubbles.find(b => b.id === selectedId);
    if (sel) {
      const hbox = editMode === 'bubble'
        ? { x: sel.bx, y: sel.by, w: sel.bw, h: sel.bh }
        : { x: sel.tx, y: sel.ty, w: sel.tw, h: sel.th };
      const h = hitHandle(hbox, s, px, py);
      if (h) {
        dragState = { type: 'resize', handle: h.name, bubId: sel.id, field: editMode,
                      startCx: px, startCy: py, origBox: { ...hbox } };
        return;
      }
    }
  }

  // 2. Hit an existing bubble (check active-mode box first, then fallback to either)
  for (let i = bubbles.length - 1; i >= 0; i--) {
    const b = bubbles[i];
    const inBubble = ptInBox(b, 'bubble', s, px, py);
    const inText   = ptInBox(b, 'text',   s, px, py);
    if (inBubble || inText) {
      selectBubble(b.id);
      // Prefer the box that was actually hit in the current editMode
      const field    = (editMode === 'text' && inText) ? 'text'
                     : (editMode === 'bubble' && inBubble) ? 'bubble'
                     : inBubble ? 'bubble' : 'text';
      const origBox  = field === 'bubble'
        ? { x: b.bx, y: b.by, w: b.bw, h: b.bh }
        : { x: b.tx, y: b.ty, w: b.tw, h: b.th };
      dragState = { type: 'move', bubId: b.id, field,
                    startCx: px, startCy: py, origBox };
      return;
    }
  }

  // 3. Empty space → draw a new bubble
  selectBubble(null);
  drawStart = { cx: px / s, cy: py / s };
});

reviewCanvas.addEventListener('pointermove', e => {
  e.preventDefault();
  const { px, py } = canvasXY(reviewCanvas, e);
  const s = reviewScale;

  if (drawStart) {
    redrawReview();
    const x = Math.min(drawStart.cx, px / s) * s;
    const y = Math.min(drawStart.cy, py / s) * s;
    const w = Math.abs(px - drawStart.cx * s);
    const h = Math.abs(py - drawStart.cy * s);
    rCtx.strokeStyle = editMode === 'bubble' ? '#E24B4A' : '#378ADD';
    rCtx.lineWidth   = 2;
    rCtx.setLineDash(editMode === 'bubble' ? [5, 3] : []);
    rCtx.strokeRect(x, y, w, h);
    rCtx.setLineDash([]);
    return;
  }

  if (!dragState) return;
  const dx  = (px - dragState.startCx) / s;
  const dy  = (py - dragState.startCy) / s;
  const b   = bubbles.find(b => b.id === dragState.bubId);
  if (!b) return;
  const val = applyDrag(dragState.origBox, dragState, dx, dy);
  writeBox(b, dragState.field, val);
  redrawReview();
  updateListCoords(b);
});

// Single handler for pointerup AND pointercancel so drag never gets stuck
function onReviewPointerEnd(e) {
  e.preventDefault();
  const { px, py } = canvasXY(reviewCanvas, e);
  const s = reviewScale;

  if (drawStart && e.type === 'pointerup') {
    const x = Math.round(Math.min(drawStart.cx, px / s));
    const y = Math.round(Math.min(drawStart.cy, py / s));
    const w = Math.round(Math.abs(px / s - drawStart.cx));
    const h = Math.round(Math.abs(py / s - drawStart.cy));
    if (w > 10 && h > 10) {
      const newId = bubbles.length ? Math.max(...bubbles.map(b => b.id)) + 1 : 0;
      bubbles.push({ id: newId,
        bx: x, by: y, bw: w, bh: h,
        tx: x, ty: y, tw: w, th: h,
        has_text_box: false, japanese: '' });
      renderBubbleList();
      selectBubble(newId);
      runOcrOnBox(newId, x, y, w, h);
    }
  }

  drawStart = null;
  dragState = null;
  redrawReview();
}
reviewCanvas.addEventListener('pointerup',     onReviewPointerEnd);
reviewCanvas.addEventListener('pointercancel', onReviewPointerEnd);

/* ═══════════════════════════════════════════════════════════════════════════
   CANVAS EVENTS — step 3 (inpaint preview, text box only)
   ═══════════════════════════════════════════════════════════════════════════ */
inpaintCanvas.addEventListener('pointerdown', e => {
  e.preventDefault();
  inpaintCanvas.setPointerCapture(e.pointerId);
  const { px, py } = canvasXY(inpaintCanvas, e);
  const s = inpaintScale;

  if (selectedId !== null) {
    const sel = bubbles.find(b => b.id === selectedId);
    if (sel) {
      const h = hitHandle({ x: sel.tx, y: sel.ty, w: sel.tw, h: sel.th }, s, px, py);
      if (h) {
        dragState = { type: 'resize', handle: h.name, bubId: sel.id, field: 'text',
                      startCx: px, startCy: py,
                      origBox: { x: sel.tx, y: sel.ty, w: sel.tw, h: sel.th } };
        return;
      }
    }
  }

  for (let i = bubbles.length - 1; i >= 0; i--) {
    const b = bubbles[i];
    if (ptInBox(b, 'text', s, px, py)) {
      selectBubbleInpaint(b.id);
      dragState = { type: 'move', bubId: b.id, field: 'text',
                    startCx: px, startCy: py,
                    origBox: { x: b.tx, y: b.ty, w: b.tw, h: b.th } };
      return;
    }
  }
  selectBubbleInpaint(null);
});

inpaintCanvas.addEventListener('pointermove', e => {
  e.preventDefault();
  if (!dragState) return;
  const { px, py } = canvasXY(inpaintCanvas, e);
  const s  = inpaintScale;
  const dx = (px - dragState.startCx) / s;
  const dy = (py - dragState.startCy) / s;
  const b  = bubbles.find(b => b.id === dragState.bubId);
  if (!b) return;
  const val = applyDrag(dragState.origBox, dragState, dx, dy);
  writeBox(b, 'text', val);
  redrawInpaint();
  updateInpaintListCoords(b);
});

function onInpaintPointerEnd(e) {
  e.preventDefault();
  dragState = null;
}
inpaintCanvas.addEventListener('pointerup',     onInpaintPointerEnd);
inpaintCanvas.addEventListener('pointercancel', onInpaintPointerEnd);

/* ── Delete key ──────────────────────────────────────────────────────────── */
document.addEventListener('keydown', e => {
  if (e.key !== 'Delete' && e.key !== 'Backspace') return;
  if (document.activeElement?.tagName === 'TEXTAREA') return;
  if (selectedId === null) return;
  bubbles    = bubbles.filter(b => b.id !== selectedId);
  selectedId = null;
  renderBubbleList();
  redrawReview();
  renderInpaintList();
  redrawInpaint();
});

/* ── box-mode toggle ─────────────────────────────────────────────────────── */
document.querySelectorAll('.toggle-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    editMode = btn.dataset.mode;
    document.querySelectorAll('.toggle-btn').forEach(b =>
      b.classList.toggle('active', b.dataset.mode === editMode)
    );
    redrawReview();
  });
});

/* ═══════════════════════════════════════════════════════════════════════════
   BUBBLE LIST — step 2
   ═══════════════════════════════════════════════════════════════════════════ */
function coordsText(b) {
  return `balão ${b.bx},${b.by} ${b.bw}×${b.bh} · texto ${b.tx},${b.ty} ${b.tw}×${b.th}`;
}

function renderBubbleList() {
  const list = document.getElementById('bubble-list');
  document.getElementById('bub-count').textContent = bubbles.length;

  if (!bubbles.length) {
    list.innerHTML = '<div style="padding:16px;font-size:13px;color:var(--muted)">Nenhum balão. Clique e arraste para criar.</div>';
    return;
  }

  list.innerHTML = bubbles.map((b, i) => `
    <div class="bub-item${b.id === selectedId ? ' selected' : ''}" data-id="${b.id}" id="bub-item-${b.id}">
      <div class="bub-item-head">
        <div class="bub-num">${i + 1}</div>
        <div class="bub-jp" id="bub-jp-${b.id}">${escHtml(b.japanese) || '<span style="color:var(--faint)">sem texto</span>'}</div>
      </div>
      <textarea data-id="${b.id}" placeholder="Texto OCR (japonês)...">${escHtml(b.japanese)}</textarea>
      <div class="bub-meta">
        <div class="bub-coords" id="coords-${b.id}">${coordsText(b)}</div>
        <button class="btn-danger" data-del="${b.id}">Remover</button>
      </div>
    </div>
  `).join('');

  list.querySelectorAll('.bub-item').forEach(el => {
    el.addEventListener('click', e => {
      if (e.target.tagName === 'BUTTON' || e.target.tagName === 'TEXTAREA') return;
      selectBubble(Number(el.dataset.id));
    });
  });

  list.querySelectorAll('textarea').forEach(ta => {
    ta.addEventListener('input', () => {
      const b = bubbles.find(b => b.id === Number(ta.dataset.id));
      if (!b) return;
      b.japanese = ta.value;
      const head = document.getElementById('bub-jp-' + b.id);
      if (head) head.innerHTML = escHtml(ta.value) || '<span style="color:var(--faint)">sem texto</span>';
    });
    ta.addEventListener('focus', () => selectBubble(Number(ta.dataset.id)));
  });

  list.querySelectorAll('[data-del]').forEach(btn => {
    btn.addEventListener('click', ev => {
      ev.stopPropagation();
      const id = Number(btn.dataset.del);
      bubbles    = bubbles.filter(b => b.id !== id);
      if (selectedId === id) selectedId = null;
      renderBubbleList();
      redrawReview();
    });
  });
}

function updateListCoords(b) {
  const el = document.getElementById('coords-' + b.id);
  if (el) el.textContent = coordsText(b);
}

function selectBubble(id) {
  selectedId = id;
  document.querySelectorAll('#bubble-list .bub-item').forEach(el =>
    el.classList.toggle('selected', Number(el.dataset.id) === id)
  );
  if (id !== null) {
    document.getElementById('bub-item-' + id)
      ?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }
  redrawReview();
}

/* ── bubble list — step 3 ────────────────────────────────────────────────── */
function renderInpaintList() {
  const list    = document.getElementById('inpaint-bubble-list');
  const withTxt = bubbles.filter(b => b.japanese.trim());
  document.getElementById('inp-bub-count').textContent = withTxt.length;

  if (!withTxt.length) {
    list.innerHTML = '<div style="padding:16px;font-size:13px;color:var(--muted)">Nenhum balão com texto.</div>';
    return;
  }

  list.innerHTML = withTxt.map((b, i) => `
    <div class="bub-item${b.id === selectedId ? ' selected' : ''}" data-id="${b.id}" id="inp-item-${b.id}">
      <div class="bub-item-head">
        <div class="bub-num">${i + 1}</div>
        <div class="bub-jp">${escHtml(b.japanese)}</div>
      </div>
      <div class="bub-meta">
        <div class="bub-coords" id="inp-coords-${b.id}">${b.tx},${b.ty} · ${b.tw}×${b.th}px</div>
      </div>
    </div>
  `).join('');

  list.querySelectorAll('.bub-item').forEach(el => {
    el.addEventListener('click', () => selectBubbleInpaint(Number(el.dataset.id)));
  });
}

function updateInpaintListCoords(b) {
  const el = document.getElementById('inp-coords-' + b.id);
  if (el) el.textContent = `${b.tx},${b.ty} · ${b.tw}×${b.th}px`;
}

function selectBubbleInpaint(id) {
  selectedId = id;
  document.querySelectorAll('#inpaint-bubble-list .bub-item').forEach(el =>
    el.classList.toggle('selected', Number(el.dataset.id) === id)
  );
  if (id !== null) {
    document.getElementById('inp-item-' + id)
      ?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }
  redrawInpaint();
}

/* ═══════════════════════════════════════════════════════════════════════════
   STEP NAVIGATION
   setStep() shows/hides sections, then uses requestAnimationFrame to ensure
   the newly-visible canvas container has been laid out before we call
   fitToParent() — this fixes the "image only appears after resize" bug.
   ═══════════════════════════════════════════════════════════════════════════ */
const STEP_IDS = ['step-upload', 'step-review', 'step-inpaint', 'step-results'];

function setStep(n) {
  STEP_IDS.forEach((id, idx) => {
    document.getElementById(id).style.display = (idx + 1 === n) ? 'block' : 'none';
  });
  [1, 2, 3, 4].forEach(i => {
    const el = document.getElementById('si-' + i);
    if (el) el.className = 'step-item' + (i === n ? ' active' : i < n ? ' done' : '');
  });

  // Re-fit canvases after layout is updated
  requestAnimationFrame(() => {
    if (n === 2 && reviewImg) {
      reviewScale = fitToParent(reviewCanvas, reviewImg);
      redrawReview();
    }
    if (n === 3 && inpaintImg) {
      inpaintScale = fitToParent(inpaintCanvas, inpaintImg);
      redrawInpaint();
    }
  });
}

/* ── progress / error ────────────────────────────────────────────────────── */
function showProgress(label, pct) {
  document.getElementById('progress-bar-wrap').style.display = 'block';
  document.getElementById('pbar-fill').style.width = pct + '%';
  document.getElementById('pbar-label').textContent = label;
}
function hideProgress() {
  document.getElementById('progress-bar-wrap').style.display = 'none';
}
function showError(msg) {
  const el = document.getElementById('error-box');
  el.textContent   = msg;
  el.style.display = msg ? 'block' : 'none';
}

/* ═══════════════════════════════════════════════════════════════════════════
   UPLOAD + /detect
   ═══════════════════════════════════════════════════════════════════════════ */
const fileInput  = document.getElementById('file-input');
const uploadZone = document.getElementById('upload-zone');

fileInput.addEventListener('change', e => { if (e.target.files[0]) doDetect(e.target.files[0]); });
uploadZone.addEventListener('dragover',  e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', ()  => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f?.type.startsWith('image/')) doDetect(f);
});

async function doDetect(file) {
  showError('');
  showProgress('Detectando balões…', 20);
  originalFile = file;

  const form = new FormData();
  form.append('file', file);

  try {
    const resp = await fetch(API + '/detect', { method: 'POST', body: form });
    if (!resp.ok) { const e = await resp.json(); throw new Error(e.detail); }
    const data = await resp.json();

    originalB64 = data.original_image;
    imgW        = data.image_width;
    imgH        = data.image_height;
    bubbles     = data.bubbles;
    selectedId  = null;
    inpaintedB64 = null;
    inpaintImg   = null;

    showProgress('Carregando imagem…', 70);
    reviewImg = await loadImage(originalB64);

    // setStep first so the canvas container becomes visible + has layout width,
    // then rAF inside setStep will call fitToParent + redraw correctly.
    setStep(2);
    renderBubbleList();
    hideProgress();

  } catch (err) {
    hideProgress();
    showError('Erro na detecção: ' + err.message);
  }
}

/* ── OCR on a manually-drawn box ─────────────────────────────────────────── */
async function runOcrOnBox(bubId, x, y, w, h) {
  const ta = document.querySelector(`#bubble-list textarea[data-id="${bubId}"]`);
  if (ta) { ta.value = '…'; ta.disabled = true; }

  try {
    const form = new FormData();
    form.append('file', originalFile);
    form.append('x', x); form.append('y', y);
    form.append('w', w); form.append('h', h);
    const resp = await fetch(API + '/ocr-crop', { method: 'POST', body: form });
    if (!resp.ok) throw new Error('OCR falhou');
    const data = await resp.json();
    const b    = bubbles.find(b => b.id === bubId);
    if (b) b.japanese = data.japanese;
    if (ta) {
      ta.value    = data.japanese;
      ta.disabled = false;
      const head  = document.getElementById('bub-jp-' + bubId);
      if (head) head.innerHTML = escHtml(data.japanese) || '<span style="color:var(--faint)">sem texto</span>';
    }
  } catch (err) {
    console.warn('OCR crop error:', err);
    if (ta) { ta.value = ''; ta.disabled = false; }
  }
}

/* ── back: step 2 → 1 ───────────────────────────────────────────────────── */
document.getElementById('btn-back-upload').addEventListener('click', () => {
  fileInput.value = '';
  setStep(1);
  showError('');
});

/* ── step 2 → 3: /inpaint ────────────────────────────────────────────────── */
document.getElementById('btn-do-inpaint').addEventListener('click', async () => {
  const toInpaint = bubbles.filter(b => b.japanese.trim());
  if (!toInpaint.length) { showError('Nenhum balão com texto para processar.'); return; }

  showError('');
  showProgress('Inpainting…', 20);

  const form = new FormData();
  form.append('file', originalFile);
  form.append('bubbles', JSON.stringify(toInpaint));

  try {
    const resp = await fetch(API + '/inpaint', { method: 'POST', body: form });
    if (!resp.ok) { const e = await resp.json(); throw new Error(e.detail); }
    const data = await resp.json();

    inpaintedB64 = data.inpainted_image;
    selectedId   = null;

    showProgress('Carregando…', 80);
    inpaintImg = await loadImage(inpaintedB64);

    setStep(3);
    renderInpaintList();
    hideProgress();

  } catch (err) {
    hideProgress();
    showError('Erro no inpainting: ' + err.message);
  }
});

/* ── back: step 3 → 2 ───────────────────────────────────────────────────── */
document.getElementById('btn-back-review').addEventListener('click', () => {
  selectedId = null;
  setStep(2);
  showError('');
});

/* ── step 3 → 4: /render ────────────────────────────────────────────────── */
document.getElementById('btn-do-translate').addEventListener('click', async () => {
  const toRender = bubbles.filter(b => b.japanese.trim());
  if (!toRender.length) { showError('Nenhum balão com texto.'); return; }

  showError('');
  showProgress('Traduzindo e renderizando…', 20);

  try {
    const selectedFont = document.getElementById('font-select').value || null;
    const resp = await fetch(API + '/render', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ inpainted_b64: inpaintedB64, bubbles: toRender, font: selectedFont }),
    });
    if (!resp.ok) { const e = await resp.json(); throw new Error(e.detail); }
    const data = await resp.json();

    showProgress('Renderizando…', 90);
    document.getElementById('img-original').src   = 'data:image/png;base64,' + originalB64;
    document.getElementById('img-translated').src = 'data:image/png;base64,' + data.translated_image;

    const resList = data.bubbles || [];
    document.getElementById('res-count').textContent = resList.length;
    document.getElementById('res-list').innerHTML = resList.map(b => `
      <div class="bubble-row">
        <div class="b-num">${b.id + 1}</div>
        <div style="font-size:12px">${escHtml(b.japanese)}</div>
        <div style="font-size:12px;color:var(--muted)">${escHtml(b.portuguese)}</div>
      </div>
    `).join('');

    document.getElementById('btn-download').onclick = () => {
      const a    = document.createElement('a');
      a.href     = 'data:image/png;base64,' + data.translated_image;
      a.download = 'manga_traduzido.png';
      a.click();
    };

    document.getElementById('btn-copy-texts').onclick = () => {
      const txt = resList.map(b =>
        `[${b.id + 1}] JP: ${b.japanese}\n     PT: ${b.portuguese}`
      ).join('\n\n');
      navigator.clipboard.writeText(txt).then(() => {
        const btn = document.getElementById('btn-copy-texts');
        btn.textContent = 'Copiado!';
        setTimeout(() => btn.textContent = 'Copiar textos', 1500);
      });
    };

    hideProgress();
    setStep(4);

  } catch (err) {
    hideProgress();
    showError('Erro na tradução: ' + err.message);
  }
});

/* ── new image ───────────────────────────────────────────────────────────── */
document.getElementById('btn-new').addEventListener('click', () => {
  fileInput.value = '';
  bubbles = []; selectedId = null;
  originalB64 = null; inpaintedB64 = null;
  reviewImg = null; inpaintImg = null;
  setStep(1); showError('');
});

/* ── window resize — refit both canvases ─────────────────────────────────── */
window.addEventListener('resize', () => {
  if (reviewImg) {
    reviewScale = fitToParent(reviewCanvas, reviewImg);
    redrawReview();
  }
  if (inpaintImg) {
    inpaintScale = fitToParent(inpaintCanvas, inpaintImg);
    redrawInpaint();
  }
});

/* ── helpers ─────────────────────────────────────────────────────────────── */
function escHtml(s) {
  return String(s || '')
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

/* ── server health + font list ───────────────────────────────────────────── */
(async () => {
  const pill = document.getElementById('server-pill');
  try {
    const r = await fetch(API + '/health');
    if (r.ok) { pill.style.color = '#1D9E75'; pill.textContent = '⬤  online'; }
    else throw new Error();
  } catch { pill.style.color = '#E24B4A'; pill.textContent = '⬤  offline'; }
})();

(async () => {
  try {
    const r    = await fetch(API + '/fonts');
    if (!r.ok) return;
    const data = await r.json();
    const sel  = document.getElementById('font-select');
    data.fonts.forEach(name => {
      const opt   = document.createElement('option');
      opt.value   = name;
      opt.textContent = name.replace(/\.[^.]+$/, '');   // strip extension
      sel.appendChild(opt);
    });
  } catch { /* fonts/ missing or server offline — default option stays */ }
})();
