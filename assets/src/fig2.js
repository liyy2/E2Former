#!/usr/bin/env node
/*
 * Generates assets/fig2.svg and assets/fig2.png (the E2Former method schematic).
 *
 *   cd assets/src && npm install && node fig2.js
 *
 * Equations are typeset by MathJax (TeX -> SVG paths, so the SVG needs no
 * fonts for the maths).  The PNG is rendered from the SVG with Playwright's
 * bundled Chromium at 3x.  Every position below is computed from a small
 * set of layout constants so the figure can be re-flowed by editing them.
 */
const fs = require("fs");
const path = require("path");

const { mathjax } = require("mathjax-full/js/mathjax.js");
const { TeX } = require("mathjax-full/js/input/tex.js");
const { SVG } = require("mathjax-full/js/output/svg.js");
const { liteAdaptor } = require("mathjax-full/js/adaptors/liteAdaptor.js");
const { RegisterHTMLHandler } = require("mathjax-full/js/handlers/html.js");
const { AllPackages } = require("mathjax-full/js/input/tex/AllPackages.js");

// ---------------------------------------------------------------- palette --
// One colour, one meaning: what a factor depends on.
const C = {
  ink: "#1f2328",       // text, operators
  muted: "#6b7280",     // captions, secondary labels
  scaffold: "#b8bec8",  // guide lines, cutoff circle, brackets
  node_i: "#8b2f97",    // depends on the centre atom i
  node_j: "#0b7a8f",    // depends on a neighbour j
  edge: "#d1451b",      // depends on the edge (i, j)
  paper: "#ffffff",
};
// Stroke weights (px at 1x): hair / norm / lead.
const W = { hair: 0.8, norm: 1.4, lead: 2.6 };
const FONT = "'Helvetica Neue', Helvetica, Arial, 'Liberation Sans', sans-serif";

// --------------------------------------------------------------- mathjax --
const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const mjDoc = mathjax.document("", {
  InputJax: new TeX({ packages: AllPackages }),
  OutputJax: new SVG({ fontCache: "none" }),
});

/** Typeset TeX; returns {w, h, asc, desc, svg(x, y)} with sizes in px. */
function tex(src, size = 26, color = C.ink) {
  const node = mjDoc.convert(src, { display: true });
  const svg = adaptor.firstChild(node);
  const vb = adaptor.getAttribute(svg, "viewBox").split(" ").map(Number);
  const inner = adaptor.innerHTML(svg);
  const s = size / 1000;
  const [minx, miny, vw, vh] = vb;
  const box = { w: vw * s, h: vh * s, asc: -miny * s, desc: (vh + miny) * s };
  // place the equation with its baseline at (x, y); anchor: 'start'|'middle'|'end'
  box.svg = (x, y, anchor = "start") => {
    const dx = anchor === "middle" ? -box.w / 2 : anchor === "end" ? -box.w : 0;
    return `<g transform="translate(${(x + dx).toFixed(2)},${y.toFixed(2)}) scale(${s})" color="${color}" fill="${color}">` +
      `<g transform="translate(${-minx},0)">${inner}</g></g>`;
  };
  return box;
}

// TeX macros shared by all equations.
const ci = (s) => `\\textcolor{${C.node_i}}{${s}}`;
const cj = (s) => `\\textcolor{${C.node_j}}{${s}}`;
const ce = (s) => `\\textcolor{${C.edge}}{${s}}`;
const CG = `\\overset{\\scriptscriptstyle\\mathrm{CG}}{\\otimes}`;
const SIXJ = `\\overset{\\scriptscriptstyle 6j}{\\otimes}`;
const R = (k, arg) => `\\mathcal{R}^{(${k})}(${arg})`;
const ri = `\\mathbf{r}_i`, rj = `\\mathbf{r}_j`, rij = `\\mathbf{r}_{ij}`;
const hj = `\\mathbf{h}_j`;

// ------------------------------------------------------------ primitives --
const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
function text(x, y, str, { size = 16, color = C.ink, weight = 400, anchor = "start", italic = false, ls = 0 } = {}) {
  return `<text x="${x}" y="${y}" font-family="${FONT}" font-size="${size}" font-weight="${weight}"` +
    ` fill="${color}" text-anchor="${anchor}"${italic ? ' font-style="italic"' : ""}` +
    `${ls ? ` letter-spacing="${ls}"` : ""}>${esc(str)}</text>`;
}
function line(x1, y1, x2, y2, { color = C.ink, w = W.norm, dash = "", cap = "round" } = {}) {
  return `<line x1="${x1.toFixed(2)}" y1="${y1.toFixed(2)}" x2="${x2.toFixed(2)}" y2="${y2.toFixed(2)}" stroke="${color}"` +
    ` stroke-width="${w}" stroke-linecap="${cap}"${dash ? ` stroke-dasharray="${dash}"` : ""}/>`;
}
/** Straight arrow with a computed round-tipped head (no markers, so colour and weight travel together). */
function arrow(x1, y1, x2, y2, { color = C.ink, w = W.norm, head = 9, shorten = 0 } = {}) {
  const dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy);
  const ux = dx / L, uy = dy / L;
  const ex = x2 - ux * shorten, ey = y2 - uy * shorten;   // tip
  const bx = ex - ux * head, by = ey - uy * head;         // base of head
  const hw = head * 0.42;
  const p1 = [bx - uy * hw, by + ux * hw], p2 = [bx + uy * hw, by - ux * hw];
  return line(x1, y1, bx + ux * w * 0.5, by + uy * w * 0.5, { color, w }) +
    `<path d="M${ex.toFixed(2)},${ey.toFixed(2)} L${p1[0].toFixed(2)},${p1[1].toFixed(2)} L${p2[0].toFixed(2)},${p2[1].toFixed(2)} Z"` +
    ` fill="${color}" stroke="${color}" stroke-width="${w * 0.6}" stroke-linejoin="round"/>`;
}
function disc(x, y, r, fill, label, { lsize = 15 } = {}) {
  return `<circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="${r}" fill="${fill}" stroke="${C.paper}" stroke-width="${W.norm}"/>` +
    (label ? text(x, y + lsize * 0.36, label, { size: lsize, color: C.paper, weight: 700, anchor: "middle", italic: true }) : "");
}

/**
 * Irreps glyph: rows l = 0..lmax, each of (2l+1) cells, centred, l = 0 on top.
 * Returns {svg, w, h}; (x, y) is the top-centre.
 */
function irreps(x, y, lmax, color, cell = 13, { lminOnly = null } = {}) {
  let out = "";
  const rows = lminOnly === null ? [...Array(lmax + 1).keys()] : [lminOnly];
  const wmax = (2 * lmax + 1) * cell;
  rows.forEach((l, k) => {
    const n = 2 * l + 1, w = n * cell, x0 = x - w / 2, y0 = y + k * cell;
    for (let m = 0; m < n; m++) {
      out += `<rect x="${(x0 + m * cell).toFixed(2)}" y="${y0.toFixed(2)}" width="${cell}" height="${cell}"` +
        ` fill="${color}" fill-opacity="${0.18 + 0.22 * (l / Math.max(lmax, 1))}" stroke="${color}" stroke-width="${W.hair}"/>`;
    }
  });
  return { svg: out, w: wmax, h: rows.length * cell };
}

/** Tall rounded parenthesis drawn as a path; side = 'l' | 'r'. */
function paren(x, ytop, ybot, side, color = C.scaffold, w = W.norm) {
  const h = ybot - ytop, bow = Math.min(9, h * 0.1), dir = side === "l" ? 1 : -1;
  return `<path d="M${(x + dir * bow).toFixed(2)},${ytop.toFixed(2)} Q${(x - dir * bow * 0.6).toFixed(2)},${(ytop + h / 2).toFixed(2)} ${(x + dir * bow).toFixed(2)},${ybot.toFixed(2)}"` +
    ` fill="none" stroke="${color}" stroke-width="${w}" stroke-linecap="round"/>`;
}

// ------------------------------------------------------------------ page --
const PAGE = { w: 1800, h: 1240, m: 44 };
const parts = [];
const put = (s) => parts.push(s);

// Column rails for panel (a).
const COL = {
  left: { x0: PAGE.m, x1: 540 },      // geometry sketch + identities
  mid: { x0: 600, x1: 1470 },         // the three equation stages
  rail: { x0: 1530, x1: PAGE.w - PAGE.m }, // tensor-product count
};
const midC = (COL.mid.x0 + COL.mid.x1) / 2;
const railC = (COL.rail.x0 + COL.rail.x1) / 2;

// ----- panel titles ---------------------------------------------------------
put(text(PAGE.m, 72, "a", { size: 26, weight: 700 }));
put(text(PAGE.m + 34, 72, "Wigner-6j recoupling moves the tensor products from the edges to the nodes", { size: 24, weight: 700 }));

// colour key: type colour carries meaning, no swatches
{
  const y = 104, x = PAGE.m + 34;
  const parts2 = [
    ["Colour marks what a factor depends on:", C.muted, 400],
    ["centre atom i", C.node_i, 700],
    ["·", C.muted, 400],
    ["neighbour j", C.node_j, 700],
    ["·", C.muted, 400],
    ["edge (i, j)", C.edge, 700],
  ];
  let t = `<text x="${x}" y="${y}" font-family="${FONT}" font-size="17">`;
  parts2.forEach(([s, c, wgt], k) => {
    t += `<tspan fill="${c}" font-weight="${wgt}"${k ? ' dx="9"' : ""}>${esc(s)}</tspan>`;
  });
  put(t + "</text>");
}

// ----- left column: geometry ------------------------------------------------
{
  const cx = 322, cy = 318;                         // centre atom i
  const rc = 122;                                   // cutoff radius (drawn)
  const O = { x: 84, y: 548 };                      // origin of the coordinate frame
  const nbrs = [160, 250, 340, 75].map((deg) => {   // neighbours inside the cutoff
    const a = (deg * Math.PI) / 180, d = 88;
    return { x: cx + d * Math.cos(a), y: cy - d * Math.sin(a) };
  });
  const far = [28, 330].map((deg) => {              // atoms beyond the cutoff (not in N(i))
    const a = (deg * Math.PI) / 180, d = 164;
    return { x: cx + d * Math.cos(a), y: cy - d * Math.sin(a) };
  });
  const J = nbrs[0];                                // the neighbour we label j

  // frame axes at the origin
  put(arrow(O.x, O.y, O.x + 46, O.y, { color: C.scaffold, w: W.hair, head: 6 }));
  put(arrow(O.x, O.y, O.x, O.y - 46, { color: C.scaffold, w: W.hair, head: 6 }));
  put(arrow(O.x, O.y, O.x - 26, O.y + 26, { color: C.scaffold, w: W.hair, head: 6 }));
  put(text(O.x - 8, O.y + 16, "O", { size: 13, color: C.muted, anchor: "end", italic: true }));

  // cutoff circle
  put(`<circle cx="${cx}" cy="${cy}" r="${rc}" fill="none" stroke="${C.scaffold}" stroke-width="${W.hair}" stroke-dasharray="5 5"/>`);
  put(text(cx + rc * 0.72 + 6, cy - rc * 0.72 - 4, "cutoff", { size: 13, color: C.muted, italic: true }));

  // bonds (scaffold) under the atoms; message arrows from each neighbour into i
  nbrs.forEach((p) => put(line(cx, cy, p.x, p.y, { color: C.scaffold, w: W.hair })));
  nbrs.forEach((p) => {
    // arrow along the bond, starting just outside the neighbour disc, ending at the rim of i
    const dx = cx - p.x, dy = cy - p.y, L = Math.hypot(dx, dy), ux = dx / L, uy = dy / L;
    put(arrow(p.x + ux * 22, p.y + uy * 22, cx - ux * 26, cy - uy * 26, { color: C.muted, w: W.norm, head: 8 }));
  });

  // position vectors: r_i (plum), r_j (teal), and the edge vector r_ij = r_i - r_j (vermillion, lead weight)
  put(arrow(O.x, O.y, cx, cy, { color: C.node_i, w: W.norm, head: 9, shorten: 21 }));
  put(arrow(O.x, O.y, J.x, J.y, { color: C.node_j, w: W.norm, head: 9, shorten: 16 }));
  put(arrow(J.x, J.y, cx, cy, { color: C.edge, w: W.lead, head: 11, shorten: 22 }));

  // atoms
  far.forEach((p) => put(disc(p.x, p.y, 12, C.scaffold, "")));
  nbrs.forEach((p, k) => put(disc(p.x, p.y, 15, C.node_j, k === 0 ? "j" : "")));
  put(disc(cx, cy, 21, C.node_i, "i", { lsize: 18 }));

  // vector labels (placed at fixed fractions along each vector, offset to its side)
  const lab = (a, b, f, off, s, col) => {
    const x = a.x + (b.x - a.x) * f, y = a.y + (b.y - a.y) * f;
    const dx = b.x - a.x, dy = b.y - a.y, L = Math.hypot(dx, dy);
    const box = tex(s, 19, col);
    put(box.svg(x + (-dy / L) * off, y + (dx / L) * off + box.asc / 2, "middle"));
  };
  lab(O, { x: cx, y: cy }, 0.56, -16, ri, C.node_i);
  lab(O, J, 0.5, 16, rj, C.node_j);
  lab(J, { x: cx, y: cy }, 0.5, 22, rij, C.edge);
  put(text(cx + 10, cy + rc + 24, "messages from all neighbours j of i", { size: 13, color: C.muted, anchor: "middle", italic: true }));

  // identities under the sketch
  let y = 618;
  const row = (label, src, size = 21) => {
    const box = tex(src, size);
    put(box.svg(COL.left.x0 + 4, y + box.asc, "start"));
    put(text(COL.left.x0 + 4, y + box.asc + box.desc + 19, label, { size: 14, color: C.muted, italic: true }));
    y += box.asc + box.desc + 44;
  };
  row("solid spherical harmonics of order ℓ",
    `\\mathcal{R}^{(\\ell)}_m(\\mathbf{r}) = |\\mathbf{r}|^{\\ell}\\, Y^{\\ell}_m(\\hat{\\mathbf{r}})`);
  row("order 1 is linear, so an edge factor splits into two node factors",
    `${ce(R(1, rij))} = ${ci(R(1, ri))} - ${cj(R(1, rj))}`);
  row("higher orders are tensor powers of the order-1 factor",
    `${ce(R(2, rij))} \\propto \\big[\\,${ce(R(1, rij))}^{\\otimes 2}\\big]^{(2)}`);
}

// ----- middle column: the three stages -------------------------------------
const stages = [
  {
    name: "SO(3) convolution",
    note: "the solid harmonic is evaluated on every edge",
    eq: `m_i = \\sum_{j\\in\\mathcal{N}(i)} ${cj(hj)} \\, ${CG} \\, ${ce(R(2, rij))}`,
    cost: "\\mathcal{O}(|\\mathcal{E}|)", costName: "edge-level",
  },
  {
    name: "Binomial expansion in tensor space",
    note: "no edge factor is left, but i and j are still multiplied inside the neighbour sum",
    eq: `m_i = \\sum_{j} ${cj(hj)} \\, ${CG} \\Big[\\, ${ci(R(1, ri) + "^{\\otimes 2}")} - 2\\, ${ci(R(1, ri))} \\, ${CG} \\, ${cj(R(1, rj))} + ${cj(R(1, rj) + "^{\\otimes 2}")} \\,\\Big]^{(2)}`,
    cost: "\\mathcal{O}(|\\mathcal{E}|)", costName: "edge-level",
  },
  {
    name: "Wigner-6j convolution",
    note: "every neighbour sum depends on j alone: aggregate once per node, then multiply by the centre factor",
    eq: `m_i = ${ci(R(2, ri))} \\, ${CG} \\, ${cj("\\sum_{j} " + hj)} \\;+\\; 2\\, ${ci(R(1, ri))} \\, ${SIXJ} \\, ${cj("\\sum_{j}\\big(" + hj + " \\otimes " + R(1, rj) + "\\big)")} \\;+\\; ${cj("\\sum_{j} " + hj + " \\, " + CG + " \\, " + R(2, rj))}`,
    cost: "\\mathcal{O}(|\\mathcal{V}|)", costName: "node-level",
  },
];
const arrowsBetween = [
  `\\text{expand } ${ce(R(2, rij))} \\text{ using } ${ce(rij)} = ${ci(ri)} - ${cj(rj)}`,
  `\\text{regroup by node, recouple with Wigner-6j symbols (panel b)}`,
];

const stageY = [232, 452, 690];   // equation baselines
put(text(railC, 150, "tensor-product count", { size: 15, color: C.muted, anchor: "middle", ls: 0.5 }));

stages.forEach((st, k) => {
  const y = stageY[k];
  let box = tex(st.eq, 27);
  const maxW = COL.mid.x1 - COL.mid.x0;
  if (box.w > maxW) box = tex(st.eq, Math.floor(27 * maxW / box.w));   // shrink-to-fit, never overflow
  // stage label + note above the equation
  put(text(COL.mid.x0, y - box.asc - 34, st.name, { size: 18, weight: 700 }));
  put(text(COL.mid.x0, y - box.asc - 12, st.note, { size: 14, color: C.muted, italic: true }));
  put(box.svg(midC, y, "middle"));
  st.bottom = y + box.desc;
  st.top = y - box.asc - 50;

  // cost on the right rail, on the same baseline as the equation
  const cost = tex(st.cost, 40, k === 2 ? C.ink : C.muted);
  put(cost.svg(railC, y, "middle"));
  put(text(railC, y + 26, st.costName, { size: 15, color: k === 2 ? C.ink : C.muted, anchor: "middle", weight: k === 2 ? 700 : 400 }));
  if (k === 2) put(text(railC, y + 46, "linear in atoms, not in neighbours", { size: 13, color: C.muted, anchor: "middle", italic: true }));
});

// arrows between stages, in the middle column, with their reason to the right
for (let k = 0; k < 2; k++) {
  const y1 = stages[k].bottom + 16, y2 = stages[k + 1].top - 10;
  put(arrow(midC, y1, midC, y2, { color: C.ink, w: W.norm, head: 9 }));
  const lb = tex(arrowsBetween[k], 15, C.muted);
  put(lb.svg(midC + 16, (y1 + y2) / 2 + lb.asc / 2 - 2, "start"));
}
// rail: the lead-weight arrow is spent on the one transition that changes the complexity
put(arrow(railC, stageY[1] + 66, railC, stageY[2] - 56, { color: C.ink, w: W.lead, head: 12 }));

// ----- panel (b): the recoupling identity ----------------------------------
const bY = 918;
put(line(PAGE.m, bY - 46, PAGE.w - PAGE.m, bY - 46, { color: C.scaffold, w: W.hair, cap: "butt" }));
put(text(PAGE.m, bY, "b", { size: 26, weight: 700 }));
put(text(PAGE.m + 34, bY, "The Wigner-6j symbol re-associates a double tensor product, so the neighbour-only pair is contracted first", { size: 24, weight: 700 }));

{
  const cell = 16;
  const gTop = bY + 92;                       // top of the irreps glyphs
  const gMid = gTop + 1.5 * cell + 1;         // vertical centre of the 3-row glyph
  const yop = gMid + 8;                       // baseline for operators
  const op = (s, x, col = C.ink) => { const b = tex(s, 24, col); put(b.svg(x, yop, "middle")); return b.w; };
  const glyph = (x, lmax, col, only = null) => { const g = irreps(x, only === null ? gTop : gMid - cell / 2, lmax, col, cell, { lminOnly: only }); put(g.svg); return g.w; };
  const lab = (x, ytop, s, col) => { const b = tex(s, 17, col); put(b.svg(x, ytop + b.asc, "middle")); return b; };

  // ---- left-hand side:  h_j  ⊗CG ( R(r_i) ⊗CG R(r_j) )
  let x = 330;
  glyph(x, 2, C.node_j); lab(x, gTop + 3 * cell + 12, cj(hj), C.node_j); x += 70;
  op(CG, x); x += 40;
  const pL1 = x; x += 34;
  glyph(x, 1, C.node_i, 1); lab(x, gTop + 3 * cell + 12, ci(R(1, ri)), C.node_i); x += 58;
  op(CG, x); x += 40;
  glyph(x, 1, C.node_j, 1); lab(x, gTop + 3 * cell + 12, cj(R(1, rj)), C.node_j); x += 34;
  const pL2 = x;
  put(paren(pL1, gTop - 18, gTop + 3 * cell + 18, "l", C.edge, W.norm));
  put(paren(pL2, gTop - 18, gTop + 3 * cell + 18, "r", C.edge, W.norm));
  put(text((pL1 + pL2) / 2, gTop + 3 * cell + 60, "inner product mixes i and j", { size: 14, color: C.edge, anchor: "middle", italic: true, weight: 700 }));
  put(text((pL1 + pL2) / 2, gTop + 3 * cell + 78, "recomputed on every edge", { size: 14, color: C.muted, anchor: "middle", italic: true }));

  // ---- equivalence
  x += 92;
  const eqv = tex("\\equiv", 30, C.ink); put(eqv.svg(x, yop, "middle"));
  put(text(x, yop + 34, "Wigner-6j", { size: 14, color: C.muted, anchor: "middle", italic: true }));
  put(text(x, yop + 51, "recoupling", { size: 14, color: C.muted, anchor: "middle", italic: true }));
  x += 96;

  // ---- right-hand side:  ( h_j ⊗CG R(r_j) )  ⊗6j  R(r_i)
  const pR1 = x; x += 60;
  glyph(x, 2, C.node_j); lab(x, gTop + 3 * cell + 12, cj(hj), C.node_j); x += 70;
  op(CG, x); x += 40;
  glyph(x, 1, C.node_j, 1); lab(x, gTop + 3 * cell + 12, cj(R(1, rj)), C.node_j); x += 36;
  const pR2 = x;
  put(paren(pR1, gTop - 18, gTop + 3 * cell + 18, "l", C.node_j, W.norm));
  put(paren(pR2, gTop - 18, gTop + 3 * cell + 18, "r", C.node_j, W.norm));
  put(text((pR1 + pR2) / 2, gTop + 3 * cell + 60, "inner product depends on j only", { size: 14, color: C.node_j, anchor: "middle", italic: true, weight: 700 }));
  put(text((pR1 + pR2) / 2, gTop + 3 * cell + 78, "summed over j once per node, before the last product", { size: 14, color: C.muted, anchor: "middle", italic: true }));
  x += 48;
  op(SIXJ, x); x += 46;
  glyph(x, 1, C.node_i, 1); lab(x, gTop + 3 * cell + 12, ci(R(1, ri)), C.node_i);
  const xEnd = x + 40;

  // ---- glyph key, at the far right
  const kx = 1500;
  const g = irreps(kx + 40, gTop, 2, C.muted, cell); put(g.svg);
  [0, 1, 2].forEach((l) => put(text(kx + 40 - g.w / 2 - 10, gTop + l * cell + 10.5, `ℓ = ${l}`, { size: 13, color: C.muted, anchor: "end" })));
  put(text(kx + 40, gTop + 3 * cell + 20, "irreducible representation", { size: 13, color: C.muted, anchor: "middle" }));
  put(text(kx + 40, gTop + 3 * cell + 37, "one cell per m, rows for each ℓ", { size: 13, color: C.muted, anchor: "middle", italic: true }));
  const g1 = irreps(kx + 40, gTop + 3 * cell + 62, 1, C.muted, cell, { lminOnly: 1 }); put(g1.svg);
  put(text(kx + 40, gTop + 3 * cell + 62 + cell + 18, "solid harmonic, ℓ = 1 (3 components)", { size: 13, color: C.muted, anchor: "middle" }));
  void xEnd;
}

// ----- assemble ---------------------------------------------------------------
const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${PAGE.w}" height="${PAGE.h}" viewBox="0 0 ${PAGE.w} ${PAGE.h}">
<rect width="${PAGE.w}" height="${PAGE.h}" fill="${C.paper}"/>
${parts.join("\n")}
</svg>`;

const outDir = path.resolve(__dirname, "..");
fs.writeFileSync(path.join(outDir, "fig2.svg"), svg);
console.log("wrote", path.join(outDir, "fig2.svg"));

// ----- rasterise with Chromium ---------------------------------------------------
(async () => {
  let playwright;
  try { playwright = require("playwright"); } catch { playwright = require("/opt/node22/lib/node_modules/playwright"); }
  const browser = await playwright.chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: PAGE.w, height: PAGE.h }, deviceScaleFactor: 3 });
  const page = await ctx.newPage();
  await page.setContent(`<html><body style="margin:0;background:#fff">${svg}</body></html>`);
  await page.screenshot({ path: path.join(outDir, "fig2.png"), clip: { x: 0, y: 0, width: PAGE.w, height: PAGE.h } });
  await browser.close();
  console.log("wrote", path.join(outDir, "fig2.png"));
})();
