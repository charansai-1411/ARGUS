import { useEffect, useRef } from "react";
import cytoscape from "cytoscape";

const COLORS = { entry: "#ffb547", mule: "#ff5a5f", exit: "#b48cff", legit: "#5b6f8a" };
const LABEL = { entry: "VICTIM", exit: "CASH-OUT" };

const style = [
  { selector: "node", style: {
      "background-color": (n) => COLORS[n.data("role")] || COLORS.legit,
      label: (n) => (n.data("role") === "legit" ? "" :
        `${LABEL[n.data("role")] || `…${n.id().slice(-5)}`}\nBank ${n.data("bank")}`),
      "text-wrap": "wrap", "line-height": 1.3,
      color: "#e6edf6", "font-size": 10, "text-valign": "bottom", "text-margin-y": 5,
      width: 30, height: 30, "border-width": 2, "border-color": "#0b1320",
      "transition-property": "opacity, border-width, background-color", "transition-duration": "300ms" } },
  { selector: "node[role = 'legit']", style: { width: 10, height: 10 } },
  { selector: "edge", style: {
      "curve-style": "bezier", "target-arrow-shape": "triangle", "arrow-scale": 0.8, width: 1,
      "line-color": "#3a4e6b", "target-arrow-color": "#3a4e6b", opacity: 0.55,
      "transition-property": "opacity, line-color, width", "transition-duration": "350ms" } },
  { selector: "edge.ring", style: { width: 3.5, "line-color": "#ff5a5f", "target-arrow-color": "#ff5a5f", opacity: 1,
      label: (e) => `+${e.data("minute")}m`, "font-size": 9, color: "#8ea2bd", "text-background-color": "#0b1320",
      "text-background-opacity": 0.8, "text-background-padding": 2 } },
  { selector: "edge.camouflage", style: { "line-style": "dotted", "line-color": "#8ea2bd", "target-arrow-color": "#8ea2bd", opacity: 0.6 } },
  { selector: "edge.ring.miss", style: { "line-style": "dashed", "line-color": "#ff5a5f", "target-arrow-color": "#ff5a5f", opacity: 0.3, width: 2 } },
  { selector: "edge.ring.hit", style: { "line-color": "#3ddc97", "target-arrow-color": "#3ddc97", width: 4 } },
  { selector: ".hidden", style: { display: "none" } },
  { selector: ".ghost", style: { opacity: 0.12 } },
  { selector: "node.pulse", style: { "border-width": 12, "border-color": "#ff5a5f", "border-opacity": 0.55 } },
  { selector: "node:selected", style: { "border-width": 4, "border-color": "#4cc3ff" } },
];

const elements = (graph) => [
  ...graph.nodes.map((n) => ({ data: n })),
  ...graph.edges.map((e) => ({ data: e, classes: e.kind === "ring" ? "ring" : e.kind })),
];

/** Deterministic layout: ring accounts on a circle in the order the money reaches them;
 *  every other account fanned outward around the ring account it transacts with. */
function layout(c) {
  const ring = c.edges(".ring").sort((a, b) => a.data("minute") - b.data("minute"));
  const order = [];
  ring.forEach((e) => [e.source().id(), e.target().id()].forEach((id) => !order.includes(id) && order.push(id)));
  const R = Math.max(150, 55 * order.length);
  const pos = new Map(order.map((id, i) => {
    const a = -Math.PI / 2 + (2 * Math.PI * i) / order.length;
    return [id, { x: R * Math.cos(a), y: R * Math.sin(a), a }];
  }));
  const fan = new Map();
  c.nodes().forEach((n) => {
    if (pos.has(n.id())) return;
    const anchor = n.neighborhood("node").filter((m) => pos.has(m.id()))[0];
    if (!anchor) return pos.set(n.id(), { x: 0, y: 0 });
    const p = pos.get(anchor.id()), k = fan.get(anchor.id()) || 0;
    fan.set(anchor.id(), k + 1);
    const a = p.a + (k % 2 ? 1 : -1) * 0.22 * Math.ceil(k / 2), r = 110 + 26 * Math.floor(k / 9);
    pos.set(n.id(), { x: p.x + r * Math.cos(a), y: p.y + r * Math.sin(a) });
  });
  c.layout({ name: "preset", fit: false, positions: (n) => pos.get(n.id()) }).run();
}

/** mode: alert | reveal | full | adapt ; model: naive | hardened (adapt only) */
export default function Graph({ graph, mode, model, threshold, alertTx, onSelect }) {
  const box = useRef(null);
  const cy = useRef(null);
  const timers = useRef([]);

  useEffect(() => {
    cy.current = cytoscape({ container: box.current, elements: elements(graph), style, wheelSensitivity: 0.2,
      minZoom: 0.2, maxZoom: 1.6 });
    layout(cy.current);
    window.__argusCy = cy.current; // handy for debugging in the browser console
    cy.current.on("tap", "node", (e) => onSelect?.(e.target.data()));
    return () => cy.current.destroy();
  }, [graph]);

  useEffect(() => {
    const c = cy.current;
    timers.current.forEach(clearTimeout);
    timers.current = [];
    c.elements().removeClass("hidden ghost pulse hit miss");
    const ring = c.edges(".ring").sort((a, b) => a.data("minute") - b.data("minute"));
    const ringNodes = ring.connectedNodes();
    const victim = c.nodes("[role = 'entry']");

    // Fit the camera first: Cytoscape ignores hidden elements when fitting.
    if (mode === "alert") {
      const first = c.getElementById(`t${alertTx}`);
      c.fit(victim.union(first.target()), 140);
      c.elements().addClass("hidden");
      victim.removeClass("hidden");
      const pulse = () => {
        victim.toggleClass("pulse");
        timers.current.push(setTimeout(pulse, 650));
      };
      pulse();
      timers.current.push(setTimeout(() => {
        first.removeClass("hidden"); first.target().removeClass("hidden");
      }, 1200));
    } else if (mode === "reveal") {
      c.fit(ring.union(ringNodes), 60);
      c.elements().addClass("hidden");
      victim.removeClass("hidden");
      ring.forEach((e, i) => timers.current.push(setTimeout(() => {
        e.removeClass("hidden");
        e.connectedNodes().removeClass("hidden");
      }, 300 + i * 380)));
      timers.current.push(setTimeout(() => {
        c.elements().removeClass("hidden");
        c.elements().not(ring).not(ringNodes).addClass("ghost");
      }, 500 + ring.length * 380));
    } else if (mode === "full") {
      c.elements().not(ring).not(ringNodes).addClass("ghost");
      c.fit(undefined, 40);
    } else if (mode === "adapt") {
      ring.forEach((e) => e.addClass(e.data(`score_${model}`) >= threshold ? "hit" : "miss"));
      c.edges("[kind = 'background']").addClass("ghost");
      c.fit(undefined, 40);
    }
    return () => timers.current.forEach(clearTimeout);
  }, [graph, mode, model, threshold, alertTx]);

  return <div className="cy" ref={box} />;
}
