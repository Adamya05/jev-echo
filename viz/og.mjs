// Render the social preview (docs/og.png): the results card, 1200x630.
//   node viz/og.mjs
import { spawn } from "node:child_process";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const PAGE = pathToFileURL(resolve("docs/race.html")).href + "?capture&theme=dark";
const OW = 1200, OH = 630, PORT = 9335;
const profile = mkdtempSync(join(tmpdir(), "echo-og-"));
const chrome = spawn("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  ["--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
   "--hide-scrollbars", "--disable-gpu", "about:blank"], { stdio: "ignore" });
const sleep = ms => new Promise(r => setTimeout(r, ms));
let target;
for (let i = 0; i < 50 && !target; i++) {
  try { target = (await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json())
          .find(t => t.type === "page"); } catch { await sleep(200); }
}
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise(r => ws.addEventListener("open", r, { once: true }));
let id = 0; const waiting = new Map();
ws.addEventListener("message", m => { const d = JSON.parse(m.data);
  if (d.id && waiting.has(d.id)) { waiting.get(d.id)(d); waiting.delete(d.id); } });
const send = (method, params = {}) => new Promise(r => {
  const n = ++id; waiting.set(n, r); ws.send(JSON.stringify({ id: n, method, params })); });
const evaluate = async e => (await send("Runtime.evaluate",
  { expression: e, awaitPromise: true, returnByValue: true })).result?.result?.value;

await send("Emulation.setDeviceMetricsOverride", { width: 1280, height: 720, deviceScaleFactor: 2, mobile: false });
await send("Page.navigate", { url: PAGE });
await sleep(1500);
await evaluate("document.fonts.ready.then(() => true)");
await evaluate("window.__setFrame(window.__beat, 5)");
await sleep(300);
// Frame the card with a margin, at the preview's aspect ratio.
const r = await evaluate("(() => { const b = document.querySelector('#ovl .card').getBoundingClientRect(); return {x:b.x, y:b.y, w:b.width, h:b.height}; })()");
const w = Math.ceil(Math.max(r.w + 96, (r.h + 64) * OW / OH) / 40) * 40, h = w * OH / OW;  // whole pixels
const clip = { x: r.x + r.w / 2 - w / 2, y: Math.max(0, r.y + r.h / 2 - h / 2), width: w, height: h, scale: OW / w / 2 };   // /2 for the device scale factor
const shot = await send("Page.captureScreenshot", { format: "png", clip });
writeFileSync("docs/og.png", Buffer.from(shot.result.data, "base64"));
ws.close(); chrome.kill(); await sleep(300);
rmSync(profile, { recursive: true, force: true });
console.log("docs/og.png");
