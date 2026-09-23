// Render the replay to video, frame by frame, with headless Chrome.
//   node viz/video.mjs [out.mp4]
// 1x for the opening seconds, then 5x to the results card, then a hold.
import { spawn } from "node:child_process";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const OUT = resolve(process.argv[2] || "echo.mp4");
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PAGE = pathToFileURL(resolve("docs/index.html")).href + "?capture&theme=dark&mode=net";
const W = 1280, H = 720, DPR = 1.5, FPS = 30;
const OPEN_S = 4, HOLD_S = 3;              // 1x opening, then 5x, then hold on the card
const KEEP_H = 712;                        // headline and lede, nothing below

const frames = mkdtempSync(join(tmpdir(), "echo-frames-"));
const profile = mkdtempSync(join(tmpdir(), "echo-chrome-"));
const PORT = 9333;
const chrome = spawn(CHROME, ["--headless=new", `--remote-debugging-port=${PORT}`,
  `--user-data-dir=${profile}`, "--hide-scrollbars", "--disable-gpu", "about:blank"],
  { stdio: "ignore" });

const sleep = ms => new Promise(r => setTimeout(r, ms));
let target;
for (let i = 0; i < 50 && !target; i++) {
  try { target = (await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json())
          .find(t => t.type === "page"); } catch { await sleep(200); }
}
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise(r => ws.addEventListener("open", r, { once: true }));
let id = 0; const waiting = new Map(); const events = [];
ws.addEventListener("message", m => {
  const d = JSON.parse(m.data);
  if (d.id && waiting.has(d.id)) { waiting.get(d.id)(d); waiting.delete(d.id); }
  else if (d.method) events.push(d.method);
});
const send = (method, params = {}) => new Promise(r => {
  const n = ++id; waiting.set(n, r); ws.send(JSON.stringify({ id: n, method, params }));
});
const evaluate = async expr => (await send("Runtime.evaluate",
  { expression: expr, awaitPromise: true, returnByValue: true })).result?.result?.value;

await send("Page.enable");
await send("Emulation.setDeviceMetricsOverride", { width: W, height: H, deviceScaleFactor: DPR, mobile: false });
await send("Page.navigate", { url: PAGE });
while (!events.includes("Page.loadEventFired")) await sleep(50);
await evaluate("document.fonts.ready.then(() => true)");
await sleep(300);

const beat = await evaluate("window.__beat");
const fastS = (beat / 1000 - OPEN_S) / 5;
const plan = [];
for (let f = 0; f < OPEN_S * FPS; f++) plan.push([f / FPS * 1000, 1]);
for (let f = 0; f < Math.round(fastS * FPS); f++) plan.push([OPEN_S * 1000 + f / FPS * 5000, 5]);
plan.push([beat, 5]);

let n = 0;
for (const [t, spd] of plan) {
  await evaluate(`window.__setFrame(${t}, ${spd})`);
  const shot = await send("Page.captureScreenshot", { format: "png" });
  writeFileSync(join(frames, `f${String(n++).padStart(5, "0")}.png`), Buffer.from(shot.result.data, "base64"));
}
// hold the last frame
for (let f = 0; f < HOLD_S * FPS; f++)
  writeFileSync(join(frames, `f${String(n++).padStart(5, "0")}.png`),
    Buffer.from((await send("Page.captureScreenshot", { format: "png" })).result.data, "base64"));
ws.close(); chrome.kill();

await new Promise((res, rej) => spawn("ffmpeg", ["-y", "-loglevel", "error", "-framerate", String(FPS),
  "-i", join(frames, "f%05d.png"), "-vf", `crop=iw:${KEEP_H * DPR}:0:0`, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
  "-preset", "slow", "-movflags", "+faststart", OUT], { stdio: "inherit" })
  .on("exit", c => c === 0 ? res() : rej(new Error("ffmpeg " + c))));
rmSync(frames, { recursive: true, force: true }); rmSync(profile, { recursive: true, force: true });
console.log(`${n} frames, ${(n / FPS).toFixed(1)} s -> ${OUT}`);
