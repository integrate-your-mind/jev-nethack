import { copyFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(fileURLToPath(import.meta.url));
const workerPath = join(root, "worker.mjs");
const pagePath = join(root, "public", "index.html");
const hostingPath = join(root, ".openai", "hosting.json");
const outputWorker = join(root, "dist", "server", "index.js");
const outputHosting = join(root, "dist", ".openai", "hosting.json");
const placeholder = JSON.stringify("__JEV_SITE_INDEX_HTML__");

const [worker, page] = await Promise.all([
  readFile(workerPath, "utf8"),
  readFile(pagePath, "utf8"),
]);
if (!worker.includes(placeholder)) {
  throw new Error("worker.mjs is missing the HTML build placeholder");
}
const built = worker.replace(placeholder, () => JSON.stringify(page));
await Promise.all([
  mkdir(dirname(outputWorker), { recursive: true }),
  mkdir(dirname(outputHosting), { recursive: true }),
]);
await Promise.all([
  writeFile(outputWorker, built, "utf8"),
  copyFile(hostingPath, outputHosting),
]);
console.log(JSON.stringify({ built: "dist/server/index.js", hosting: "dist/.openai/hosting.json" }));
