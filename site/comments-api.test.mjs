import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { handleRequest } from "./worker.mjs";

const TOKEN = "test-ingest-token-with-enough-entropy";
const ORIGIN = "https://example.test";
const encoder = new TextEncoder();

async function bodyBytes(value) {
  if (typeof value === "string") return encoder.encode(value);
  if (value instanceof Uint8Array) return value;
  if (value instanceof ReadableStream) return new Uint8Array(await new Response(value).arrayBuffer());
  throw new TypeError("unsupported mock body");
}

class CommentBucket {
  constructor({ throwOnConflict = false } = {}) {
    this.objects = new Map();
    this.version = 0;
    this.throwOnConflict = throwOnConflict;
    this.headPause = null;
  }

  pauseNextRateHead() {
    let signalEntered;
    let release;
    const entered = new Promise((resolve) => { signalEntered = resolve; });
    const released = new Promise((resolve) => { release = resolve; });
    this.headPause = { signalEntered, released };
    return { entered, release };
  }

  metadata(key, entry) {
    return {
      key,
      size: entry.bytes.byteLength,
      etag: entry.etag,
      httpEtag: `"${entry.etag}"`,
      uploaded: entry.uploaded,
      httpMetadata: entry.httpMetadata,
      customMetadata: entry.customMetadata,
      writeHttpMetadata(headers) {
        if (entry.httpMetadata?.contentType) headers.set("Content-Type", entry.httpMetadata.contentType);
      },
    };
  }

  async head(key) {
    if (this.headPause && key.startsWith("comments-rate/")) {
      const pause = this.headPause;
      this.headPause = null;
      pause.signalEntered();
      await pause.released;
    }
    const entry = this.objects.get(key);
    return entry ? this.metadata(key, entry) : null;
  }

  async get(key) {
    const entry = this.objects.get(key);
    if (!entry) return null;
    const bytes = entry.bytes.slice();
    return {
      ...this.metadata(key, entry),
      body: new Response(bytes).body,
      async text() { return new TextDecoder().decode(bytes); },
      async arrayBuffer() { return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength); },
    };
  }

  async put(key, value, options = {}) {
    const bytes = await bodyBytes(value);
    await Promise.resolve();
    const existing = this.objects.get(key);
    const onlyIf = options.onlyIf;
    const createOnly = onlyIf instanceof Headers && onlyIf.get("If-None-Match") === "*";
    const etagMismatch = onlyIf?.etagMatches && existing?.etag !== onlyIf.etagMatches;
    if ((createOnly && existing) || etagMismatch) {
      if (this.throwOnConflict) throw new Error("TooManyRequests: same-key R2 write throttled (10058)");
      return null;
    }
    const entry = {
      bytes,
      etag: `etag-${++this.version}`,
      uploaded: new Date(),
      httpMetadata: options.httpMetadata ?? {},
      customMetadata: options.customMetadata ?? {},
    };
    this.objects.set(key, entry);
    return this.metadata(key, entry);
  }

  async list(options = {}) {
    const keys = [...this.objects.keys()].filter((key) => key.startsWith(options.prefix ?? "")).sort();
    const start = Number(options.cursor ?? 0);
    const page = keys.slice(start, start + (options.limit ?? 1_000));
    const next = start + page.length;
    return {
      objects: page.map((key) => this.metadata(key, this.objects.get(key))),
      truncated: next < keys.length,
      cursor: next < keys.length ? String(next) : undefined,
    };
  }

  async delete(key) {
    this.objects.delete(key);
  }
}

function environment(bucket = new CommentBucket()) {
  return { BUCKET: bucket, INGEST_TOKEN: TOKEN };
}

function request(path, init = {}) {
  return new Request(`${ORIGIN}${path}`, init);
}

function commentRequest(payload, { ip = "203.0.113.7", origin = ORIGIN, contentType = "application/json", headers = {} } = {}) {
  const body = typeof payload === "string" ? payload : JSON.stringify(payload);
  return request("/api/comments", {
    method: "POST",
    headers: {
      Origin: origin,
      "CF-Connecting-IP": ip,
      "Content-Type": contentType,
      ...headers,
    },
    body,
  });
}

async function post(environmentValue, payload, options = {}) {
  return handleRequest(commentRequest(payload, options), environmentValue);
}

describe("public comments API", () => {
  it("persists and lists plain comment data without turning markup into HTML", async () => {
    const bucket = new CommentBucket();
    const writer = environment(bucket);
    const response = await post(writer, { displayName: "<b>Alice</b>", body: "<img src=x onerror=alert(1)>", website: "" });
    assert.equal(response.status, 201, await response.clone().text());
    assert.equal(response.headers.get("X-Content-Type-Options"), "nosniff");
    const created = (await response.json()).comment;
    assert.deepEqual(Object.keys(created).sort(), ["body", "createdAt", "displayName", "id"]);

    const reader = environment(bucket);
    const listing = await handleRequest(request("/api/comments"), reader);
    assert.equal(listing.status, 200);
    assert.match(listing.headers.get("Content-Type"), /^application\/json/);
    const payload = await listing.json();
    assert.deepEqual(payload.comments, [created]);
    assert.equal(payload.comments[0].body, "<img src=x onerror=alert(1)>");
    assert.equal(payload.cursor, null);
    assert.equal([...bucket.objects.keys()].some((key) => key.startsWith("comments/")), true);
  });

  it("enforces origin, JSON, honeypot, schema, character, byte, and caller checks", async () => {
    const cases = [
      [commentRequest({ displayName: "A", body: "B" }, { origin: "https://evil.test" }), 403],
      [commentRequest({ displayName: "A", body: "B" }, { origin: "" }), 403],
      [commentRequest({ displayName: "A", body: "B" }, { contentType: "text/plain" }), 415],
      [commentRequest("{", {}), 400],
      [commentRequest({ displayName: "A", body: "B", website: "spam.test" }), 400],
      [commentRequest({ displayName: "A", body: "B", verified: true }), 400],
      [commentRequest({ displayName: "x".repeat(41), body: "B" }), 400],
      [commentRequest({ displayName: "A", body: "x".repeat(2_001) }), 400],
      [commentRequest({ displayName: "A", body: "B" }, { ip: "" }), 403],
      [commentRequest({ displayName: "A", body: "B" }, { headers: { "Content-Length": "20000" } }), 413],
    ];
    for (const [input, expected] of cases) {
      const response = await handleRequest(input, environment());
      assert.equal(response.status, expected, await response.text());
    }
  });

  it("atomically admits only three concurrent comments per caller and anonymizes rate keys", async () => {
    const bucket = new CommentBucket({ throwOnConflict: true });
    const runtime = environment(bucket);
    const caller = "2001:db8::42";
    const responses = await Promise.all(Array.from({ length: 12 }, (_, index) => post(runtime, {
      displayName: `Caller ${index}`,
      body: `Concurrent comment ${index}`,
    }, { ip: caller })));
    const statuses = responses.map((response) => response.status);
    assert.equal(statuses.filter((status) => status === 201).length, 3);
    assert.equal(statuses.filter((status) => status === 429).length, 9);
    assert.equal(statuses.some((status) => status === 500), false);
    for (const response of responses.filter((candidate) => candidate.status === 429)) {
      assert.ok(Number(response.headers.get("Retry-After")) >= 1);
    }
    const commentKeys = [...bucket.objects.keys()].filter((key) => key.startsWith("comments/"));
    const rateKeys = [...bucket.objects.keys()].filter((key) => key.startsWith("comments-rate/"));
    assert.equal(commentKeys.length, 3);
    assert.equal(rateKeys.length, 3);
    assert.equal([...bucket.objects.entries()].some(([key, entry]) => key.includes(caller) || new TextDecoder().decode(entry.bytes).includes(caller)), false);
    assert.equal((await post(runtime, { displayName: "Other", body: "Independent caller" }, { ip: "198.51.100.11" })).status, 201);
  });

  it("never lets a stale-window request overwrite newer rate slots", async () => {
    const realDateNow = Date.now;
    const bucket = new CommentBucket();
    const runtime = environment(bucket);
    const caller = "203.0.113.99";
    const windowMs = 10 * 60 * 1_000;
    const newerWindow = Math.floor(realDateNow() / windowMs) * windowMs;
    const staleWindow = newerWindow - windowMs;
    const gate = bucket.pauseNextRateHead();
    let staleRequest;
    try {
      Date.now = () => staleWindow + windowMs - 1;
      staleRequest = post(runtime, { displayName: "Stale", body: "Old window" }, { ip: caller });
      await gate.entered;

      Date.now = () => newerWindow + 1;
      const newerStatuses = [];
      for (let index = 0; index < 3; index += 1) {
        newerStatuses.push((await post(runtime, { displayName: `New ${index}`, body: `New window ${index}` }, { ip: caller })).status);
      }
      assert.deepEqual(newerStatuses, [201, 201, 201]);
      const beforeStaleResumes = [...bucket.objects.entries()]
        .filter(([key]) => key.startsWith("comments-rate/"))
        .map(([key, entry]) => [key, entry.customMetadata.windowStart])
        .sort();
      assert.equal(beforeStaleResumes.length, 3);
      assert.equal(beforeStaleResumes.every(([, epoch]) => epoch === String(newerWindow)), true);

      gate.release();
      assert.equal((await staleRequest).status, 429);
      assert.equal((await post(runtime, { displayName: "New 4", body: "Must remain limited" }, { ip: caller })).status, 429);
      const afterStaleResumes = [...bucket.objects.entries()]
        .filter(([key]) => key.startsWith("comments-rate/"))
        .map(([key, entry]) => [key, entry.customMetadata.windowStart])
        .sort();
      assert.deepEqual(afterStaleResumes, beforeStaleResumes, "slot epochs must never decrease");
      assert.equal((await (await handleRequest(request("/api/comments"), runtime)).json()).comments.length, 3);
    } finally {
      gate.release();
      Date.now = realDateNow;
      await staleRequest?.catch(() => {});
    }
  });

  it("fails closed for rate slots with malformed legacy epochs", async () => {
    const realDateNow = Date.now;
    const bucket = new CommentBucket();
    const runtime = environment(bucket);
    const caller = "203.0.113.100";
    const windowMs = 10 * 60 * 1_000;
    const windowStart = Math.floor(realDateNow() / windowMs) * windowMs;
    try {
      Date.now = () => windowStart + 1;
      for (let index = 0; index < 3; index += 1) {
        assert.equal((await post(runtime, { displayName: `Legacy ${index}`, body: "Claim slot" }, { ip: caller })).status, 201);
      }
      for (const [key, entry] of bucket.objects) {
        if (key.startsWith("comments-rate/")) entry.customMetadata.windowStart = "malformed";
      }
      Date.now = () => windowStart + windowMs + 1;
      assert.equal((await post(runtime, { displayName: "Later", body: "Must fail closed" }, { ip: caller })).status, 429);
      assert.equal([...bucket.objects.keys()].filter((key) => key.startsWith("comments/")).length, 3);
    } finally {
      Date.now = realDateNow;
    }
  });

  it("lists newest first with bounded cursor pagination", async () => {
    const runtime = environment();
    for (let index = 0; index < 26; index += 1) {
      const response = await post(runtime, { displayName: `Person ${index}`, body: `Comment ${index}` }, { ip: `198.51.100.${index + 1}` });
      assert.equal(response.status, 201, await response.text());
      await new Promise((resolve) => setTimeout(resolve, 2));
    }
    const firstResponse = await handleRequest(request("/api/comments"), runtime);
    const first = await firstResponse.json();
    assert.equal(first.comments.length, 25);
    assert.equal(typeof first.cursor, "string");
    for (let index = 1; index < first.comments.length; index += 1) {
      assert.ok(first.comments[index - 1].createdAt >= first.comments[index].createdAt);
    }
    const secondResponse = await handleRequest(request(`/api/comments?cursor=${encodeURIComponent(first.cursor)}`), runtime);
    const second = await secondResponse.json();
    assert.equal(second.comments.length, 1);
    assert.equal(second.cursor, null);
    assert.equal(new Set([...first.comments, ...second.comments].map((comment) => comment.id)).size, 26);
  });

  it("requires moderator authentication for deletion and leaves existing routes intact", async () => {
    const runtime = environment();
    const createdResponse = await post(runtime, { displayName: "Temporary", body: "Remove me" });
    const { comment } = await createdResponse.json();
    const path = `/api/comments/${encodeURIComponent(comment.id)}`;

    assert.equal((await handleRequest(request(path, { method: "DELETE" }), runtime)).status, 401);
    assert.equal((await handleRequest(request(path, { method: "DELETE", headers: { Authorization: "Bearer wrong" } }), runtime)).status, 401);
    const removed = await handleRequest(request(path, { method: "DELETE", headers: { Authorization: `Bearer ${TOKEN}` } }), runtime);
    assert.equal(removed.status, 200);
    assert.deepEqual(await removed.json(), { deleted: true, id: comment.id });
    assert.equal((await handleRequest(request(path, { method: "DELETE", headers: { Authorization: `Bearer ${TOKEN}` } }), runtime)).status, 404);
    assert.deepEqual((await (await handleRequest(request("/api/comments"), runtime)).json()).comments, []);

    const live = await handleRequest(request("/api/live"), runtime);
    assert.equal(live.status, 200);
    assert.equal((await live.json()).frame, null);
  });
});
