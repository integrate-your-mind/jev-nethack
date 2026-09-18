import assert from "node:assert/strict";
import { beforeEach, describe, it } from "node:test";

import { handleRequest } from "./worker.mjs";

const TOKEN = "test-ingest-token-with-enough-entropy";
const encoder = new TextEncoder();
const VALID_MP4_BASE64 = "AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAMVbW9vdgAAAGxtdmhkAAAAAAAAAAAAAAAAAAAD6AAAA+gAAQAAAQAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAj90cmFrAAAAXHRraGQAAAADAAAAAAAAAAAAAAABAAAAAAAAA+gAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAAABAAAAAQAAAAAAAkZWR0cwAAABxlbHN0AAAAAAAAAAEAAAPoAAAAAAABAAAAAAG3bWRpYQAAACBtZGhkAAAAAAAAAAAAAAAAAABAAAAAQABVxAAAAAAALWhkbHIAAAAAAAAAAHZpZGUAAAAAAAAAAAAAAABWaWRlb0hhbmRsZXIAAAABYm1pbmYAAAAUdm1oZAAAAAEAAAAAAAAAAAAAACRkaW5mAAAAHGRyZWYAAAAAAAAAAQAAAAx1cmwgAAAAAQAAASJzdGJsAAAAvnN0c2QAAAAAAAAAAQAAAK5hdmMxAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAABAAEABIAAAASAAAAAAAAAABFUxhdmM2Mi4yOC4xMDEgbGlieDI2NAAAAAAAAAAAAAAAGP//AAAANGF2Y0MBZAAK/+EAF2dkAAqs2V7ARAAAAwAEAAADAAg8SJZYAQAGaOvjyyLA/fj4AAAAABBwYXNwAAAAAQAAAAEAAAAUYnRydAAAAAAAABYoAAAAAAAAABhzdHRzAAAAAAAAAAEAAAABAABAAAAAABxzdHNjAAAAAAAAAAEAAAABAAAAAQAAAAEAAAAUc3RzegAAAAAAAALFAAAAAQAAABRzdGNvAAAAAAAAAAEAAANFAAAAYnVkdGEAAABabWV0YQAAAAAAAAAhaGRscgAAAAAAAAAAbWRpcmFwcGwAAAAAAAAAAAAAAAAtaWxzdAAAACWpdG9vAAAAHWRhdGEAAAABAAAAAExhdmY2Mi4xMi4xMDEAAAAIZnJlZQAAAs1tZGF0AAACrQYF//+p3EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDE2NSByMzIyMiBiMzU2MDVhIC0gSC4yNjQvTVBFRy00IEFWQyBjb2RlYyAtIENvcHlsZWZ0IDIwMDMtMjAyNSAtIGh0dHA6Ly93d3cudmlkZW9sYW4ub3JnL3gyNjQuaHRtbCAtIG9wdGlvbnM6IGNhYmFjPTEgcmVmPTMgZGVibG9jaz0xOjA6MCBhbmFseXNlPTB4MzoweDExMyBtZT1oZXggc3VibWU9NyBwc3k9MSBwc3lfcmQ9MS4wMDowLjAwIG1peGVkX3JlZj0xIG1lX3JhbmdlPTE2IGNocm9tYV9tZT0xIHRyZWxsaXM9MSA4eDhkY3Q9MSBjcW09MCBkZWFkem9uZT0yMSwxMSBmYXN0X3Bza2lwPTEgY2hyb21hX3FwX29mZnNldD0tMiB0aHJlYWRzPTEgbG9va2FoZWFkX3RocmVhZHM9MSBzbGljZWRfdGhyZWFkcz0wIG5yPTAgZGVjaW1hdGU9MSBpbnRlcmxhY2VkPTAgYmx1cmF5X2NvbXBhdD0wIGNvbnN0cmFpbmVkX2ludHJhPTAgYmZyYW1lcz0zIGJfcHlyYW1pZD0yIGJfYWRhcHQ9MSBiX2JpYXM9MCBkaXJlY3Q9MSB3ZWlnaHRiPTEgb3Blbl9nb3A9MCB3ZWlnaHRwPTIga2V5aW50PTI1MCBrZXlpbnRfbWluPTEgc2NlbmVjdXQ9NDAgaW50cmFfcmVmcmVzaD0wIHJjX2xvb2thaGVhZD00MCByYz1jcmYgbWJ0cmVlPTEgY3JmPTIzLjAgcWNvbXA9MC42MCBxcG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00IGlwX3JhdGlvPTEuNDAgYXE9MToxLjAwAIAAAAAQZYiEABX//vfJ78Cm69vfgQ==";

function decodeBase64(value) {
  return Uint8Array.from(Buffer.from(value, "base64"));
}

function zeroTrackMp4() {
  const bytes = new Uint8Array(262);
  const view = new DataView(bytes.buffer);
  const writeBox = (offset, size, type) => {
    view.setUint32(offset, size);
    bytes.set(encoder.encode(type), offset + 4);
  };
  writeBox(0, 24, "ftyp");
  bytes.set(encoder.encode("isom"), 8);
  writeBox(24, 8, "moov");
  writeBox(32, 214, "free");
  writeBox(246, 16, "mdat");
  bytes.fill(1, 254);
  return bytes;
}

function toHex(buffer) {
  return Array.from(new Uint8Array(buffer), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function sha256(value) {
  const bytes = typeof value === "string" ? encoder.encode(value) : value;
  return toHex(await crypto.subtle.digest("SHA-256", bytes));
}

async function bodyBytes(value) {
  if (typeof value === "string") return encoder.encode(value);
  if (value instanceof Uint8Array) return value;
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  if (ArrayBuffer.isView(value)) return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  if (value instanceof Blob) return new Uint8Array(await value.arrayBuffer());
  if (value instanceof ReadableStream) return new Uint8Array(await new Response(value).arrayBuffer());
  throw new TypeError("unsupported mock body");
}

function parseRange(header, size) {
  const match = /^bytes=(\d+)-(\d*)$/.exec(header ?? "");
  if (!match) throw new Error("invalid range");
  const start = Number(match[1]);
  const requestedEnd = match[2] ? Number(match[2]) : size - 1;
  if (start >= size || requestedEnd < start) throw new Error("invalid range");
  const end = Math.min(requestedEnd, size - 1);
  return { offset: start, length: end - start + 1 };
}

class MockBucket {
  constructor() {
    this.objects = new Map();
    this.version = 0;
  }

  async head(key) {
    const entry = this.objects.get(key);
    return entry ? this.metadata(key, entry) : null;
  }

  metadata(key, entry, range = undefined) {
    return {
      key,
      size: entry.bytes.byteLength,
      etag: entry.etag,
      httpEtag: `"${entry.etag}"`,
      uploaded: entry.uploaded,
      httpMetadata: entry.httpMetadata,
      customMetadata: entry.customMetadata,
      range,
      writeHttpMetadata(headers) {
        if (entry.httpMetadata?.contentType) headers.set("Content-Type", entry.httpMetadata.contentType);
        if (entry.httpMetadata?.contentDisposition) headers.set("Content-Disposition", entry.httpMetadata.contentDisposition);
      },
    };
  }

  async get(key, options = undefined) {
    const entry = this.objects.get(key);
    if (!entry) return null;
    let bytes = entry.bytes;
    let range;
    if (options?.range instanceof Headers) {
      range = parseRange(options.range.get("Range"), bytes.byteLength);
      bytes = bytes.slice(range.offset, range.offset + range.length);
    } else if (options?.range) {
      range = options.range;
      bytes = bytes.slice(range.offset, range.offset + range.length);
    }
    const metadata = this.metadata(key, entry, range);
    return {
      ...metadata,
      body: new Response(bytes).body,
      async text() { return new TextDecoder().decode(bytes); },
      async arrayBuffer() { return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength); },
    };
  }

  async put(key, value, options = undefined) {
    const existing = this.objects.get(key);
    const onlyIf = options?.onlyIf;
    if (onlyIf instanceof Headers && onlyIf.get("If-None-Match") === "*" && existing) return null;
    if (onlyIf?.etagMatches && (!existing || existing.etag !== onlyIf.etagMatches)) return null;
    const bytes = await bodyBytes(value);
    if (options?.sha256 && (await sha256(bytes)) !== options.sha256) throw new Error("SHA256 checksum mismatch");
    const etag = `etag-${++this.version}`;
    const entry = {
      bytes,
      etag,
      uploaded: new Date(),
      httpMetadata: options?.httpMetadata ?? {},
      customMetadata: options?.customMetadata ?? {},
    };
    this.objects.set(key, entry);
    return this.metadata(key, entry);
  }

  async list(options = {}) {
    const keys = [...this.objects.keys()].filter((key) => key.startsWith(options.prefix ?? "")).sort();
    const start = options.cursor ? Number(options.cursor) : 0;
    const limit = options.limit ?? 1000;
    const page = keys.slice(start, start + limit);
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

function env(bucket = new MockBucket()) {
  return { BUCKET: bucket, INGEST_TOKEN: TOKEN };
}

function authHeaders(extra = undefined) {
  return { Authorization: `Bearer ${TOKEN}`, ...extra };
}

function request(path, init = undefined) {
  return new Request(`https://example.test${path}`, init);
}

function liveFrame(overrides = undefined) {
  return {
    schemaVersion: 1,
    sessionId: "broadcast-1-seg0001",
    broadcastId: "broadcast-1",
    sequence: 1,
    capturedAt: new Date().toISOString(),
    state: { player: { hp: 12, score: 3 }, terminal: "Jev the Candidate" },
    score: 3,
    isAscended: false,
    ...overrides,
  };
}

async function upload(environment, sessionId, filename, bytes, checksum = undefined) {
  const digest = checksum ?? await sha256(bytes);
  return handleRequest(request(`/api/recordings/${sessionId}/${filename}`, {
    method: "PUT",
    headers: authHeaders({
      "Content-Length": String(bytes.byteLength),
      "X-Content-SHA256": digest,
    }),
    body: bytes,
    duplex: "half",
  }), environment);
}

describe("live API", () => {
  let environment;

  beforeEach(() => { environment = env(); });

  it("returns offline before the first real frame and accepts an authenticated frame", async () => {
    const empty = await handleRequest(request("/api/live"), environment);
    assert.equal(empty.status, 200);
    assert.deepEqual((await empty.json()).frame, null);

    const posted = await handleRequest(request("/api/live", {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(liveFrame()),
    }), environment);
    assert.equal(posted.status, 201);

    const response = await handleRequest(request("/api/live"), environment);
    const payload = await response.json();
    assert.equal(payload.live, true);
    assert.equal(payload.freshnessWindowMs, 10_000);
    assert.equal(payload.frame.state.player.score, 3);
    assert.equal(typeof payload.serverTime, "string");
  });

  it("requires bearer auth and does not echo secrets", async () => {
    const response = await handleRequest(request("/api/live", {
      method: "POST",
      headers: { Authorization: "Bearer wrong-token" },
      body: JSON.stringify(liveFrame()),
    }), environment);
    assert.equal(response.status, 401);
    assert.equal((await response.text()).includes(TOKEN), false);
  });

  it("rejects oversized JSON and stale sequence numbers", async () => {
    const oversized = JSON.stringify(liveFrame({ state: { terminal: "x".repeat(300_000) } }));
    const large = await handleRequest(request("/api/live", {
      method: "POST",
      headers: authHeaders(),
      body: oversized,
    }), environment);
    assert.equal(large.status, 413);

    const frame = liveFrame();
    const first = await handleRequest(request("/api/live", { method: "POST", headers: authHeaders(), body: JSON.stringify(frame) }), environment);
    assert.equal(first.status, 201);
    const duplicate = await handleRequest(request("/api/live", { method: "POST", headers: authHeaders(), body: JSON.stringify(frame) }), environment);
    assert.equal(duplicate.status, 409);
  });

  it("marks ended, delayed, and far-future frames offline", async () => {
    for (const [index, frame] of [
      liveFrame({ sequence: 1, ended: true }),
      liveFrame({ sessionId: "broadcast-1-seg0002", sequence: 1, capturedAt: new Date(Date.now() - 20_000).toISOString() }),
      liveFrame({ sessionId: "broadcast-1-seg0003", sequence: 1, capturedAt: new Date(Date.now() + 60_000).toISOString() }),
    ].entries()) {
      const posted = await handleRequest(request("/api/live", { method: "POST", headers: authHeaders(), body: JSON.stringify(frame) }), environment);
      assert.equal(posted.status, 201, `frame ${index}`);
      const status = await handleRequest(request("/api/live"), environment);
      assert.equal((await status.json()).live, false, `frame ${index}`);
    }
  });
});

describe("recording API", () => {
  let environment;

  beforeEach(() => { environment = env(); });

  it("streams immutable checksum-validated uploads and MP4 ranges", async () => {
    const bytes = encoder.encode("0123456789");
    const created = await upload(environment, "broadcast-1-seg0001", "recording.mp4", bytes);
    assert.equal(created.status, 201);
    const duplicate = await upload(environment, "broadcast-1-seg0001", "recording.mp4", bytes);
    assert.equal(duplicate.status, 409);

    const partial = await handleRequest(request("/api/recordings/broadcast-1-seg0001/recording.mp4", {
      headers: { Range: "bytes=2-5" },
    }), environment);
    assert.equal(partial.status, 206);
    assert.equal(partial.headers.get("Content-Range"), "bytes 2-5/10");
    assert.equal(await partial.text(), "2345");

    const suffix = await handleRequest(request("/api/recordings/broadcast-1-seg0001/recording.mp4", {
      headers: { Range: "bytes=-3" },
    }), environment);
    assert.equal(suffix.status, 206);
    assert.equal(suffix.headers.get("Content-Range"), "bytes 7-9/10");
    assert.equal(await suffix.text(), "789");

    const invalidRange = await handleRequest(request("/api/recordings/broadcast-1-seg0001/recording.mp4", {
      headers: { Range: "bytes=99-100" },
    }), environment);
    assert.equal(invalidRange.status, 416);
  });

  it("rejects bad checksums, missing lengths, oversized declarations, and all deletes", async () => {
    const bytes = encoder.encode("video");
    const unauthorized = await handleRequest(request("/api/recordings/broadcast-1-seg0001/recording.mp4", {
      method: "PUT",
      headers: { "Content-Length": String(bytes.byteLength), "X-Content-SHA256": await sha256(bytes) },
      body: bytes,
      duplex: "half",
    }), environment);
    assert.equal(unauthorized.status, 401);

    const bad = await upload(environment, "broadcast-1-seg0001", "recording.mp4", bytes, "0".repeat(64));
    assert.equal(bad.status, 422);

    const noLength = await handleRequest(request("/api/recordings/broadcast-1-seg0001/events.jsonl", {
      method: "PUT",
      headers: authHeaders({ "X-Content-SHA256": await sha256(bytes) }),
      body: bytes,
      duplex: "half",
    }), environment);
    assert.equal(noLength.status, 411);

    const huge = await handleRequest(request("/api/recordings/broadcast-1-seg0001/events.jsonl", {
      method: "PUT",
      headers: authHeaders({ "Content-Length": String(32 * 1024 * 1024 + 1), "X-Content-SHA256": await sha256(bytes) }),
      body: bytes,
      duplex: "half",
    }), environment);
    assert.equal(huge.status, 413);

    const wrongLength = await handleRequest(request("/api/recordings/broadcast-1-seg0001/events.jsonl", {
      method: "PUT",
      headers: authHeaders({ "Content-Length": String(bytes.byteLength - 1), "X-Content-SHA256": await sha256(bytes) }),
      body: bytes,
      duplex: "half",
    }), environment);
    assert.equal(wrongLength.status, 400);
    assert.equal(await environment.BUCKET.head("recordings/broadcast-1-seg0001/events.jsonl"), null);

    const deletion = await handleRequest(request("/api/recordings/broadcast-1-seg0001/events.jsonl", { method: "DELETE" }), environment);
    assert.equal(deletion.status, 405);

  });

  it("publishes only completed manifests whose JSONL and MP4 metadata match", async () => {
    const sessionId = "broadcast-1-seg0001";
    const events = encoder.encode('{"frame":1}\n');
    const video = decodeBase64(VALID_MP4_BASE64);
    const eventsHash = await sha256(events);
    const videoHash = await sha256(video);
    assert.equal((await upload(environment, sessionId, "events.jsonl", events)).status, 201);
    assert.equal((await upload(environment, sessionId, "recording.mp4", video)).status, 201);
    const manifest = {
      schemaVersion: 1,
      sessionId,
      broadcastId: "broadcast-1",
      completed: true,
      startedAt: "2026-09-17T20:00:00.000Z",
      endedAt: "2026-09-17T20:01:00.000Z",
      frameCount: 120,
      actionCount: 60,
      artifacts: [
        { filename: "events.jsonl", sha256: eventsHash, bytes: events.byteLength, contentType: "application/x-ndjson" },
        { filename: "recording.mp4", sha256: videoHash, bytes: video.byteLength, contentType: "video/mp4" },
      ],
    };
    const manifestBytes = encoder.encode(JSON.stringify(manifest));
    assert.equal((await upload(environment, sessionId, "manifest.json", manifestBytes)).status, 201);

    const archive = await handleRequest(request("/api/archive"), environment);
    const listing = await archive.json();
    assert.equal(listing.recordings.length, 1);
    assert.equal(listing.recordings[0].sessionId, sessionId);
    assert.equal(listing.recordings[0].artifacts[1].url, `/api/recordings/${sessionId}/recording.mp4`);

    const manifestResponse = await handleRequest(request(`/api/recordings/${sessionId}/manifest.json`), environment);
    assert.equal(manifestResponse.status, 200);
    assert.equal((await manifestResponse.json()).completed, true);
  });

  it("rejects a zero-track MP4 manifest and filters a preserved invalid manifest from archive", async () => {
    const sessionId = "broadcast-1-seg0002";
    const events = encoder.encode('{"frame":1}\n');
    const video = zeroTrackMp4();
    const eventsHash = await sha256(events);
    const videoHash = await sha256(video);
    assert.equal(video.byteLength, 262);
    assert.equal((await upload(environment, sessionId, "events.jsonl", events)).status, 201);
    assert.equal((await upload(environment, sessionId, "recording.mp4", video)).status, 201);
    const manifest = {
      schemaVersion: 1,
      sessionId,
      broadcastId: "broadcast-1",
      completed: true,
      startedAt: "2026-09-17T20:01:00.000Z",
      endedAt: "2026-09-17T20:02:00.000Z",
      frameCount: 1,
      actionCount: 0,
      artifacts: [
        { filename: "events.jsonl", sha256: eventsHash, bytes: events.byteLength, contentType: "application/x-ndjson" },
        { filename: "recording.mp4", sha256: videoHash, bytes: video.byteLength, contentType: "video/mp4" },
      ],
    };
    const manifestBytes = encoder.encode(JSON.stringify(manifest));
    const rejected = await upload(environment, sessionId, "manifest.json", manifestBytes);
    assert.equal(rejected.status, 422);
    assert.equal((await rejected.json()).error.code, "invalid_mp4");
    assert.equal(await environment.BUCKET.head(`manifests/${sessionId}.json`), null);

    const manifestHash = await sha256(manifestBytes);
    await environment.BUCKET.put(`manifests/${sessionId}.json`, manifestBytes, {
      sha256: manifestHash,
      httpMetadata: { contentType: "application/json" },
      customMetadata: { sessionId, kind: "manifest", sha256: manifestHash },
    });
    const archive = await handleRequest(request("/api/archive"), environment);
    const listing = await archive.json();
    assert.equal(listing.recordings.length, 0);
    assert.equal(listing.invalidManifestCount, 1);
    assert.notEqual(await environment.BUCKET.head(`recordings/${sessionId}/recording.mp4`), null);
    assert.notEqual(await environment.BUCKET.head(`manifests/${sessionId}.json`), null);
  });

  it("rejects a completed manifest before referenced artifacts exist", async () => {
    const manifest = {
      schemaVersion: 1,
      sessionId: "broadcast-2-seg0001",
      completed: true,
      startedAt: "2026-09-17T20:00:00.000Z",
      endedAt: "2026-09-17T20:01:00.000Z",
      frameCount: 1,
      actionCount: 1,
      artifacts: [
        { filename: "events.jsonl", sha256: "1".repeat(64), bytes: 10, contentType: "application/x-ndjson" },
        { filename: "recording.mp4", sha256: "2".repeat(64), bytes: 10, contentType: "video/mp4" },
      ],
    };
    const bytes = encoder.encode(JSON.stringify(manifest));
    const response = await upload(environment, manifest.sessionId, "manifest.json", bytes);
    assert.equal(response.status, 409);
  });
});
