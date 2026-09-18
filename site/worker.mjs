const INDEX_HTML = "__JEV_SITE_INDEX_HTML__";

const LIVE_KEY = "live/latest.json";
const LIVE_BODY_LIMIT = 256 * 1024;
const MANIFEST_BODY_LIMIT = 256 * 1024;
const RECORDING_BODY_LIMIT = 32 * 1024 * 1024;
const MP4_PROBE_LIMIT = 2 * 1024 * 1024;
const LIVE_FRESHNESS_MS = 10_000;
const MAX_FUTURE_SKEW_MS = 30_000;
const MAX_ARCHIVE_PAGE = 50;
const SESSION_RE = /^[a-z0-9][a-z0-9_-]{0,79}$/;
const FILE_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const SHA256_RE = /^[a-f0-9]{64}$/;

const TRAINING_BODY_LIMIT = 32 * 1024 * 1024;
const TRAINING_PAGE_LIMIT = 50;
const TRAINING_MAGIC = new TextEncoder().encode("JEVNHNPZ1\n");
const TRAINING_SHARD_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$/;
const TRAINING_FILE_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;

const COMMENT_BODY_LIMIT = 16 * 1024;
const COMMENT_PAGE_LIMIT = 25;
const COMMENT_NAME_CHAR_LIMIT = 40;
const COMMENT_BODY_CHAR_LIMIT = 2_000;
const COMMENT_RATE_LIMIT = 3;
const COMMENT_RATE_WINDOW_MS = 10 * 60 * 1_000;
const COMMENT_ID_RE = /^\d{13}-[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const UNSAFE_PLAIN_TEXT_RE = /[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/;

class HttpError extends Error {
  constructor(status, code, message, headers = undefined) {
    super(message);
    this.status = status;
    this.code = code;
    this.headers = headers;
  }
}

function json(data, status = 200, extraHeaders = undefined) {
  const headers = new Headers(extraHeaders);
  headers.set("Content-Type", "application/json; charset=utf-8");
  headers.set("X-Content-Type-Options", "nosniff");
  return new Response(JSON.stringify(data), { status, headers });
}

function fail(status, code, message, headers = undefined) {
  return json({ error: { code, message } }, status, headers);
}

function isObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function validDate(value) {
  return typeof value === "string" && Number.isFinite(Date.parse(value));
}

function hex(bytes) {
  return Array.from(new Uint8Array(bytes), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function sha256Hex(value) {
  return hex(await crypto.subtle.digest("SHA-256", value));
}

async function secretsEqual(provided, expected) {
  const encoder = new TextEncoder();
  const [providedHash, expectedHash] = await Promise.all([
    crypto.subtle.digest("SHA-256", encoder.encode(provided)),
    crypto.subtle.digest("SHA-256", encoder.encode(expected)),
  ]);
  if (typeof crypto.subtle.timingSafeEqual === "function") {
    return crypto.subtle.timingSafeEqual(providedHash, expectedHash);
  }
  const left = new Uint8Array(providedHash);
  const right = new Uint8Array(expectedHash);
  let difference = 0;
  for (let index = 0; index < left.length; index += 1) {
    difference |= left[index] ^ right[index];
  }
  return difference === 0;
}

async function requireAuth(request, env) {
  const expected = typeof env.INGEST_TOKEN === "string" ? env.INGEST_TOKEN : "";
  const authorization = request.headers.get("Authorization") ?? "";
  const match = /^Bearer ([^\s]+)$/.exec(authorization);
  const supplied = match?.[1] ?? "";
  if (!expected || !supplied || !(await secretsEqual(supplied, expected))) {
    throw new HttpError(401, "unauthorized", "A valid bearer token is required.", {
      "WWW-Authenticate": "Bearer",
    });
  }
}

async function readBounded(request, limit) {
  const declared = request.headers.get("Content-Length");
  if (declared !== null) {
    const length = Number(declared);
    if (!Number.isSafeInteger(length) || length < 0) {
      throw new HttpError(400, "invalid_content_length", "Content-Length must be a non-negative integer.");
    }
    if (length > limit) {
      throw new HttpError(413, "payload_too_large", `Payload exceeds the ${limit}-byte limit.`);
    }
  }
  if (!request.body) {
    throw new HttpError(400, "missing_body", "A request body is required.");
  }
  const reader = request.body.getReader();
  const chunks = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > limit) {
      await reader.cancel("payload limit exceeded");
      throw new HttpError(413, "payload_too_large", `Payload exceeds the ${limit}-byte limit.`);
    }
    chunks.push(value);
  }
  const body = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    body.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return body;
}

async function readJson(request, limit) {
  const bytes = await readBounded(request, limit);
  try {
    return JSON.parse(new TextDecoder().decode(bytes));
  } catch {
    throw new HttpError(400, "invalid_json", "The request body must be valid JSON.");
  }
}

function assertJsonContentType(request) {
  const mediaType = (request.headers.get("Content-Type") ?? "").split(";", 1)[0].trim().toLowerCase();
  if (mediaType !== "application/json") {
    throw new HttpError(415, "unsupported_media_type", "Content-Type must be application/json.");
  }
}

function requireSameOrigin(request) {
  const requestOrigin = new URL(request.url).origin;
  const suppliedOrigin = request.headers.get("Origin");
  let normalizedOrigin;
  try {
    normalizedOrigin = suppliedOrigin ? new URL(suppliedOrigin).origin : "";
  } catch {
    normalizedOrigin = "";
  }
  if (!suppliedOrigin || normalizedOrigin !== suppliedOrigin || normalizedOrigin !== requestOrigin) {
    throw new HttpError(403, "invalid_origin", "Comment submissions require a same-origin request.");
  }
}

function commentCharacterCount(value) {
  return [...value].length;
}

function validateCommentInput(input) {
  if (!isObject(input)) {
    throw new HttpError(400, "invalid_comment", "The comment body must be a JSON object.");
  }
  const allowedFields = new Set(["displayName", "body", "website"]);
  if (Object.keys(input).some((field) => !allowedFields.has(field))) {
    throw new HttpError(400, "invalid_comment", "The comment contains an unsupported field.");
  }
  if (input.website !== undefined && typeof input.website !== "string") {
    throw new HttpError(400, "invalid_honeypot", "website must be a string when supplied.");
  }
  if ((input.website ?? "").trim()) {
    throw new HttpError(400, "honeypot_triggered", "Comment rejected.");
  }
  if (typeof input.displayName !== "string" || typeof input.body !== "string") {
    throw new HttpError(400, "invalid_comment", "displayName and body must be strings.");
  }
  const displayName = input.displayName.trim();
  const body = input.body.trim();
  if (!displayName || commentCharacterCount(displayName) > COMMENT_NAME_CHAR_LIMIT || UNSAFE_PLAIN_TEXT_RE.test(displayName)) {
    throw new HttpError(400, "invalid_display_name", `displayName must be 1 to ${COMMENT_NAME_CHAR_LIMIT} plain-text characters.`);
  }
  if (!body || commentCharacterCount(body) > COMMENT_BODY_CHAR_LIMIT || UNSAFE_PLAIN_TEXT_RE.test(body)) {
    throw new HttpError(400, "invalid_comment_body", `body must be 1 to ${COMMENT_BODY_CHAR_LIMIT} plain-text characters.`);
  }
  return { displayName, body };
}

function parseCommentPath(pathname) {
  const match = /^\/api\/comments\/([^/]+)$/.exec(pathname);
  if (!match) return null;
  let id;
  try {
    id = decodeURIComponent(match[1]);
  } catch {
    throw new HttpError(400, "invalid_comment_id", "Comment id is invalid.");
  }
  if (!COMMENT_ID_RE.test(id)) {
    throw new HttpError(400, "invalid_comment_id", "Comment id is invalid.");
  }
  return { id, key: `comments/${id}.json` };
}

function commentId(now) {
  const reverseTime = String(9_999_999_999_999 - now).padStart(13, "0");
  return `${reverseTime}-${crypto.randomUUID()}`;
}

function parseStoredComment(value) {
  if (!isObject(value) || !COMMENT_ID_RE.test(value.id ?? "") || typeof value.displayName !== "string" || typeof value.body !== "string" || !validDate(value.createdAt)) {
    throw new Error("invalid stored comment");
  }
  return { id: value.id, displayName: value.displayName, body: value.body, createdAt: value.createdAt };
}

async function commentCallerHash(request, env) {
  const caller = request.headers.get("CF-Connecting-IP") ?? "";
  if (!caller || caller.length > 64 || !/^[0-9a-f:.]+$/i.test(caller)) {
    throw new HttpError(403, "caller_identity_required", "Comment submission requires a verified caller address.");
  }
  const salt = typeof env.COMMENTS_RATE_SALT === "string" && env.COMMENTS_RATE_SALT
    ? env.COMMENTS_RATE_SALT
    : typeof env.INGEST_TOKEN === "string" ? env.INGEST_TOKEN : "";
  if (!salt) {
    throw new HttpError(500, "missing_rate_limit_salt", "Comment rate limiting is unavailable.");
  }
  return sha256Hex(new TextEncoder().encode(`${salt}\0${caller}`));
}

async function consumeCommentRateLimit(request, env) {
  const callerHash = await commentCallerHash(request, env);
  const now = Date.now();
  const windowStart = Math.floor(now / COMMENT_RATE_WINDOW_MS) * COMMENT_RATE_WINDOW_MS;
  const firstSlot = crypto.getRandomValues(new Uint32Array(1))[0] % COMMENT_RATE_LIMIT;
  for (let offset = 0; offset < COMMENT_RATE_LIMIT; offset += 1) {
    const slot = (firstSlot + offset) % COMMENT_RATE_LIMIT;
    const key = `comments-rate/${callerHash}/${slot}.json`;
    try {
      const existing = await env.BUCKET.head(key);
      const storedWindow = Number(existing?.customMetadata?.windowStart);
      const validStoredWindow = Number.isSafeInteger(storedWindow)
        && storedWindow >= 0
        && storedWindow % COMMENT_RATE_WINDOW_MS === 0;
      if (existing && (!validStoredWindow || storedWindow >= windowStart)) continue;
      const stored = await env.BUCKET.put(key, JSON.stringify({ acceptedAt: new Date(now).toISOString() }), {
        onlyIf: existing ? { etagMatches: existing.etag } : new Headers({ "If-None-Match": "*" }),
        httpMetadata: { contentType: "application/json", cacheControl: "no-store" },
        customMetadata: { kind: "comment-rate-limit", windowStart: String(windowStart) },
      });
      if (stored) return;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (!/(?:10031|10058|precondition|too\s*many\s*requests|rate.?limit|\b429\b)/i.test(message)) throw error;
    }
  }
  const retryAfter = Math.max(1, Math.ceil((windowStart + COMMENT_RATE_WINDOW_MS - now) / 1_000));
  throw new HttpError(429, "rate_limited", "Too many comments. Please try again later.", { "Retry-After": String(retryAfter) });
}

async function postComment(request, env) {
  requireSameOrigin(request);
  assertJsonContentType(request);
  const input = validateCommentInput(await readJson(request, COMMENT_BODY_LIMIT));
  await consumeCommentRateLimit(request, env);
  const createdAt = new Date().toISOString();
  for (let attempt = 0; attempt < 3; attempt += 1) {
    const id = commentId(Date.parse(createdAt));
    const comment = { id, ...input, createdAt };
    const stored = await env.BUCKET.put(`comments/${id}.json`, JSON.stringify(comment), {
      onlyIf: new Headers({ "If-None-Match": "*" }),
      httpMetadata: { contentType: "application/json; charset=utf-8", cacheControl: "no-store" },
      customMetadata: { kind: "public-comment" },
    });
    if (stored) return json({ comment }, 201, { "Cache-Control": "no-store", Location: `/api/comments/${encodeURIComponent(id)}` });
  }
  throw new HttpError(503, "comment_id_collision", "Could not allocate a comment id. Please retry.", { "Retry-After": "1" });
}

async function getComments(request, env) {
  const url = new URL(request.url);
  const cursor = url.searchParams.get("cursor") ?? undefined;
  if (cursor && cursor.length > 2_048) {
    throw new HttpError(400, "invalid_cursor", "Comment cursor is invalid.");
  }
  const listed = await env.BUCKET.list({ prefix: "comments/", limit: COMMENT_PAGE_LIMIT, cursor });
  const comments = [];
  for (const summary of listed.objects) {
    if (!/^comments\/\d{13}-[0-9a-f-]+\.json$/.test(summary.key)) continue;
    try {
      const object = await env.BUCKET.get(summary.key);
      if (!object || object.size > COMMENT_BODY_LIMIT) continue;
      comments.push(parseStoredComment(JSON.parse(await object.text())));
    } catch {
      // Invalid objects are not exposed through the public listing.
    }
  }
  return json({ comments, cursor: listed.truncated ? listed.cursor : null }, 200, { "Cache-Control": "no-store" });
}

async function deleteComment(request, env, path) {
  await requireAuth(request, env);
  const existing = await env.BUCKET.head(path.key);
  if (!existing) throw new HttpError(404, "not_found", "Comment not found.");
  await env.BUCKET.delete(path.key);
  return json({ deleted: true, id: path.id }, 200, { "Cache-Control": "no-store" });
}

function validateLiveFrame(frame) {
  if (!isObject(frame) || frame.schemaVersion !== 1) {
    throw new HttpError(400, "invalid_live_frame", "schemaVersion must be 1.");
  }
  if (!SESSION_RE.test(frame.sessionId ?? "")) {
    throw new HttpError(400, "invalid_session", "sessionId is invalid.");
  }
  if (!Number.isSafeInteger(frame.sequence) || frame.sequence < 0) {
    throw new HttpError(400, "invalid_sequence", "sequence must be a non-negative integer.");
  }
  if (!validDate(frame.capturedAt)) {
    throw new HttpError(400, "invalid_captured_at", "capturedAt must be an RFC 3339 timestamp.");
  }
  if (!isObject(frame.state)) {
    throw new HttpError(400, "invalid_state", "state must be an object.");
  }
  if (frame.ended !== undefined && typeof frame.ended !== "boolean") {
    throw new HttpError(400, "invalid_ended", "ended must be boolean when supplied.");
  }
  if (frame.broadcastId !== undefined && !SESSION_RE.test(frame.broadcastId)) {
    throw new HttpError(400, "invalid_broadcast", "broadcastId is invalid.");
  }
  if (frame.score !== undefined && (!Number.isSafeInteger(frame.score) || frame.score < 0)) {
    throw new HttpError(400, "invalid_score", "score must be a non-negative integer when supplied.");
  }
  if (frame.isAscended !== undefined && typeof frame.isAscended !== "boolean") {
    throw new HttpError(400, "invalid_ascension", "isAscended must be boolean when supplied.");
  }
}

function contentTypeFor(filename) {
  if (filename === "manifest.json") return "application/json; charset=utf-8";
  if (filename.endsWith(".jsonl")) return "application/x-ndjson; charset=utf-8";
  if (filename.endsWith(".mp4")) return "video/mp4";
  throw new HttpError(400, "invalid_filename", "Only .jsonl, .mp4, and manifest.json uploads are accepted.");
}

function objectKey(sessionId, filename) {
  if (filename === "manifest.json") return `manifests/${sessionId}.json`;
  return `recordings/${sessionId}/${filename}`;
}

function trainingObjectKey(shardId, filename) {
  return `training/${shardId}/${filename}`;
}

function parseTrainingPath(pathname) {
  const match = /^\/api\/training\/([^/]+)\/([^/]+)$/.exec(pathname);
  if (!match) return null;
  let shardId;
  let filename;
  try {
    shardId = decodeURIComponent(match[1]);
    filename = decodeURIComponent(match[2]);
  } catch {
    throw new HttpError(400, "invalid_path", "The training path is invalid.");
  }
  if (!TRAINING_SHARD_RE.test(shardId) || !TRAINING_FILE_RE.test(filename)) {
    throw new HttpError(400, "invalid_path", "The training path is invalid.");
  }
  if (filename !== "manifest.json" && !filename.endsWith(".npzpack") && !filename.endsWith(".index.jsonl")) {
    throw new HttpError(400, "invalid_filename", "Training artifacts must be .npzpack, .index.jsonl, or manifest.json.");
  }
  return { shardId, filename };
}

function trainingContentType(filename) {
  if (filename === "manifest.json") return "application/json; charset=utf-8";
  if (filename.endsWith(".npzpack")) return "application/vnd.jev-nethack.transitions+npz";
  return "application/x-ndjson; charset=utf-8";
}

function trainingBytesEqual(left, right) {
  if (left.byteLength !== right.byteLength) return false;
  let result = 0;
  for (let index = 0; index < left.byteLength; index += 1) result |= left[index] ^ right[index];
  return result === 0;
}

async function validateFramedTransitionPack(bytes) {
  if (bytes.byteLength <= TRAINING_MAGIC.byteLength || !trainingBytesEqual(bytes.subarray(0, TRAINING_MAGIC.byteLength), TRAINING_MAGIC)) {
    throw new HttpError(422, "invalid_transition_pack", "The NPZ pack has an invalid magic header.");
  }
  let offset = TRAINING_MAGIC.byteLength;
  const frames = [];
  while (offset < bytes.byteLength) {
    if (bytes.byteLength - offset < 40) throw new HttpError(422, "invalid_transition_pack", "The NPZ pack contains a truncated frame header.");
    const lengthBig = new DataView(bytes.buffer, bytes.byteOffset + offset, 8).getBigUint64(0);
    if (lengthBig === 0n || lengthBig > BigInt(TRAINING_BODY_LIMIT)) throw new HttpError(422, "invalid_transition_pack", "The NPZ pack contains an invalid frame length.");
    const length = Number(lengthBig);
    const expectedDigest = bytes.subarray(offset + 8, offset + 40);
    offset += 40;
    if (length > bytes.byteLength - offset) {
      throw new HttpError(422, "invalid_transition_pack", "The NPZ pack contains an invalid frame length.");
    }
    const payload = bytes.subarray(offset, offset + length);
    if (!trainingBytesEqual(new Uint8Array(await crypto.subtle.digest("SHA-256", payload)), expectedDigest)) throw new HttpError(422, "invalid_transition_pack", "The NPZ pack frame checksum is invalid.");
    frames.push({ offset: offset - 40, payloadBytes: length, sha256: [...expectedDigest].map(value => value.toString(16).padStart(2, "0")).join("") });
    offset += length;
  }
  if (frames.length === 0 || offset !== bytes.byteLength) throw new HttpError(422, "invalid_transition_pack", "The NPZ pack must contain at least one complete frame.");
  return frames;
}

function validateTrainingManifest(manifest, shardId) {
  if (!isObject(manifest) || manifest.schemaVersion !== 1 || manifest.schema !== "jev-nethack-transition-pack/v1" || manifest.completed !== true) {
    throw new HttpError(400, "invalid_training_manifest", "A completed jev-nethack-transition-pack/v1 manifest is required.");
  }
  if (manifest.sessionId !== shardId || !TRAINING_SHARD_RE.test(manifest.sessionId)) {
    throw new HttpError(400, "invalid_training_manifest", "Manifest sessionId must match the shard path.");
  }
  if (manifest.broadcastId !== undefined && !TRAINING_SHARD_RE.test(manifest.broadcastId)) throw new HttpError(400, "invalid_training_manifest", "Manifest broadcastId is invalid.");
  if (!validDate(manifest.startedAt) || !validDate(manifest.endedAt) || Date.parse(manifest.endedAt) < Date.parse(manifest.startedAt)) throw new HttpError(400, "invalid_training_manifest", "Manifest timestamps are invalid.");
  if (!Number.isSafeInteger(manifest.transitionCount) || manifest.transitionCount < 1) throw new HttpError(400, "invalid_training_manifest", "transitionCount must be a positive integer.");
  if (!Array.isArray(manifest.artifacts) || manifest.artifacts.length !== 2) throw new HttpError(400, "invalid_training_manifest", "Exactly two training artifacts are required.");
  const names = new Set();
  for (const artifact of manifest.artifacts) {
    if (!isObject(artifact) || !TRAINING_FILE_RE.test(artifact.filename ?? "") || artifact.filename === "manifest.json" || names.has(artifact.filename)) throw new HttpError(400, "invalid_training_manifest", "Training artifact filenames are invalid or duplicated.");
    const expected = trainingContentType(artifact.filename);
    if (artifact.contentType !== expected.split(";", 1)[0] || !SHA256_RE.test(String(artifact.sha256 ?? "")) || !Number.isSafeInteger(artifact.bytes) || artifact.bytes <= 0) throw new HttpError(400, "invalid_training_manifest", `Training artifact metadata is invalid for ${artifact.filename}.`);
    names.add(artifact.filename);
  }
  if (![...names].some((name) => name.endsWith(".npzpack")) || ![...names].some((name) => name.endsWith(".index.jsonl"))) throw new HttpError(400, "invalid_training_manifest", "A training manifest requires one .npzpack and one .index.jsonl artifact.");
}

async function verifyTrainingArtifacts(bucket, manifest, shardId, verifyBodies = true) {
  let frames;
  let index;
  for (const artifact of manifest.artifacts) {
    const key = trainingObjectKey(shardId, artifact.filename);
    const stored = await bucket.head(key);
    if (!stored || stored.size !== artifact.bytes || stored.customMetadata?.sha256 !== artifact.sha256) throw new HttpError(409, "artifact_mismatch", `Stored training artifact does not match ${artifact.filename}.`);
    if (!verifyBodies) continue;
    const object = await bucket.get(key);
    if (!object) throw new HttpError(409, "artifact_mismatch", `Stored training artifact is missing ${artifact.filename}.`);
    const bytes = new Uint8Array(await object.arrayBuffer());
    if (bytes.byteLength !== artifact.bytes || await sha256Hex(bytes) !== artifact.sha256) throw new HttpError(409, "artifact_mismatch", `Stored training artifact checksum does not match ${artifact.filename}.`);
    if (artifact.filename.endsWith(".npzpack")) frames = await validateFramedTransitionPack(bytes);
    else {
      try { index = new TextDecoder("utf-8", { fatal: true }).decode(bytes).trimEnd().split("\n").map(line => JSON.parse(line)); }
      catch { throw new HttpError(422, "invalid_transition_index", "The transition index must contain complete JSON lines."); }
    }
  }
  if (!verifyBodies) return;
  if (frames.length !== manifest.transitionCount || index.length !== frames.length) throw new HttpError(422, "transition_count_mismatch", "The manifest, index and pack must contain the same number of transitions.");
  const ids = new Set();
  for (let i = 0; i < frames.length; i += 1) {
    const entry = index[i];
    if (!isObject(entry) || entry.schema !== manifest.schema || typeof entry.eventId !== "string" || !entry.eventId || ids.has(entry.eventId) || entry.offset !== frames[i].offset || entry.payloadBytes !== frames[i].payloadBytes || entry.sha256 !== frames[i].sha256) throw new HttpError(422, "invalid_transition_index", "The transition index does not match the checksummed pack frames.");
    ids.add(entry.eventId);
  }
}

async function putTraining(request, env, path) {
  await requireAuth(request, env);
  const sha256 = (request.headers.get("X-Content-SHA256") ?? "").toLowerCase();
  if (!SHA256_RE.test(sha256)) throw new HttpError(400, "invalid_sha256", "X-Content-SHA256 must be a 64-character hex digest.");
  const limit = path.filename === "manifest.json" ? MANIFEST_BODY_LIMIT : TRAINING_BODY_LIMIT;
  const length = requiredUploadLength(request, limit);
  const key = trainingObjectKey(path.shardId, path.filename);
  const existing = await env.BUCKET.head(key);
  const bytes = await readBounded(request, limit);
  if (bytes.byteLength !== length || await sha256Hex(bytes) !== sha256) throw new HttpError(422, "checksum_mismatch", "The upload body does not match Content-Length or X-Content-SHA256.");
  const location = `/api/training/${encodeURIComponent(path.shardId)}/${encodeURIComponent(path.filename)}`;
  if (existing) {
    if (existing.size !== length || existing.customMetadata?.sha256 !== sha256) throw new HttpError(409, "immutable_object_exists", "This immutable path contains different bytes.");
    return json({ stored: true, alreadyStored: true, shardId: path.shardId, filename: path.filename, bytes: length, sha256, download_url: new URL(location, request.url).href }, 200);
  }
  let manifest;
  if (path.filename === "manifest.json") {
    try { manifest = JSON.parse(new TextDecoder().decode(bytes)); } catch { throw new HttpError(400, "invalid_json", "Training manifest must be valid JSON."); }
    validateTrainingManifest(manifest, path.shardId);
    await verifyTrainingArtifacts(env.BUCKET, manifest, path.shardId);
  } else if (path.filename.endsWith(".npzpack")) {
    await validateFramedTransitionPack(bytes);
  }
  const stored = await env.BUCKET.put(key, bytes, {
    onlyIf: new Headers({ "If-None-Match": "*" }), sha256,
    httpMetadata: { contentType: trainingContentType(path.filename), cacheControl: "public, max-age=31536000, immutable", contentDisposition: `attachment; filename="${path.filename}"` },
    customMetadata: { shardId: path.shardId, filename: path.filename, sha256, ...(manifest ? { verifiedTraining: "v1" } : {}) },
  });
  if (!stored) throw new HttpError(409, "immutable_object_exists", "Training objects are immutable and this path already exists.");
  return json({ stored: true, shardId: path.shardId, filename: path.filename, bytes: length, sha256, completed: Boolean(manifest?.completed), download_url: new URL(location, request.url).href }, 201, { ETag: stored.httpEtag, Location: location });
}

async function getTraining(request, env, path) {
  const object = await env.BUCKET.get(trainingObjectKey(path.shardId, path.filename));
  if (!object) throw new HttpError(404, "not_found", "Training object not found.");
  const headers = new Headers(); object.writeHttpMetadata(headers);
  headers.set("Content-Type", trainingContentType(path.filename)); headers.set("Content-Disposition", `attachment; filename="${path.filename}"`); headers.set("ETag", object.httpEtag); headers.set("X-Content-Type-Options", "nosniff"); headers.set("Cache-Control", "public, max-age=31536000, immutable"); headers.set("Content-Length", String(object.size));
  return new Response(request.method === "HEAD" ? null : object.body, { status: 200, headers });
}

async function getTrainingArchive(request, env) {
  const url = new URL(request.url); const rawLimit = Number(url.searchParams.get("limit") ?? "20");
  if (!Number.isSafeInteger(rawLimit) || rawLimit < 1 || rawLimit > TRAINING_PAGE_LIMIT) throw new HttpError(400, "invalid_limit", `limit must be between 1 and ${TRAINING_PAGE_LIMIT}.`);
  const cursor = url.searchParams.get("cursor") ?? undefined;
  const listed = await env.BUCKET.list({ prefix: "training/", limit: rawLimit, cursor });
  const manifests = [];
  for (const summary of listed.objects) {
    if (!summary.key.endsWith("/manifest.json")) continue;
    const parts = summary.key.split("/"); const shardId = parts[1];
    try {
      const object = await env.BUCKET.get(summary.key); if (!object || object.size > MANIFEST_BODY_LIMIT || object.customMetadata?.verifiedTraining !== "v1") continue;
      const manifest = JSON.parse(await object.text()); validateTrainingManifest(manifest, shardId); await verifyTrainingArtifacts(env.BUCKET, manifest, shardId, false);
      manifests.push({ ...manifest, manifestUrl: `/api/training/${encodeURIComponent(shardId)}/manifest.json`, artifacts: manifest.artifacts.map((artifact) => ({ ...artifact, url: `/api/training/${encodeURIComponent(shardId)}/${encodeURIComponent(artifact.filename)}` })) });
    } catch { /* incomplete or invalid claims are never listed */ }
  }
  manifests.sort((a, b) => b.endedAt.localeCompare(a.endedAt));
  return json({ manifests, cursor: listed.truncated ? listed.cursor : null }, 200, { "Cache-Control": "public, max-age=5" });
}

function parseByteRange(header, size) {
  const match = /^bytes=(\d*)-(\d*)$/.exec(header);
  if (!match || (!match[1] && !match[2]) || size <= 0) {
    throw new HttpError(416, "invalid_range", "The requested byte range is not satisfiable.", { "Content-Range": `bytes */${size}` });
  }
  if (!match[1]) {
    const suffix = Number(match[2]);
    if (!Number.isSafeInteger(suffix) || suffix <= 0) {
      throw new HttpError(416, "invalid_range", "The requested byte range is not satisfiable.", { "Content-Range": `bytes */${size}` });
    }
    const length = Math.min(suffix, size);
    return { offset: size - length, length };
  }
  const offset = Number(match[1]);
  const requestedEnd = match[2] ? Number(match[2]) : size - 1;
  if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(requestedEnd) || offset >= size || requestedEnd < offset) {
    throw new HttpError(416, "invalid_range", "The requested byte range is not satisfiable.", { "Content-Range": `bytes */${size}` });
  }
  const end = Math.min(requestedEnd, size - 1);
  return { offset, length: end - offset + 1 };
}

function parseRecordingPath(pathname) {
  const match = /^\/api\/recordings\/([^/]+)\/([^/]+)$/.exec(pathname);
  if (!match) return null;
  let sessionId;
  let filename;
  try {
    sessionId = decodeURIComponent(match[1]);
    filename = decodeURIComponent(match[2]);
  } catch {
    throw new HttpError(400, "invalid_path", "The recording path is invalid.");
  }
  if (!SESSION_RE.test(sessionId) || !FILE_RE.test(filename)) {
    throw new HttpError(400, "invalid_path", "The recording path is invalid.");
  }
  contentTypeFor(filename);
  return { sessionId, filename };
}

function validateManifest(manifest, sessionId) {
  if (!isObject(manifest) || manifest.schemaVersion !== 1 || manifest.completed !== true) {
    throw new HttpError(400, "invalid_manifest", "A completed schemaVersion 1 manifest is required.");
  }
  if (manifest.sessionId !== sessionId) {
    throw new HttpError(400, "manifest_session_mismatch", "Manifest sessionId must match the upload path.");
  }
  if (manifest.broadcastId !== undefined && !SESSION_RE.test(manifest.broadcastId)) {
    throw new HttpError(400, "invalid_manifest", "Manifest broadcastId is invalid.");
  }
  if (!validDate(manifest.startedAt) || !validDate(manifest.endedAt) || Date.parse(manifest.endedAt) < Date.parse(manifest.startedAt)) {
    throw new HttpError(400, "invalid_manifest", "Manifest start and end timestamps are invalid.");
  }
  for (const field of ["frameCount", "actionCount"]) {
    if (!Number.isSafeInteger(manifest[field]) || manifest[field] < 0) {
      throw new HttpError(400, "invalid_manifest", `${field} must be a non-negative integer.`);
    }
  }
  if (!Array.isArray(manifest.artifacts) || manifest.artifacts.length < 2 || manifest.artifacts.length > 16) {
    throw new HttpError(400, "invalid_manifest", "Manifest artifacts must contain 2 to 16 entries.");
  }
  const names = new Set();
  for (const artifact of manifest.artifacts) {
    if (!isObject(artifact) || !FILE_RE.test(artifact.filename ?? "") || artifact.filename === "manifest.json") {
      throw new HttpError(400, "invalid_manifest", "Manifest artifact filename is invalid.");
    }
    const expectedType = contentTypeFor(artifact.filename);
    if (artifact.contentType !== expectedType.split(";")[0]) {
      throw new HttpError(400, "invalid_manifest", `Manifest contentType is invalid for ${artifact.filename}.`);
    }
    if (!SHA256_RE.test(artifact.sha256 ?? "") || !Number.isSafeInteger(artifact.bytes) || artifact.bytes <= 0) {
      throw new HttpError(400, "invalid_manifest", `Manifest checksum or size is invalid for ${artifact.filename}.`);
    }
    if (names.has(artifact.filename)) {
      throw new HttpError(400, "invalid_manifest", "Manifest artifact filenames must be unique.");
    }
    names.add(artifact.filename);
  }
  if (![...names].some((name) => name.endsWith(".jsonl")) || ![...names].some((name) => name.endsWith(".mp4"))) {
    throw new HttpError(400, "invalid_manifest", "A completed segment requires JSONL and MP4 artifacts.");
  }
}

async function verifyManifestArtifacts(bucket, manifest, sessionId) {
  for (const artifact of manifest.artifacts) {
    const stored = await bucket.head(objectKey(sessionId, artifact.filename));
    if (!stored || stored.size !== artifact.bytes || stored.customMetadata?.sha256 !== artifact.sha256) {
      throw new HttpError(409, "artifact_mismatch", `Stored artifact does not match manifest entry ${artifact.filename}.`);
    }
    if (artifact.filename.endsWith(".mp4")) {
      await verifyPlayableMp4(bucket, sessionId, artifact, stored.size);
    }
  }
}

function boxType(bytes, offset) {
  return String.fromCharCode(bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]);
}

function readBoxSize(bytes, offset, limit) {
  if (offset + 8 > bytes.byteLength) return null;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const size32 = view.getUint32(offset);
  const type = boxType(bytes, offset + 4);
  let headerSize = 8;
  let size = size32;
  if (size32 === 1) {
    if (offset + 16 > bytes.byteLength) return null;
    const size64 = view.getBigUint64(offset + 8);
    if (size64 > BigInt(Number.MAX_SAFE_INTEGER)) return null;
    size = Number(size64);
    headerSize = 16;
  } else if (size32 === 0) {
    size = limit - offset;
  }
  if (size < headerSize || offset + size > limit) return null;
  return { type, start: offset, contentStart: offset + headerSize, end: offset + size, size, headerSize };
}

function childBoxes(bytes, start, end) {
  const boxes = [];
  let offset = start;
  while (offset < end) {
    const box = readBoxSize(bytes, offset, end);
    if (!box) return null;
    boxes.push(box);
    offset = box.end;
    if (offset > bytes.byteLength) break;
  }
  return offset === end ? boxes : null;
}

function hasVideoTrack(bytes, moov) {
  if (moov.end > bytes.byteLength) return false;
  const moovChildren = childBoxes(bytes, moov.contentStart, moov.end);
  if (!moovChildren) return false;
  for (const trak of moovChildren.filter((box) => box.type === "trak")) {
    const trakChildren = childBoxes(bytes, trak.contentStart, trak.end);
    if (!trakChildren) continue;
    for (const mdia of trakChildren.filter((box) => box.type === "mdia")) {
      const mdiaChildren = childBoxes(bytes, mdia.contentStart, mdia.end);
      if (!mdiaChildren) continue;
      for (const hdlr of mdiaChildren.filter((box) => box.type === "hdlr")) {
        if (hdlr.contentStart + 12 <= hdlr.end && hdlr.contentStart + 12 <= bytes.byteLength &&
            boxType(bytes, hdlr.contentStart + 8) === "vide") {
          return true;
        }
      }
    }
  }
  return false;
}

function validateMp4Structure(bytes, totalSize) {
  const boxes = [];
  let offset = 0;
  while (offset < totalSize && offset < bytes.byteLength) {
    const box = readBoxSize(bytes, offset, totalSize);
    if (!box) return false;
    boxes.push(box);
    offset = box.end;
  }
  const ftyp = boxes.find((box) => box.type === "ftyp" && box.size >= 16);
  const moov = boxes.find((box) => box.type === "moov");
  const nonemptyMdat = boxes.some((box) => box.type === "mdat" && box.size > box.headerSize);
  return Boolean(ftyp && moov && nonemptyMdat && hasVideoTrack(bytes, moov));
}

async function verifyPlayableMp4(bucket, sessionId, artifact, storedSize) {
  const length = Math.min(storedSize, MP4_PROBE_LIMIT);
  const object = await bucket.get(objectKey(sessionId, artifact.filename), { range: { offset: 0, length } });
  if (!object) {
    throw new HttpError(409, "artifact_mismatch", `Stored artifact is missing ${artifact.filename}.`);
  }
  const bytes = new Uint8Array(await object.arrayBuffer());
  if (!validateMp4Structure(bytes, storedSize)) {
    throw new HttpError(422, "invalid_mp4", `Stored MP4 has no bounded playable video structure: ${artifact.filename}.`);
  }
}

async function putLive(request, env) {
  await requireAuth(request, env);
  const frame = await readJson(request, LIVE_BODY_LIMIT);
  validateLiveFrame(frame);
  const receivedAt = new Date().toISOString();
  const storedFrame = { ...frame, receivedAt };
  const body = JSON.stringify(storedFrame);
  const current = await env.BUCKET.get(LIVE_KEY);
  let onlyIf;
  if (current) {
    let previous;
    try {
      previous = JSON.parse(await current.text());
    } catch {
      throw new HttpError(500, "invalid_live_storage", "Stored live state is invalid.");
    }
    if (previous.sessionId === frame.sessionId && previous.sequence >= frame.sequence) {
      throw new HttpError(409, "stale_sequence", "Live frame sequence must increase within a session.");
    }
    onlyIf = { etagMatches: current.etag };
  } else {
    onlyIf = new Headers({ "If-None-Match": "*" });
  }
  let stored;
  try {
    stored = await env.BUCKET.put(LIVE_KEY, body, {
      onlyIf,
      httpMetadata: { contentType: "application/json; charset=utf-8", cacheControl: "no-store" },
      customMetadata: { sessionId: frame.sessionId, sequence: String(frame.sequence) },
    });
  } catch (error) {
    if (/rate|too many|limit/i.test(error instanceof Error ? error.message : String(error))) {
      throw new HttpError(429, "live_write_rate", "Live frames must be posted at most once per second.", { "Retry-After": "1" });
    }
    throw error;
  }
  if (!stored) {
    throw new HttpError(409, "live_write_conflict", "A newer live frame was stored concurrently.");
  }
  return json({ accepted: true, sessionId: frame.sessionId, sequence: frame.sequence, receivedAt }, 201, { "Cache-Control": "no-store" });
}

async function getLive(env) {
  const serverTime = new Date().toISOString();
  const object = await env.BUCKET.get(LIVE_KEY);
  if (!object) {
    return json({ frame: null, serverTime, live: false, ageMs: null, receivedAgeMs: null, captureAgeMs: null, freshnessWindowMs: LIVE_FRESHNESS_MS }, 200, { "Cache-Control": "no-store" });
  }
  let frame;
  try {
    frame = JSON.parse(await object.text());
  } catch {
    throw new HttpError(500, "invalid_live_storage", "Stored live state is invalid.");
  }
  const now = Date.parse(serverTime);
  const receivedAgeMs = now - Date.parse(frame.receivedAt);
  const captureAgeMs = now - Date.parse(frame.capturedAt);
  const ageMs = Math.max(0, receivedAgeMs, captureAgeMs);
  const live = frame.ended !== true && receivedAgeMs >= 0 && receivedAgeMs <= LIVE_FRESHNESS_MS && captureAgeMs >= -MAX_FUTURE_SKEW_MS && captureAgeMs <= LIVE_FRESHNESS_MS;
  return json({ frame, serverTime, live, ageMs, receivedAgeMs, captureAgeMs, freshnessWindowMs: LIVE_FRESHNESS_MS }, 200, { "Cache-Control": "no-store" });
}

function requiredUploadLength(request, limit) {
  const raw = request.headers.get("Content-Length");
  if (raw === null) {
    throw new HttpError(411, "length_required", "Content-Length is required for recording uploads.");
  }
  const length = Number(raw);
  if (!Number.isSafeInteger(length) || length <= 0) {
    throw new HttpError(400, "invalid_content_length", "Content-Length must be a positive integer.");
  }
  if (length > limit) {
    throw new HttpError(413, "payload_too_large", `Payload exceeds the ${limit}-byte limit.`);
  }
  return length;
}

async function putRecording(request, env, path) {
  await requireAuth(request, env);
  const sha256 = (request.headers.get("X-Content-SHA256") ?? "").toLowerCase();
  if (!SHA256_RE.test(sha256)) {
    throw new HttpError(400, "invalid_sha256", "X-Content-SHA256 must be a lowercase or uppercase 64-character hex digest.");
  }
  const isManifest = path.filename === "manifest.json";
  const limit = isManifest ? MANIFEST_BODY_LIMIT : RECORDING_BODY_LIMIT;
  const length = requiredUploadLength(request, limit);
  const key = objectKey(path.sessionId, path.filename);
  if (await env.BUCKET.head(key)) {
    throw new HttpError(409, "immutable_object_exists", "Recording objects are immutable and this path already exists.");
  }

  let value;
  let manifest;
  if (isManifest) {
    const bytes = await readBounded(request, limit);
    if (bytes.byteLength !== length || (await sha256Hex(bytes)) !== sha256) {
      throw new HttpError(422, "checksum_mismatch", "The manifest body does not match Content-Length or X-Content-SHA256.");
    }
    try {
      manifest = JSON.parse(new TextDecoder().decode(bytes));
    } catch {
      throw new HttpError(400, "invalid_json", "Manifest must be valid JSON.");
    }
    validateManifest(manifest, path.sessionId);
    await verifyManifestArtifacts(env.BUCKET, manifest, path.sessionId);
    value = bytes;
  } else {
    if (!request.body) {
      throw new HttpError(400, "missing_body", "A request body is required.");
    }
    value = request.body;
  }

  let stored;
  try {
    stored = await env.BUCKET.put(key, value, {
      onlyIf: new Headers({ "If-None-Match": "*" }),
      sha256,
      httpMetadata: {
        contentType: contentTypeFor(path.filename),
        cacheControl: "public, max-age=31536000, immutable",
        contentDisposition: `${path.filename.endsWith(".mp4") ? "inline" : "attachment"}; filename="${path.filename}"`,
      },
      customMetadata: {
        sessionId: path.sessionId,
        filename: path.filename,
        kind: isManifest ? "manifest" : path.filename.endsWith(".mp4") ? "video" : "events",
        sha256,
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (/sha|checksum|digest/i.test(message)) {
      throw new HttpError(422, "checksum_mismatch", "The upload body does not match X-Content-SHA256.");
    }
    throw error;
  }
  if (!stored) {
    throw new HttpError(409, "immutable_object_exists", "Recording objects are immutable and this path already exists.");
  }
  if (stored.size !== length) {
    await env.BUCKET.delete(key);
    throw new HttpError(400, "content_length_mismatch", "Upload bytes do not match Content-Length.");
  }
  const location = `/api/recordings/${encodeURIComponent(path.sessionId)}/${encodeURIComponent(path.filename)}`;
  return json({
    stored: true,
    sessionId: path.sessionId,
    filename: path.filename,
    bytes: length,
    sha256,
    completed: Boolean(manifest?.completed),
    download_url: new URL(location, request.url).href,
    artifact_url: new URL(location, request.url).href,
  }, 201, {
    ETag: stored.httpEtag,
    Location: location,
  });
}

async function getRecording(request, env, path) {
  const rangeHeader = request.headers.get("Range");
  if (rangeHeader && !path.filename.endsWith(".mp4")) {
    throw new HttpError(400, "range_not_supported", "Range requests are supported for MP4 recordings only.");
  }
  let object;
  let requestedRange;
  try {
    if (rangeHeader) {
      const metadata = await env.BUCKET.head(objectKey(path.sessionId, path.filename));
      if (!metadata) throw new HttpError(404, "not_found", "Recording object not found.");
      requestedRange = parseByteRange(rangeHeader, metadata.size);
      object = await env.BUCKET.get(objectKey(path.sessionId, path.filename), { range: requestedRange });
    } else {
      object = await env.BUCKET.get(objectKey(path.sessionId, path.filename));
    }
  } catch (error) {
    if (error instanceof HttpError) throw error;
    throw new HttpError(416, "invalid_range", "The requested byte range is not satisfiable.");
  }
  if (!object) {
    throw new HttpError(404, "not_found", "Recording object not found.");
  }
  const headers = new Headers();
  object.writeHttpMetadata(headers);
  headers.set("ETag", object.httpEtag);
  headers.set("Accept-Ranges", "bytes");
  headers.set("X-Content-Type-Options", "nosniff");
  headers.set("Cache-Control", "public, max-age=31536000, immutable");
  if (rangeHeader) {
    const end = requestedRange.offset + requestedRange.length - 1;
    headers.set("Content-Range", `bytes ${requestedRange.offset}-${end}/${object.size}`);
    headers.set("Content-Length", String(requestedRange.length));
    return new Response(object.body, { status: 206, headers });
  }
  headers.set("Content-Length", String(object.size));
  return new Response(object.body, { status: 200, headers });
}

async function getArchive(request, env) {
  const url = new URL(request.url);
  const rawLimit = Number(url.searchParams.get("limit") ?? "20");
  if (!Number.isSafeInteger(rawLimit) || rawLimit < 1 || rawLimit > MAX_ARCHIVE_PAGE) {
    throw new HttpError(400, "invalid_limit", `limit must be between 1 and ${MAX_ARCHIVE_PAGE}.`);
  }
  const cursor = url.searchParams.get("cursor") ?? undefined;
  if (cursor && cursor.length > 2048) {
    throw new HttpError(400, "invalid_cursor", "Archive cursor is invalid.");
  }
  const listed = await env.BUCKET.list({ prefix: "manifests/", limit: rawLimit, cursor, include: ["httpMetadata", "customMetadata"] });
  const recordings = [];
  let invalidManifestCount = 0;
  for (const summary of listed.objects) {
    const object = await env.BUCKET.get(summary.key);
    if (!object || object.size > MANIFEST_BODY_LIMIT) {
      invalidManifestCount += 1;
      continue;
    }
    try {
      const manifest = JSON.parse(await object.text());
      validateManifest(manifest, manifest.sessionId);
      await verifyManifestArtifacts(env.BUCKET, manifest, manifest.sessionId);
      recordings.push({
        sessionId: manifest.sessionId,
        broadcastId: manifest.broadcastId ?? null,
        startedAt: manifest.startedAt,
        endedAt: manifest.endedAt,
        frameCount: manifest.frameCount,
        actionCount: manifest.actionCount,
        artifacts: manifest.artifacts.map((artifact) => ({
          ...artifact,
          url: `/api/recordings/${encodeURIComponent(manifest.sessionId)}/${encodeURIComponent(artifact.filename)}`,
        })),
        manifestUrl: `/api/recordings/${encodeURIComponent(manifest.sessionId)}/manifest.json`,
        uploadedAt: summary.uploaded.toISOString(),
      });
    } catch {
      invalidManifestCount += 1;
    }
  }
  recordings.sort((left, right) => right.endedAt.localeCompare(left.endedAt));
  return json({ recordings, cursor: listed.truncated ? listed.cursor : null, invalidManifestCount }, 200, { "Cache-Control": "public, max-age=5" });
}

function indexResponse() {
  return new Response(INDEX_HTML, {
    status: 200,
    headers: {
      "Content-Type": "text/html; charset=utf-8",
      "Cache-Control": "public, max-age=60",
      "Content-Security-Policy": "default-src 'self'; connect-src 'self'; img-src 'self' data:; media-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; base-uri 'none'; frame-ancestors 'none'",
      "Referrer-Policy": "no-referrer",
      "X-Content-Type-Options": "nosniff",
    },
  });
}

async function route(request, env) {
  if (!env?.BUCKET) throw new HttpError(500, "missing_bucket", "Storage binding is unavailable.");
  const url = new URL(request.url);
  const commentPath = parseCommentPath(url.pathname);
  if (url.pathname === "/api/comments") {
    if (request.method === "GET") return getComments(request, env);
    if (request.method === "POST") return postComment(request, env);
    throw new HttpError(405, "method_not_allowed", "Method not allowed.", { Allow: "GET, POST" });
  }
  if (commentPath) {
    if (request.method === "DELETE") return deleteComment(request, env, commentPath);
    throw new HttpError(405, "method_not_allowed", "Method not allowed.", { Allow: "DELETE" });
  }
  const trainingPath = parseTrainingPath(url.pathname);
  if (url.pathname === "/api/training") {
    if (request.method === "GET") return getTrainingArchive(request, env);
    throw new HttpError(405, "method_not_allowed", "Method not allowed.", { Allow: "GET" });
  }
  if (trainingPath) {
    if (request.method === "GET" || request.method === "HEAD") return getTraining(request, env, trainingPath);
    if (request.method === "PUT") return putTraining(request, env, trainingPath);
    throw new HttpError(405, "method_not_allowed", "Method not allowed.", { Allow: "GET, HEAD, PUT" });
  }
  const recordingPath = parseRecordingPath(url.pathname);
  if (url.pathname === "/api/live") {
    if (request.method === "GET") return getLive(env);
    if (request.method === "POST") return putLive(request, env);
    throw new HttpError(405, "method_not_allowed", "Method not allowed.", { Allow: "GET, POST" });
  }
  if (url.pathname === "/api/archive") {
    if (request.method === "GET") return getArchive(request, env);
    throw new HttpError(405, "method_not_allowed", "Method not allowed.", { Allow: "GET" });
  }
  if (recordingPath) {
    if (request.method === "GET" || request.method === "HEAD") {
      const response = await getRecording(request, env, recordingPath);
      return request.method === "HEAD" ? new Response(null, response) : response;
    }
    if (request.method === "PUT") return putRecording(request, env, recordingPath);
    throw new HttpError(405, "method_not_allowed", "Method not allowed.", { Allow: "GET, HEAD, PUT" });
  }
  if (url.pathname.startsWith("/api/")) {
    throw new HttpError(404, "not_found", "API route not found.");
  }
  if (request.method !== "GET" && request.method !== "HEAD") {
    throw new HttpError(405, "method_not_allowed", "Method not allowed.", { Allow: "GET, HEAD" });
  }
  const response = indexResponse();
  return request.method === "HEAD" ? new Response(null, response) : response;
}

export async function handleRequest(request, env) {
  try {
    return await route(request, env);
  } catch (error) {
    if (error instanceof HttpError) {
      return fail(error.status, error.code, error.message, error.headers);
    }
    console.error(JSON.stringify({ message: "worker request failed", path: new URL(request.url).pathname, error: error instanceof Error ? error.message : "unknown error" }));
    return fail(500, "internal_error", "Internal server error.");
  }
}

export default {
  async fetch(request, env) {
    return handleRequest(request, env);
  },
};
