import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import vm from "node:vm";

const htmlPath = new URL("./public/index.html", import.meta.url);
const html = readFileSync(htmlPath, "utf8");

test("viewer keeps the existing live and archive bridges and parses as JavaScript", () => {
  assert.match(html, /fetch\("\/api\/live"/);
  assert.match(html, /fetch\(`\/api\/archive/);
  assert.match(html, /frame\.decision|decisionObject/);
  assert.match(html, /metricsObject|frame\.metrics/);
  const script = html.match(/<script>\s*([\s\S]*?)\s*<\/script>/)?.[1];
  assert.ok(script);
  const result = spawnSync(process.execPath, ["--check", "/dev/stdin"], { input: script, encoding: "utf8" });
  assert.equal(result.status, 0, result.stderr);
});

test("viewer exposes bounded, session-scoped history and honest accuracy semantics", () => {
  assert.match(html, /HISTORY_LIMIT = 180/);
  assert.match(html, /Bounded browser view for session/);
  assert.match(html, /This is not action accuracy/);
  assert.match(html, /Model reported confidence/);
  assert.match(html, /Decision model/);
  assert.match(html, /Reported deaths \(recent\)/);
  assert.match(html, /Ascensions/);
  assert.match(html, /Recovery\/backoff/);
});

function viewerHelpers() {
  const script = html.match(/<script>\s*([\s\S]*?)\s*<\/script>/)?.[1];
  const start = script.indexOf("const first =");
  const end = script.indexOf("const status =");
  return vm.runInNewContext(`(function(){${script.slice(start, end)}; return {asProbability, probabilityEntries, actionLabel, telemetrySample, outcomeSummary, movementSummary, mergeArchivePage, runtimePauseState, commentPage, commentRequest};})()`);
}

test("provider credit backoff is a bounded pause and clears when the runner resumes", () => {
  const {runtimePauseState} = viewerHelpers();
  const now = Date.parse("2026-09-18T18:46:00Z");
  const paused = {runtime: {phase: "provider_backoff", errorType: "JevHTTPError:402", receivedAgeMs: 255865, retryAt: "2026-09-18T18:47:29Z"}};
  assert.equal(runtimePauseState(paused, now).kind, "credits");
  assert.equal(runtimePauseState({...paused, runtime: {...paused.runtime, errorType: "JevHTTPError:429 quota"}}, now).kind, "provider");
  assert.equal(runtimePauseState(paused, Date.parse("2026-09-18T18:54:00Z")).kind, "offline");
  const resumed = {live: true, runtime: {phase: "playing", receivedAgeMs: 200, retryAt: null}};
  assert.equal(runtimePauseState(resumed, now).kind, "offline");
  assert.match(html, /pause\.kind === "credits"/);
  assert.match(html, /else if \(fresh\) status\("live", "Live"\)/);
});

test("comments normalize pages and keep user content on the text-only path", () => {
  const {commentPage, commentRequest} = viewerHelpers();
  const page = commentPage({comments: [{id: "1", displayName: "<name>", body: "<script>alert(1)</script>"}, null, "ignored"], cursor: "next"});
  assert.equal(page.comments.length, 1);
  assert.equal(page.cursor, "next");
  assert.deepEqual(JSON.parse(JSON.stringify(commentRequest("  Romy  ", "  hello  ", ""))), {displayName: "Romy", body: "hello", website: ""});
  assert.match(html, /name\.textContent = String\(item\.displayName/);
  assert.match(html, /body\.textContent = String\(item\.body/);
  assert.doesNotMatch(html, /comments-list[\s\S]{0,2500}innerHTML/);
  assert.match(html, /maxlength="40"/);
  assert.match(html, /maxlength="2000"/);
  assert.match(html, /still in the form/);
  assert.match(html, /Retry-After/);
});

test("pure viewer helpers validate probabilities, criteria labels, and episode identity", () => {
  const helpers = viewerHelpers();
  assert.equal(helpers.asProbability(.5), .5);
  assert.ok(Number.isNaN(helpers.asProbability(50)));
  const payload = {frame: {episode: 7, sequence: 12, state: {player: {turn: 91}}, decision: {choice: "a0", probabilities: {a0: .7, a1: -.1, a2: 1.2}, criteria: {a0: "Press k: move north"}}}};
  const entries = helpers.probabilityEntries(payload);
  assert.deepEqual(JSON.parse(JSON.stringify(entries)), [{id: "a0", label: "Press k: move north", value: .7}]);
  assert.equal(helpers.telemetrySample(payload).key, "7:12");
  assert.equal(helpers.telemetrySample(payload).turn, 91);
});

test("outcome summary keeps authoritative counters and never invents deaths", () => {
  const {outcomeSummary} = viewerHelpers();
  assert.deepEqual(JSON.parse(JSON.stringify(outcomeSummary({completedEpisodes: 8, ascensions: 2, interruptedEpisodes: 3}))), {
    source: "counters", windowLabel: "Continuous run totals; score chart shows up to 100 recent completed episodes", total: 11,
    completed: 8, ascension: 2, interrupted: 3, otherCompleted: 6,
    death: null, unknown: null, scores: [], recentCompleted: 0, recentOutcomesReported: 0, recentDeaths: null, latestCompleted: null,
  });
  assert.equal(outcomeSummary({completedEpisodes: 8, ascensions: 2}).death, null);
  assert.equal(outcomeSummary({}), null);
});

test("episode result summaries use strict terminal ascension evidence and a bounded score window", () => {
  const {outcomeSummary} = viewerHelpers();
  const results = Array.from({length: 101}, (_, index) => ({episodeId: index + 1, officialFinalScore: index, terminated: true, isAscended: index === 100}));
  results[0] = {episodeId: 1, score: 999, status: "dead"};
  results.push({episodeId: 102, score: 12, truncated: true});
  const summary = outcomeSummary({episodeResults: results});
  assert.equal(summary.source, "episodeResults");
  assert.equal(summary.completed, 99);
  assert.equal(summary.ascension, 1);
  assert.equal(summary.interrupted, 1);
  assert.equal(summary.otherCompleted, 98);
  assert.equal(summary.death, null);
  assert.equal(summary.scores.length, 99);
  assert.equal(summary.scores[0].episode, 3);
  assert.match(summary.windowLabel, /^Last 100 reported episode results only$/);
});

 test("continuous counters take precedence over the bounded result window", () => {
  const {outcomeSummary} = viewerHelpers();
  const value = outcomeSummary({completedEpisodes:200,ascensions:1,interruptedEpisodes:2,episodeResults:[{episodeId:200,officialFinalScore:120,terminated:true,isAscended:false}]});
  assert.equal(value.completed,200); assert.equal(value.ascension,1); assert.equal(value.scores.length,1);
  assert.equal(value.interrupted,2); assert.equal(value.total,202);
  assert.equal(outcomeSummary({completedEpisodes:1,ascensions:2}),null);
  assert.equal(outcomeSummary({completedEpisodes:1.5,ascensions:0}),null);
});

const position = (x, turn, extra = {}) => ({dungeon: 0, level: 1, x, y: 4, turn, score: 102, depth: 1, ...extra});
const movementFrame = actions => ({frame: {capturedAt: "2026-09-18T14:50:00Z", state: {recent_actions: actions}}});

test("movement counts actual contiguous action history and does not confuse turns with progress", () => {
  const {movementSummary} = viewerHelpers();
  const actions = Array.from({length: 8}, (_, i) => ({before: position(37 + i % 2, 100 + i), after: position(37 + (i + 1) % 2, 101 + i)}));
  const payload = movementFrame(actions);
  const summary = movementSummary(payload);
  assert.equal(summary.actions, 8); assert.equal(summary.positions, 2);
  assert.equal(summary.positionChanges, 8); assert.equal(summary.returns, 7); assert.equal(summary.pairs, 7);
  assert.equal(summary.turnDelta, 8); assert.equal(summary.scoreDelta, 0); assert.equal(summary.depthDelta, 0);
  assert.deepEqual(movementSummary(payload), summary, "repeated polling does not accumulate actions");
});

test("waits, missing data and non-contiguous windows never create apparent backtracking", () => {
  const {movementSummary} = viewerHelpers();
  assert.equal(movementSummary({}), null);
  const waits = movementSummary(movementFrame([{before: position(37, 1), after: position(37, 2)}, {before: position(37, 2), after: position(37, 3)}]));
  assert.equal(waits.returns, 0); assert.equal(waits.positionChanges, 0); assert.equal(waits.positions, 1);
  for (const bad of [null, {before: position(99, 3), after: position(37, 4)}, {before: {x: 38, y: 4}, after: {x: 37, y: 4}}]) {
    const summary = movementSummary(movementFrame([{before: position(37, 1), after: position(38, 2)}, bad]));
    assert.equal(summary.returns, null); assert.equal(summary.positions, null); assert.equal(summary.turnDelta, null);
  }
});

test("locations include dungeon level and the reported window is bounded", () => {
  const {movementSummary} = viewerHelpers();
  const levelChange = movementSummary(movementFrame([{before: position(37, 1), after: position(37, 2, {level: 2, depth: 2})}]));
  assert.equal(levelChange.positions, 2); assert.equal(levelChange.positionChanges, 1); assert.equal(levelChange.depthDelta, 1); assert.equal(levelChange.pairs, 0);
  const many = Array.from({length: 120}, (_, i) => ({before: position(i, i), after: position(i + 1, i + 1)}));
  assert.equal(movementSummary(movementFrame(many)).actions, 100);
});

test("archive pagination reaches beyond the old cap, deduplicates refreshes and preserves existing recordings", () => {
  const {mergeArchivePage} = viewerHelpers();
  let page = {records: [], cursor: null, seen: []};
  const first = {sessionId: "recording-0", frameCount: 100};
  page = mergeArchivePage(page, [first], "cursor-1", false);
  for (let i = 1; i < 1002; i++) page = mergeArchivePage(page, [{sessionId: `recording-${i}`}], `cursor-${i + 1}`, true);
  assert.equal(page.records.length, 1002);
  page = mergeArchivePage(page, [{sessionId: "recording-0", frameCount: 100}], "cursor-1", false);
  assert.equal(page.records.length, 1002); assert.equal(page.records[0], first);
  assert.throws(() => mergeArchivePage(page, [], "cursor-1", true), /repeated cursor/);
  page = mergeArchivePage(page, [], null, true); assert.equal(page.cursor, null);
  assert.equal(page.records[0], first);
  assert.match(html, /if \(!key \|\| archiveCards\.has\(key\)\) continue/);
  assert.match(html, /player\.preload = "none"/);
  assert.doesNotMatch(html, /setInterval\(loadArchive/);
});

 test("native final scores, observed score, reward and reported deaths stay distinct", () => {
  const {outcomeSummary} = viewerHelpers();
  const rows = [
    {episodeId: 0, score: 0, terminated: true},
    {episodeId: 1, officialFinalScore: null, score: 777, lastObservedScore: 90, totalReward: 92, terminated: true, endStatus: "DEATH"},
    {episodeId: 2, officialFinalScore: 138, lastObservedScore: 138, totalReward: 138, terminated: true, endStatus: "DEATH", deathCause: "died of starvation"},
    {episodeId: 3, officialFinalScore: 0, terminated: true, endStatus: "DEATH"},
    {episodeId: 4, officialFinalScore: "999", terminated: true, endStatus: "DEATH"},
    {episodeId: 5, officialFinalScore: 999, terminated: false, status: "stop_requested", endStatus: "DEATH"},
  ];
  const value = outcomeSummary({completedEpisodes: 500, ascensions: 0, interruptedEpisodes: 1, episodeResults: rows});
  assert.deepEqual(JSON.parse(JSON.stringify(value.scores)), [{episode: 2, score: 138}, {episode: 3, score: 0}]);
  assert.equal(value.completed, 500);
  assert.equal(value.recentDeaths, 4);
  assert.equal(value.recentOutcomesReported, 4);
  assert.equal(value.recentCompleted, 5);
  assert.equal(value.latestCompleted.episodeId, 4);
  assert.equal(outcomeSummary({episodeResults: [{terminated: true, isAscended: false}]}).recentDeaths, null);
  assert.equal(outcomeSummary({episodeResults: [{terminated: false, status: "stop_requested"}]}).interrupted, 1);
});
