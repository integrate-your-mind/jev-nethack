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
  assert.match(html, /Other completed/);
  assert.match(html, /Ascensions/);
  assert.match(html, /Recovery\/backoff/);
});

function viewerHelpers() {
  const script = html.match(/<script>\s*([\s\S]*?)\s*<\/script>/)?.[1];
  const start = script.indexOf("const first =");
  const end = script.indexOf("const status =");
  return vm.runInNewContext(`(function(){${script.slice(start, end)}; return {asProbability, probabilityEntries, actionLabel, telemetrySample, outcomeSummary};})()`);
}

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
    death: null, unknown: null, scores: [],
  });
  assert.equal(outcomeSummary({completedEpisodes: 8, ascensions: 2}).death, null);
  assert.equal(outcomeSummary({}), null);
});

test("episode result summaries use strict terminal ascension evidence and a bounded score window", () => {
  const {outcomeSummary} = viewerHelpers();
  const results = Array.from({length: 101}, (_, index) => ({episodeId: index + 1, score: index, terminated: true, isAscended: index === 100}));
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
  const value = outcomeSummary({completedEpisodes:200,ascensions:1,interruptedEpisodes:2,episodeResults:[{episodeId:200,score:120,terminated:true,isAscended:false}]});
  assert.equal(value.completed,200); assert.equal(value.ascension,1); assert.equal(value.scores.length,1);
  assert.equal(value.interrupted,2); assert.equal(value.total,202);
  assert.equal(outcomeSummary({completedEpisodes:1,ascensions:2}),null);
  assert.equal(outcomeSummary({completedEpisodes:1.5,ascensions:0}),null);
});
