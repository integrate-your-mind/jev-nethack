# Bounded local recording storage

The continuous runner and archive publisher use separate limits:

| Limit | Default | What it covers |
| --- | --- | --- |
| Closed recording cache | 64 MiB | Local MP4 and per-segment JSONL copies |
| Derived upload archive cache | 64 MiB | Verified local GitHub upload tar copies |
| Recorder backlog | 128 MiB | MP4, PNG frames and per-segment JSONL under the live recording root, plus reservations |
| Upload backlog | 128 MiB | The publisher's local release-outbox copies, with room reserved for new work |
| Free-space reserve | 64 MiB plus reservations | Room for recording work before another Jev decision or game action |

Reservations include 32 MiB for each active encoder, 32 MiB for the next segment and 4 MiB for the next frame/event. A playable but truncated or oversized encoding is rejected; source frames remain available for recovery.

When capacity is unavailable, gameplay pauses before calling Jev or taking another action. The runner checks again every 15 seconds and resumes when capacity returns. It seals a partial segment when safe so the publisher can upload it. The viewer shows “Paused · archiving recordings” only while pause heartbeats are fresh; it preserves the original capture time and does not label a stale game frame Live.

The publisher evicts the oldest eligible closed recording copies until the cache is within budget. Each removal requires the final recorder closure receipt, matching local catalog and artifact hashes, fresh full Site readback, a matching GitHub release asset digest and its previously verified item/batch binding, and no open writer. A durable tombstone is written before unlinking. Interrupted eviction can resume from that tombstone. Recorder finalization and publisher rediscovery recognize those exact missing files.

Unverified footage, PNG source frames, manifests, receipts, state, recovery journals, transition packs, NPZ observations, native ttyrec and top-level event logs are not evicted by this policy. If enough footage cannot be safely archived, the runner stays paused instead of silently discarding it. These are recording limits, not a limit on the entire training corpus or other applications' storage.

Derived upload archives have their own cache and admission check so they cannot bypass the recording pause. Removing a verified tar duplicate requires matching GitHub batch and item receipts, a fresh remote digest, an unchanged local file and no pending operation. Training originals stay local. The two backlogs are checked separately so two caches at their target sizes do not prevent play.

## Configuration

The publisher flags `--recording-cache-bytes 67108864` and `--release-cache-bytes 67108864` make both defaults explicit; `0` disables the corresponding eviction. Run the continuous player with `--recording-tombstone-root <publisher-runtime-root>/recording-tombstones` so it uses the same proof directory as the publisher and checks that publisher's release-outbox backlog. Keep that directory and publisher receipts after pruning.

Deploy the compatible recorder and publisher together. Gracefully stop/finalize an older recorder before enabling deletion, then resume the saved game through its verified action history. Deleting recovery data or starting a new seed is not part of this update.

Remote copies remain in the public viewer archive and daily GitHub Releases. Verification proves their contents at the recorded check time; continued hosting depends on those services and does not guarantee permanent availability.
