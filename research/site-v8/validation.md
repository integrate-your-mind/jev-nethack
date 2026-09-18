# Public Site v8 verification

The viewer now links to the public codebase and supports persistent plain-text comments. Provider credit exhaustion is distinguished from a lost connection, and the original ten-second gameplay freshness rule remains intact.

- Source: d3f5b3b491b9bdfac1a6d9aebde599f4d7ff370f.
- All 35 local Site tests passed with no skips.
- Independent backend review found and verified a fix for an old request overwriting a newer rate-limit window. Monotonic stored epochs and atomic conditional writes prevent that rollback.
- Deployed main script matched the local source byte for byte. The hosting layer adds its own script, so whole-page equality was not claimed.
- A clearly labelled temporary system comment returned HTTP 201, was read back through the public API, deleted using the private owner credential, and confirmed absent. No test comment was retained.
- The public feed was live at action 5,656 at 19:03 UTC on September 18, 2026. The existing process resumed from action 5,280 after credits returned; no restart or new seed was used for that resumption.
- Jev consultation a7c782dc-0b14-4c2d-8397-a18be67fecca selected completing the Site while observing automatic resumption. Model jev-1.13.0, selected probability 1.0. This advice is not execution evidence.
- Browser automation could not reconnect to the existing tab during this update. Verification used public HTTP, the complete local regression suite, and the live runner state; visual browser acceptance is not claimed.
- No ascension, SOTA result, or trained-policy success is claimed.

Comments remain separate from game decisions and training data. Rate limiting is per observed network address with salted hashes and three stable slots per caller. Storage isolation uses key prefixes within the existing bucket; it does not isolate hosting quotas. Anonymous listing reads at most 25 comments per page.
