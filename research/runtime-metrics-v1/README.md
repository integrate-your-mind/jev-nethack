# Runtime metric repair

The implementation is in [until-win](../../until-win/). The three reviews cover the metric correction and a subsequent small shutdown fix and a provider-backoff display fix. Their source digests identify the exact reviewed code; the awaiting-decision telemetry review identifies the final until_win.py digest.

The deployed runtime separates native final score, last observed score, maximum observed score, and accumulated reward. The one historical correction changes only the completed episode-0 result and preserves earlier interrupted results and the current game resume point. The shutdown fix clears the active fragment only after its durable training close succeeds.

Local candidate and copied public-source suites each passed 95 tests with one explicit opt-in production-copy test skipped. Independent focused review passed. See [native metrics](../../docs/native-metrics.md) for the evidence and limitations. Production verification is recorded separately after the live readback.

Review copies redact the private runtime root. Original and public hashes are recorded in source-public-hashes.json.

The telemetry fix publishes a current observation before the first provider decision, including after exact replay. Provider backoff retains that observation timestamp and has no invented action or probability distribution. HTTP errors expose only a sanitized status code.
