"""Fail-closed retention for remotely verified closed recording artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
from typing import Any, Callable, Mapping, Sequence


DEFAULT_RECORDING_CACHE_BYTES = 64 * 1024 * 1024
DEFAULT_RELEASE_CACHE_BYTES = 64 * 1024 * 1024
TOMBSTONE_SCHEMA = "jev-nethack-recording-tombstone/v1"
_ELIGIBLE = re.compile(r"^segment-\d{4}\.(?:mp4|jsonl)$")
_RELEASE_ARCHIVE = re.compile(r"^jev-nethack-archive-[A-Za-z0-9._-]+\.tar\.gz$")
_SITE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")
_SHA = re.compile(r"^[a-f0-9]{64}$")


class RetentionError(RuntimeError):
    pass


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value), ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8") + b"\n"


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(_canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path, limit: int = 4 * 1024 * 1024) -> dict[str, Any]:
    if path.is_symlink():
        raise RetentionError(f"retention metadata may not be a symlink: {path}")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise RetentionError(f"invalid retention metadata: {path}") from exc
    if not 0 < len(raw) <= limit or not isinstance(value, dict):
        raise RetentionError(f"invalid retention metadata: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_inside(path: Path, roots: Sequence[Path]) -> Path:
    absolute = Path(os.path.abspath(path))
    resolved_roots = [root.resolve() for root in roots]
    try:
        parent = absolute.parent.resolve(strict=True)
    except OSError as exc:
        raise RetentionError(f"recording artifact parent cannot be resolved: {path}") from exc
    candidate = parent / absolute.name
    if not any(candidate.is_relative_to(root) for root in resolved_roots):
        raise RetentionError(f"recording artifact escapes configured roots: {path}")
    return candidate


def _fingerprint(path: Path, size: int, digest: str) -> dict[str, Any]:
    try:
        details = path.lstat()
    except OSError as exc:
        raise RetentionError(f"recording artifact cannot be inspected: {path}") from exc
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise RetentionError(f"recording artifact is not a regular non-symlink file: {path}")
    if details.st_size != size or _sha256(path) != digest:
        raise RetentionError(f"recording artifact changed after completion: {path}")
    return {
        "path": str(path), "bytes": size, "sha256": digest,
        "device": details.st_dev, "inode": details.st_ino, "mtimeNs": details.st_mtime_ns,
    }


def _matches_fingerprint(path: Path, expected: Mapping[str, Any]) -> bool:
    try:
        details = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISREG(details.st_mode) and not stat.S_ISLNK(details.st_mode)
        and details.st_dev == expected.get("device") and details.st_ino == expected.get("inode")
        and details.st_size == expected.get("bytes") and details.st_mtime_ns == expected.get("mtimeNs")
        and _sha256(path) == expected.get("sha256")
    )


def _default_open_writer(path: Path) -> bool:
    """Use lsof access modes; inability to prove closure fails closed."""
    command = "/usr/sbin/lsof" if Path("/usr/sbin/lsof").exists() else "lsof"
    try:
        result = subprocess.run(
            [command, "-F", "a", "--", str(path)], capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RetentionError("open-writer inspection is unavailable") from exc
    if result.returncode not in (0, 1):
        raise RetentionError("open-writer inspection failed")
    return any(line in ("aw", "au") for line in result.stdout.splitlines())


def _tombstone_path(root: Path, item: Any) -> Path:
    identity = f"{item.source_path.resolve()}\0{item.source_sha256}".encode()
    return root / f"recording-{hashlib.sha256(identity).hexdigest()}.json"


def _artifact_specs(item: Any, roots: Sequence[Path]) -> list[dict[str, Any]]:
    if item.kind != "recording":
        return []
    specs: list[dict[str, Any]] = []
    for artifact in item.artifacts:
        if not _ELIGIBLE.fullmatch(artifact.filename):
            continue
        path = _absolute_inside(Path(artifact.path), roots)
        if path.name != artifact.filename:
            raise RetentionError("recording artifact filename/path mismatch")
        specs.append({"artifact": artifact, "path": path})
    return specs


def _validate_catalog(item: Any, catalog_root: Path) -> None:
    key = hashlib.sha256(str(item.source_path.resolve()).encode()).hexdigest()
    path = catalog_root / f"recording-{key}.json"
    value = _read_json(path)
    if value.get("schema") != "jev-nethack-source-catalog/v1":
        raise RetentionError("recording source catalog schema is invalid")
    matches = [
        raw for raw in value.get("items", []) if isinstance(raw, dict)
        and raw.get("kind") == "recording" and raw.get("itemId") == item.item_id
        and raw.get("sourceSha256") == item.source_sha256
    ]
    if len(matches) != 1:
        raise RetentionError("recording source catalog lacks exact item binding")
    expected = {
        (str(Path(artifact.path)), artifact.size, artifact.sha256) for artifact in item.artifacts
    }
    actual = {
        (str(Path(raw.get("path", ""))), raw.get("bytes"), raw.get("sha256"))
        for raw in matches[0].get("artifacts", []) if isinstance(raw, dict)
    }
    if not expected.issubset(actual):
        raise RetentionError("recording source catalog lacks exact artifact binding")


def _completion_receipt(item: Any, roots: Sequence[Path]) -> dict[str, Any]:
    match = re.fullmatch(r"segment-(\d{4})\.manifest\.json", item.source_path.name)
    if match is None:
        raise RetentionError("recording manifest filename cannot identify its closure receipt")
    path = _absolute_inside(
        item.source_path.with_name(f"segment-{match.group(1)}.receipt.json"), roots
    )
    value = _read_json(path)
    try:
        receipt_ended = datetime.fromisoformat(str(value.get("endedAt")).replace("Z", "+00:00"))
        item_ended = datetime.fromisoformat(str(item.ended_at).replace("Z", "+00:00"))
    except ValueError as exc:
        raise RetentionError("recording closure receipt endedAt is invalid") from exc
    if receipt_ended.tzinfo is None or item_ended.tzinfo is None or receipt_ended < item_ended:
        raise RetentionError("recording closure receipt predates the completed manifest")
    if (
        value.get("schemaVersion") != 1 or value.get("completed") is not True
        or value.get("segmentId") != item.item_id
    ):
        raise RetentionError("recording closure receipt is incomplete or mismatched")
    raw_artifacts = value.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise RetentionError("recording closure receipt artifacts are invalid")
    expected = {
        (artifact.filename, artifact.size, artifact.sha256) for artifact in item.artifacts
    } | {
        ("manifest.json", len(item.site_manifest), hashlib.sha256(item.site_manifest).hexdigest())
    }
    actual = {
        (raw.get("filename"), raw.get("bytes"), raw.get("sha256"))
        for raw in raw_artifacts if isinstance(raw, dict)
    }
    if not expected.issubset(actual):
        raise RetentionError("recording closure receipt lacks exact artifact hashes")
    return {
        "path": str(path), "sha256": _sha256(path),
        "endedAt": value["endedAt"], "segmentId": item.item_id,
    }


def _remote_evidence(
    item: Any, receipts: Path,
    site_verify: Callable[[str, int, str], bool],
    github_verify: Callable[[str, str, str], bool],
    site_path: Callable[[Any, str], str],
    checked_at: str,
) -> dict[str, Any]:
    site_receipt_path = receipts / "site" / f"recording-{item.item_id}.verified.json"
    github_item_path = receipts / "github-items" / f"recording-{item.source_id}.verified.json"
    site = _read_json(site_receipt_path)
    github_item = _read_json(github_item_path)
    manifest_sha = hashlib.sha256(item.site_manifest).hexdigest()
    if (
        site.get("verified") is not True or site.get("sourceSha256") != item.source_sha256
        or site.get("manifestSha256") != manifest_sha
    ):
        raise RetentionError("Site receipt lacks exact recording binding")
    tag, asset, archive_sha = (
        github_item.get("tag"), github_item.get("asset"), github_item.get("archiveSha256")
    )
    if (
        github_item.get("verified") is not True
        or github_item.get("sourceSha256") != item.source_sha256
        or not all(isinstance(value, str) and value for value in (tag, asset, archive_sha))
        or not _SHA.fullmatch(archive_sha)
    ):
        raise RetentionError("GitHub item receipt lacks exact recording binding")
    batch_path = receipts / "github-batches" / f"{asset}.verified.json"
    batch = _read_json(batch_path)
    bound = any(
        isinstance(raw, dict) and raw.get("kind") == "recording"
        and raw.get("id") == item.source_id and raw.get("sourceSha256") == item.source_sha256
        for raw in batch.get("items", [])
    )
    if (
        batch.get("schema") != "jev-nethack-github-publish-receipt/v1"
        or batch.get("verified") is not True or batch.get("tag") != tag
        or batch.get("asset") != asset or batch.get("sha256") != archive_sha or not bound
    ):
        raise RetentionError("GitHub batch receipt lacks exact recording membership")
    for artifact in item.artifacts:
        try:
            verified = site_verify(site_path(item, artifact.filename), artifact.size, artifact.sha256)
        except Exception as exc:
            raise RetentionError(f"fresh Site verification unavailable: {artifact.filename}") from exc
        if not verified:
            raise RetentionError(f"fresh Site verification failed: {artifact.filename}")
    try:
        manifest_verified = site_verify(
            site_path(item, "manifest.json"), len(item.site_manifest), manifest_sha
        )
    except Exception as exc:
        raise RetentionError("fresh Site manifest verification unavailable") from exc
    if not manifest_verified:
        raise RetentionError("fresh Site manifest verification failed")
    try:
        github_verified = github_verify(tag, asset, archive_sha)
    except Exception as exc:
        raise RetentionError("fresh GitHub asset verification unavailable") from exc
    if not github_verified:
        raise RetentionError("fresh GitHub asset verification failed")
    return {
        "checkedAt": checked_at,
        "site": {"manifestSha256": manifest_sha, "receipt": str(site_receipt_path)},
        "github": {
            "tag": tag, "asset": asset, "archiveSha256": archive_sha,
            "itemReceipt": str(github_item_path), "batchReceipt": str(batch_path),
        },
    }


def _validate_tombstone(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.get("schema") != TOMBSTONE_SCHEMA:
        raise RetentionError("recording tombstone schema is invalid")
    if raw.get("state") not in ("prepared", "pruned"):
        raise RetentionError("recording tombstone state is invalid")
    if (
        raw.get("kind") != "recording" or not isinstance(raw.get("itemId"), str)
        or not isinstance(raw.get("sourceId"), str) or not isinstance(raw.get("sourcePath"), str)
        or not Path(raw["sourcePath"]).is_absolute()
        or not isinstance(raw.get("sourceSha256"), str)
        or not _SHA.fullmatch(raw["sourceSha256"])
    ):
        raise RetentionError("recording tombstone source identity is invalid")
    closure, remote = raw.get("closureReceipt"), raw.get("remote")
    if (
        not isinstance(closure, dict) or not isinstance(closure.get("path"), str)
        or not Path(closure["path"]).is_absolute()
        or not isinstance(closure.get("sha256"), str) or not _SHA.fullmatch(closure["sha256"])
        or closure.get("segmentId") != raw["itemId"]
        or not isinstance(remote, dict) or not isinstance(remote.get("checkedAt"), str)
        or not isinstance(remote.get("site"), dict)
        or not isinstance(remote.get("github"), dict)
        or not isinstance(remote["site"].get("manifestSha256"), str)
        or not _SHA.fullmatch(remote["site"]["manifestSha256"])
        or not isinstance(remote["github"].get("archiveSha256"), str)
        or not _SHA.fullmatch(remote["github"]["archiveSha256"])
        or not all(
            isinstance(remote["github"].get(field), str) and remote["github"][field]
            for field in ("tag", "asset", "itemReceipt", "batchReceipt")
        )
    ):
        raise RetentionError("recording tombstone verification evidence is invalid")
    artifacts = raw.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RetentionError("recording tombstone artifacts are invalid")
    for artifact in artifacts:
        if (
            not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str)
            or not Path(artifact["path"]).is_absolute()
            or not isinstance(artifact.get("bytes"), int) or artifact["bytes"] <= 0
            or not isinstance(artifact.get("sha256"), str)
            or not _SHA.fullmatch(artifact["sha256"])
        ):
            raise RetentionError("recording tombstone artifact is invalid")
    return raw


def tombstone_covers(
    path: Path, size: int, digest: str, tombstone_root: Path,
) -> bool:
    """Return true only for an absent path covered by exact durable metadata."""
    absolute = str(Path(os.path.abspath(path)))
    if path.exists() or path.is_symlink() or not tombstone_root.is_dir():
        return False
    for marker in tombstone_root.glob("recording-*.json"):
        try:
            value = _validate_tombstone(_read_json(marker))
        except RetentionError:
            continue
        if any(
            raw.get("path") == absolute and raw.get("bytes") == size and raw.get("sha256") == digest
            for raw in value["artifacts"]
        ):
            return True
    return False


def recover_tombstones(
    tombstone_root: Path, recording_roots: Sequence[Path], *,
    open_writer: Callable[[Path], bool] | None = None,
) -> list[str]:
    """Idempotently finish prepared unlinks after a process crash."""
    open_writer = open_writer or _default_open_writer
    recovered: list[str] = []
    if not tombstone_root.is_dir():
        return recovered
    for marker in sorted(tombstone_root.glob("recording-*.json")):
        value = _validate_tombstone(_read_json(marker))
        if value["state"] == "pruned":
            continue
        source = _absolute_inside(Path(value["sourcePath"]), recording_roots)
        closure = _absolute_inside(Path(value["closureReceipt"]["path"]), recording_roots)
        if (
            not source.is_file() or source.is_symlink()
            or _sha256(source) != value["sourceSha256"]
            or not closure.is_file() or closure.is_symlink()
            or _sha256(closure) != value["closureReceipt"]["sha256"]
        ):
            raise RetentionError("prepared recording source or closure receipt changed")
        for expected in value["artifacts"]:
            path = _absolute_inside(Path(expected["path"]), recording_roots)
            if not path.exists() and not path.is_symlink():
                continue
            if open_writer(path):
                raise RetentionError(f"prepared artifact has an open writer: {path}")
            if not _matches_fingerprint(path, expected):
                raise RetentionError(f"prepared artifact changed before recovery: {path}")
            path.unlink()
            _fsync_dir(path.parent)
            recovered.append(str(path))
        value["state"] = "pruned"
        value["prunedAt"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(marker, value)
    return recovered


def prune_recording_cache(
    items: Sequence[Any], recording_roots: Sequence[Path], *, budget_bytes: int,
    tombstone_root: Path, catalog_root: Path, receipts: Path,
    site_verify: Callable[[str, int, str], bool],
    github_verify: Callable[[str, str, str], bool],
    site_path: Callable[[Any, str], str],
    open_writer: Callable[[Path], bool] | None = None,
    before_unlink: Callable[[Path], None] = lambda _path: None,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    """Bound eligible local recordings; zero disables pruning."""
    open_writer = open_writer or _default_open_writer
    if budget_bytes < 0:
        raise RetentionError("recording cache budget must not be negative")
    roots = tuple(root.resolve() for root in recording_roots)
    recovered = recover_tombstones(tombstone_root, roots, open_writer=open_writer)
    candidates: list[tuple[Any, list[dict[str, Any]]]] = []
    seen: set[Path] = set()
    total = 0
    for item in items:
        specs = _artifact_specs(item, roots)
        local: list[dict[str, Any]] = []
        for spec in specs:
            path, artifact = spec["path"], spec["artifact"]
            if path in seen:
                raise RetentionError(f"recording artifact is referenced more than once: {path}")
            seen.add(path)
            if path.exists() or path.is_symlink():
                local.append(spec)
                total += artifact.size
            elif not tombstone_covers(path, artifact.size, artifact.sha256, tombstone_root):
                raise RetentionError(f"recording artifact is missing without a tombstone: {path}")
        if local:
            candidates.append((item, local))
    report = {
        "budgetBytes": budget_bytes, "beforeBytes": total, "afterBytes": total,
        "pruned": [], "recovered": recovered, "blocked": [],
        "withinBudget": budget_bytes == 0 or total <= budget_bytes,
        "disabled": budget_bytes == 0,
    }
    if budget_bytes == 0 or total <= budget_bytes:
        return report
    for item, specs in sorted(candidates, key=lambda pair: (pair[0].ended_at, pair[0].item_id)):
        marker: Path | None = None
        removed_count = 0
        try:
            source = _absolute_inside(Path(item.source_path), roots)
            source_details = source.lstat()
            if stat.S_ISLNK(source_details.st_mode) or not stat.S_ISREG(source_details.st_mode):
                raise RetentionError("recording completion manifest is not a regular file")
            if _sha256(source) != item.source_sha256:
                raise RetentionError("recording completion manifest changed")
            _validate_catalog(item, catalog_root)
            closure = _completion_receipt(item, roots)
            fingerprints = []
            for spec in specs:
                if open_writer(spec["path"]):
                    raise RetentionError(f"recording artifact has an open writer: {spec['path']}")
                fingerprints.append(
                    _fingerprint(spec["path"], spec["artifact"].size, spec["artifact"].sha256)
                )
            checked = now().astimezone(timezone.utc).isoformat()
            remote = _remote_evidence(
                item, receipts, site_verify, github_verify, site_path, checked,
            )
            marker = _tombstone_path(tombstone_root, item)
            value = {
                "schema": TOMBSTONE_SCHEMA, "state": "prepared", "preparedAt": checked,
                "kind": "recording", "itemId": item.item_id, "sourceId": item.source_id,
                "sourcePath": str(source), "sourceSha256": item.source_sha256,
                "artifacts": fingerprints, "closureReceipt": closure, "remote": remote,
            }
            _atomic_json(marker, value)
            for expected in fingerprints:
                path = Path(expected["path"])
                before_unlink(path)
                if open_writer(path):
                    raise RetentionError(f"recording artifact acquired an open writer: {path}")
                if not _matches_fingerprint(path, expected):
                    raise RetentionError(f"recording artifact changed before unlink: {path}")
                path.unlink()
                _fsync_dir(path.parent)
                removed_count += 1
                report["pruned"].append(expected)
                total -= expected["bytes"]
            value["state"] = "pruned"
            value["prunedAt"] = now().astimezone(timezone.utc).isoformat()
            _atomic_json(marker, value)
        except (OSError, RetentionError) as exc:
            if marker is not None and removed_count == 0 and marker.exists():
                marker.unlink()
                _fsync_dir(marker.parent)
            report["blocked"].append({"path": str(item.source_path), "error": str(exc)})
        if total <= budget_bytes:
            break
    report["afterBytes"] = total
    report["withinBudget"] = total <= budget_bytes
    return report


def prune_release_cache(
    release_root: Path, receipts: Path, *, budget_bytes: int,
    github_verify: Callable[[str, str, str], bool],
) -> dict[str, Any]:
    """Bound verified local GitHub release archives; zero disables cleanup."""
    if budget_bytes < 0:
        raise RetentionError("release cache budget must not be negative")
    absolute_root = Path(os.path.abspath(release_root))
    if absolute_root.is_symlink():
        raise RetentionError("release cache root may not be a symlink")
    root = absolute_root.parent.resolve() / absolute_root.name
    if root.exists() and not root.is_dir():
        raise RetentionError("release cache root is not a safe directory")
    candidates: list[tuple[Path, os.stat_result]] = []
    total = 0
    if root.is_dir() and not root.is_symlink():
        for path in sorted(root.glob("*.tar.gz")):
            if path.parent != root or not _RELEASE_ARCHIVE.fullmatch(path.name):
                continue
            details = path.lstat()
            total += details.st_size
            candidates.append((path, details))
    report = {
        "budgetBytes": budget_bytes, "beforeBytes": total, "afterBytes": total,
        "removed": [], "blocked": [], "withinBudget": budget_bytes == 0 or total <= budget_bytes,
        "disabled": budget_bytes == 0,
    }
    if budget_bytes == 0 or total <= budget_bytes:
        return report
    verified: list[
        tuple[float, str, Path, os.stat_result, str, list[tuple[Path, str]], Path]
    ] = []
    for path, details in candidates:
        try:
            if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
                raise RetentionError("release cache entry is not a regular non-symlink file")
            digest = _sha256(path)
            batch_path = receipts / "github-batches" / f"{path.name}.verified.json"
            batch = _read_json(batch_path)
            tag, asset, expected_sha = batch.get("tag"), batch.get("asset"), batch.get("sha256")
            if (
                batch.get("schema") != "jev-nethack-github-publish-receipt/v1"
                or batch.get("verified") is not True or asset != path.name
                or not isinstance(tag, str) or not tag
                or not isinstance(expected_sha, str) or not _SHA.fullmatch(expected_sha)
                or digest != expected_sha
            ):
                raise RetentionError("release archive lacks an exact verified batch receipt")
            raw_items = batch.get("items")
            if not isinstance(raw_items, list) or not raw_items:
                raise RetentionError("release batch receipt has no item bindings")
            receipt_snapshots = [(batch_path, _sha256(batch_path))]
            for raw in raw_items:
                if (
                    not isinstance(raw, dict) or raw.get("kind") not in ("recording", "training")
                    or not isinstance(raw.get("id"), str) or not _SITE_ID.fullmatch(raw["id"])
                    or not isinstance(raw.get("sourceSha256"), str)
                    or not _SHA.fullmatch(raw["sourceSha256"])
                ):
                    raise RetentionError("release batch contains an invalid item binding")
                item_path = (
                    receipts / "github-items" / f"{raw['kind']}-{raw['id']}.verified.json"
                )
                item = _read_json(item_path)
                if (
                    item.get("schema") != "jev-nethack-github-item-receipt/v1"
                    or item.get("verified") is not True
                    or item.get("sourceSha256") != raw["sourceSha256"]
                    or item.get("tag") != tag or item.get("asset") != asset
                    or item.get("archiveSha256") != expected_sha
                ):
                    raise RetentionError("release item receipt does not bind the verified batch")
                receipt_snapshots.append((item_path, _sha256(item_path)))
            pending = receipts / "outbox" / f"github-{asset}.json"
            if pending.exists() or pending.is_symlink():
                raise RetentionError("release archive still has a pending publish operation")
            try:
                remote_verified = github_verify(tag, asset, expected_sha)
            except Exception as exc:
                raise RetentionError("fresh GitHub release verification is unavailable") from exc
            if not remote_verified:
                raise RetentionError("fresh GitHub release verification failed")
            try:
                verified_epoch = datetime.fromisoformat(
                    str(batch.get("verifiedAt")).replace("Z", "+00:00")
                ).timestamp()
            except ValueError:
                verified_epoch = details.st_mtime
            verified.append(
                (
                    verified_epoch, path.name, path, details, digest,
                    receipt_snapshots, pending,
                )
            )
        except (OSError, RetentionError) as exc:
            report["blocked"].append({"path": str(path), "error": str(exc)})
    for _epoch, _name, path, expected, digest, receipt_snapshots, pending in sorted(verified):
        if total <= budget_bytes:
            break
        current = path.lstat()
        if pending.exists() or pending.is_symlink() or any(
            receipt.is_symlink() or not receipt.is_file() or _sha256(receipt) != expected_receipt
            for receipt, expected_receipt in receipt_snapshots
        ):
            report["blocked"].append(
                {"path": str(path), "error": "release verification state changed before cleanup"}
            )
            continue
        if (
            not stat.S_ISREG(current.st_mode) or stat.S_ISLNK(current.st_mode)
            or current.st_dev != expected.st_dev or current.st_ino != expected.st_ino
            or current.st_size != expected.st_size or current.st_mtime_ns != expected.st_mtime_ns
            or _sha256(path) != digest
        ):
            report["blocked"].append(
                {"path": str(path), "error": "release archive changed before cleanup"}
            )
            continue
        path.unlink()
        _fsync_dir(root)
        total -= expected.st_size
        report["removed"].append(
            {"path": str(path), "bytes": expected.st_size, "sha256": digest}
        )
    report["afterBytes"] = total
    report["withinBudget"] = total <= budget_bytes
    return report
