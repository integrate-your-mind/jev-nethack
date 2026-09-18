"""Small, bounded TypeSafe Jev client used by the NetHack prototype.

The client deliberately has no retry policy.  A call reservation is made before
opening the connection, so transport, HTTP, and response failures consume the
same call and input-byte budget as successful calls.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any


ENDPOINT = "https://api.typesafe.ai/v1/systemone"
PINNED_MODEL = "jev-1.13.0"
# Keep an individual request bounded even when a caller gives the client a much
# larger cumulative budget.
MAX_REQUEST_BYTES = 24 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024
# TypeSafe choice probabilities are emitted on a rounded .01 grid.  Preserve
# the raw distribution, while allowing the documented response rounding band.
PROBABILITY_SUM_TOLERANCE = 0.02
PROBABILITY_EPSILON = 1e-9


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Prevent urllib from forwarding the bearer credential to another host."""

    def redirect_request(
        self,
        req: Any,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirectHandler())


def _default_opener(request: urllib.request.Request, *, timeout: float) -> Any:
    return _NO_REDIRECT_OPENER.open(request, timeout=timeout)


class JevClientError(Exception):
    """Base class for safe, user-facing Jev client failures."""


class JevConfigurationError(JevClientError):
    """The client was constructed with invalid or missing configuration."""


class JevBudgetExceededError(JevClientError):
    """The configured call or serialized-input budget is exhausted."""


class JevPayloadError(JevClientError):
    """The local request could not be represented as a bounded JSON payload."""


class JevTransportError(JevClientError):
    """The request could not be completed at the transport layer."""


class JevHTTPError(JevClientError):
    """TypeSafe returned a non-success HTTP status."""

    def __init__(self, status: int):
        self.status = status
        super().__init__(f"TypeSafe request failed with HTTP status {status}")


class JevResponseError(JevClientError):
    """TypeSafe returned malformed or semantically invalid JSON."""


# Short aliases make it convenient for callers to catch the broad protocol or
# budget categories without sacrificing the more specific exception classes.
JevError = JevClientError
JevBudgetError = JevBudgetExceededError
JevProtocolError = JevResponseError


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_nonnegative_int(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise JevConfigurationError(f"{name} must be a non-negative integer")
    return value


class JevClient:
    """Bounded stdlib client for the TypeSafe SystemOne choice endpoint.

    ``opener`` is injectable to keep tests offline.  It must have the same
    calling convention as :func:`urllib.request.urlopen`: ``opener(request,
    timeout=seconds)`` and return an object with ``read()`` and optionally
    ``status``/``getcode()`` and ``close()``.
    """

    def __init__(
        self,
        max_calls: int,
        max_input_bytes: int,
        timeout: float = 20,
        *,
        api_key: str | None = None,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self.max_calls = _require_nonnegative_int("max_calls", max_calls)
        self.max_input_bytes = _require_nonnegative_int(
            "max_input_bytes", max_input_bytes
        )
        if not _is_number(timeout) or not math.isfinite(float(timeout)) or timeout <= 0:
            raise JevConfigurationError("timeout must be a positive finite number")
        self.timeout = float(timeout)

        configured_key = os.environ.get("TYPESAFE_API_KEY") if api_key is None else api_key
        if not isinstance(configured_key, str) or not configured_key.strip():
            raise JevConfigurationError("TYPESAFE_API_KEY is required")
        self._api_key = configured_key.strip()
        self._opener = _default_opener if opener is None else opener

        self._calls_used = 0
        self._input_bytes_used = 0
        self._lock = threading.Lock()

    @property
    def calls_used(self) -> int:
        with self._lock:
            return self._calls_used

    @property
    def input_bytes_used(self) -> int:
        with self._lock:
            return self._input_bytes_used

    def choose(
        self,
        state: Mapping[str, Any],
        criteria: Mapping[str, str | None],
        instructions: str,
    ) -> dict[str, Any]:
        """Ask Jev to choose one action and return validated telemetry.

        The request is attempted at most once.  ``attempts_total`` is therefore
        always one for a successful result; failed calls are still reserved in
        the client's budgets.
        """

        payload = self._build_payload(state, criteria, instructions)
        try:
            serialized = json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError):
            raise JevPayloadError("request state is not JSON serializable") from None

        serialized_size = len(serialized)
        if serialized_size > MAX_REQUEST_BYTES:
            raise JevPayloadError(
                f"serialized request exceeds {MAX_REQUEST_BYTES} byte per-request cap"
            )
        self._reserve(serialized_size)

        digest = hashlib.sha256(serialized).hexdigest()
        request = urllib.request.Request(
            ENDPOINT,
            data=serialized,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        started = time.monotonic()
        raw_response = self._perform_request(request)
        response = self._decode_response(raw_response)
        answer, input_tokens = self._validate_response(response, criteria)
        latency_seconds = time.monotonic() - started

        probabilities = {
            key: float(answer["probabilities"][key]) for key in criteria
        }
        return {
            "choice": answer["choice"],
            "confidence": float(answer["confidence"]),
            "probabilities": probabilities,
            "probability_sum": float(answer["probability_sum"]),
            "model": PINNED_MODEL,
            "usage": {"input_tokens": input_tokens},
            "latency_seconds": latency_seconds,
            "request_digest": digest,
            # These are the exact bounded JSON bodies, excluding HTTP headers
            # and therefore excluding the bearer credential. Keeping them in
            # local training records makes every model decision reproducible
            # and auditable without trusting a later reconstruction.
            "request_body_base64": base64.b64encode(serialized).decode("ascii"),
            "response_body_base64": base64.b64encode(raw_response).decode("ascii"),
            "response_digest": hashlib.sha256(raw_response).hexdigest(),
            "attempts_total": 1,
            "calls_used": self.calls_used,
            "input_bytes_used": self.input_bytes_used,
        }

    @staticmethod
    def _build_payload(
        state: Mapping[str, Any],
        criteria: Mapping[str, str | None],
        instructions: str,
    ) -> dict[str, Any]:
        if not isinstance(state, Mapping):
            raise JevPayloadError("state must be a JSON object")
        if not isinstance(criteria, Mapping) or not criteria:
            raise JevPayloadError("criteria must be a non-empty mapping")
        if not isinstance(instructions, str) or not instructions.strip():
            raise JevPayloadError("instructions must be a non-empty string")

        normalized_criteria: dict[str, str | None] = {}
        for action_id, description in criteria.items():
            if not isinstance(action_id, str) or not action_id:
                raise JevPayloadError("criteria keys must be non-empty strings")
            if description is not None and not isinstance(description, str):
                raise JevPayloadError("criteria descriptions must be strings or null")
            normalized_criteria[action_id] = description

        return {
            "model": PINNED_MODEL,
            "state": dict(state),
            "questions": {
                "action": {
                    "type": "choice",
                    "instructions": instructions,
                    "criteria": normalized_criteria,
                }
            },
        }

    def _reserve(self, serialized_size: int) -> None:
        with self._lock:
            if self._calls_used >= self.max_calls:
                raise JevBudgetExceededError("maximum Jev call budget exhausted")
            if self._input_bytes_used + serialized_size > self.max_input_bytes:
                raise JevBudgetExceededError("maximum serialized input-byte budget exhausted")
            # Reserve before opening the connection.  Any later failure counts.
            self._calls_used += 1
            self._input_bytes_used += serialized_size

    def _perform_request(self, request: urllib.request.Request) -> bytes:
        response: Any = None
        try:
            response = self._opener(request, timeout=self.timeout)
            status = self._response_status(response)
            if status is not None and status >= 400:
                raise JevHTTPError(status)
            try:
                body = response.read(MAX_RESPONSE_BYTES + 1)
            except TypeError:
                # A tiny test double may expose read() without urllib's optional
                # size argument.  Enforce the cap after reading in that case.
                body = response.read()
            if not isinstance(body, bytes):
                raise JevTransportError("TypeSafe response was not bytes")
            if len(body) > MAX_RESPONSE_BYTES:
                raise JevResponseError("TypeSafe response exceeds the response-size cap")
            return body
        except JevClientError:
            raise
        except urllib.error.HTTPError as exc:
            # Never expose the HTTP error body: it can contain server details.
            status = exc.code if isinstance(exc.code, int) else 0
            raise JevHTTPError(status) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise JevTransportError("TypeSafe request could not be completed") from None
        except Exception:
            # Keep injected transports and unexpected library failures opaque.
            raise JevTransportError("TypeSafe request could not be completed") from None
        finally:
            if response is not None:
                close = getattr(response, "close", None)
                if callable(close):
                    close()

    @staticmethod
    def _response_status(response: Any) -> int | None:
        status = getattr(response, "status", None)
        if status is None:
            getcode = getattr(response, "getcode", None)
            if callable(getcode):
                status = getcode()
        if status is None:
            return None
        if not isinstance(status, int) or isinstance(status, bool):
            raise JevTransportError("TypeSafe response had an invalid HTTP status")
        return status

    @staticmethod
    def _decode_response(raw_response: bytes) -> Any:
        try:
            return json.loads(raw_response.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
            raise JevResponseError("TypeSafe response was not valid JSON") from None

    @staticmethod
    def _validate_response(
        response: Any, criteria: Mapping[str, str | None]
    ) -> tuple[dict[str, Any], int]:
        if not isinstance(response, Mapping):
            raise JevResponseError("TypeSafe response must be a JSON object")
        if response.get("model") != PINNED_MODEL:
            raise JevResponseError("TypeSafe response used an unexpected model")

        answers = response.get("answers")
        if not isinstance(answers, Mapping):
            raise JevResponseError("TypeSafe response is missing answers")
        answer = answers.get("action")
        if not isinstance(answer, Mapping) or answer.get("type") != "choice":
            raise JevResponseError("TypeSafe response is missing a choice answer")

        choice = answer.get("choice")
        if not isinstance(choice, str) or choice not in criteria:
            raise JevResponseError("TypeSafe returned an action outside the criteria")

        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, Mapping):
            raise JevResponseError("TypeSafe choice is missing probabilities")
        if set(probabilities) != set(criteria):
            raise JevResponseError("TypeSafe probabilities do not match criteria")

        checked_probabilities: dict[str, float] = {}
        for action_id in criteria:
            value = probabilities[action_id]
            if not _is_number(value) or not math.isfinite(float(value)):
                raise JevResponseError("TypeSafe probabilities must be finite numbers")
            numeric_value = float(value)
            if numeric_value < 0 or numeric_value > 1:
                raise JevResponseError("TypeSafe probabilities must be between zero and one")
            checked_probabilities[action_id] = numeric_value
        probability_sum = math.fsum(checked_probabilities.values())
        if abs(probability_sum - 1.0) > PROBABILITY_SUM_TOLERANCE + PROBABILITY_EPSILON:
            raise JevResponseError("TypeSafe probabilities are outside the rounding tolerance")
        max_probability = max(checked_probabilities.values())
        if (
            abs(checked_probabilities[choice] - max_probability)
            > PROBABILITY_EPSILON
        ):
            raise JevResponseError("TypeSafe choice is not the highest-probability action")

        confidence = answer.get("confidence")
        if (
            not _is_number(confidence)
            or not math.isfinite(float(confidence))
            or float(confidence) < 0
            or float(confidence) > 1
        ):
            raise JevResponseError("TypeSafe confidence must be a finite number between zero and one")

        usage = response.get("usage")
        input_tokens = usage.get("input_tokens") if isinstance(usage, Mapping) else None
        if (
            not isinstance(input_tokens, int)
            or isinstance(input_tokens, bool)
            or input_tokens < 0
        ):
            raise JevResponseError("TypeSafe usage is missing input_tokens")

        return (
            {
                "choice": choice,
                "confidence": float(confidence),
                "probabilities": checked_probabilities,
                "probability_sum": probability_sum,
            },
            input_tokens,
        )


__all__ = [
    "ENDPOINT",
    "MAX_REQUEST_BYTES",
    "PROBABILITY_EPSILON",
    "PROBABILITY_SUM_TOLERANCE",
    "PINNED_MODEL",
    "JevBudgetError",
    "JevBudgetExceededError",
    "JevClient",
    "JevClientError",
    "JevConfigurationError",
    "JevError",
    "JevHTTPError",
    "JevPayloadError",
    "JevProtocolError",
    "JevResponseError",
    "JevTransportError",
]
