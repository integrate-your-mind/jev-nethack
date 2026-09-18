"""Offline unit tests for :mod:`jev_client`.

Every transport is an in-memory fake; running this module never contacts
TypeSafe and therefore never creates a paid inference request.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sys
import unittest
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from jev_client import (  # noqa: E402
    PINNED_MODEL,
    JevBudgetExceededError,
    JevClient,
    JevHTTPError,
    JevResponseError,
)


CRITERIA = {"wait": "keep position", "move": "move toward the target"}


def response_body(**answer_overrides: object) -> bytes:
    answer = {
        "type": "choice",
        "choice": "move",
        "probabilities": {"wait": 0.2, "move": 0.8},
        "confidence": 0.75,
    }
    answer.update(answer_overrides)
    return json.dumps(
        {
            "model": PINNED_MODEL,
            "answers": {"action": answer},
            "usage": {"input_tokens": 37, "output_tokens": 11},
        }
    ).encode()


class FakeResponse:
    def __init__(self, body: bytes, status: int = 200):
        self.body = body
        self.status = status
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        return self.body

    def close(self) -> None:
        self.closed = True


class FakeTransport:
    def __init__(self, response: FakeResponse | Exception):
        self.response = response
        self.requests: list[object] = []
        self.timeouts: list[float] = []

    def __call__(self, request: object, *, timeout: float) -> FakeResponse:
        self.requests.append(request)
        self.timeouts.append(timeout)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class JevClientTests(unittest.TestCase):
    def test_valid_response_is_structured_and_request_is_pinned(self) -> None:
        transport = FakeTransport(FakeResponse(response_body()))
        client = JevClient(
            max_calls=2,
            max_input_bytes=10_000,
            timeout=7,
            api_key="test-key",
            opener=transport,
        )

        result = client.choose(
            {"hp": 10, "glyph": "@"}, CRITERIA, "Choose the safest legal action."
        )

        self.assertEqual(result["choice"], "move")
        self.assertEqual(result["confidence"], 0.75)
        self.assertEqual(result["probabilities"], {"wait": 0.2, "move": 0.8})
        self.assertEqual(result["probability_sum"], 1.0)
        self.assertEqual(result["model"], PINNED_MODEL)
        self.assertEqual(result["usage"], {"input_tokens": 37})
        self.assertGreaterEqual(result["latency_seconds"], 0)
        self.assertEqual(result["attempts_total"], 1)
        self.assertEqual(client.calls_used, 1)

        request = transport.requests[0]
        body = request.data  # urllib.request.Request exposes the encoded body.
        self.assertEqual(result["request_digest"], hashlib.sha256(body).hexdigest())
        self.assertEqual(body, base64.b64decode(result["request_body_base64"]))
        self.assertEqual(response_body(), base64.b64decode(result["response_body_base64"]))
        self.assertEqual(result["response_digest"], hashlib.sha256(response_body()).hexdigest())
        self.assertNotIn(b"test-key", base64.b64decode(result["request_body_base64"]))
        self.assertNotIn(b"test-key", base64.b64decode(result["response_body_base64"]))
        decoded = json.loads(body)
        self.assertEqual(decoded["model"], PINNED_MODEL)
        self.assertEqual(decoded["questions"]["action"]["criteria"], CRITERIA)
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(transport.timeouts, [7.0])

    def test_out_of_menu_choice_fails_closed(self) -> None:
        transport = FakeTransport(FakeResponse(response_body(choice="attack_unknown")))
        client = JevClient(1, 10_000, api_key="secret-key", opener=transport)

        with self.assertRaises(JevResponseError):
            client.choose({}, CRITERIA, "Choose an action.")

        self.assertEqual(client.calls_used, 1)

    def test_nonfinite_probability_fails_closed(self) -> None:
        raw = response_body(probabilities={"wait": float("nan"), "move": 0.8})
        transport = FakeTransport(FakeResponse(raw))
        client = JevClient(1, 10_000, api_key="secret-key", opener=transport)

        with self.assertRaises(JevResponseError):
            client.choose({}, CRITERIA, "Choose an action.")

    def test_rounded_probability_sums_are_preserved(self) -> None:
        for probabilities, expected_sum in (
            ({"wait": 0.19, "move": 0.8}, 0.99),
            ({"wait": 0.21, "move": 0.8}, 1.01),
        ):
            with self.subTest(expected_sum=expected_sum):
                transport = FakeTransport(
                    FakeResponse(response_body(probabilities=probabilities))
                )
                client = JevClient(1, 10_000, api_key="secret-key", opener=transport)
                result = client.choose({}, CRITERIA, "Choose an action.")
                self.assertAlmostEqual(result["probability_sum"], expected_sum)
                self.assertEqual(result["probabilities"], probabilities)

    def test_materially_unnormalized_probability_sum_fails_closed(self) -> None:
        transport = FakeTransport(
            FakeResponse(response_body(probabilities={"wait": 0.1, "move": 0.8}))
        )
        client = JevClient(1, 10_000, api_key="secret-key", opener=transport)

        with self.assertRaises(JevResponseError):
            client.choose({}, CRITERIA, "Choose an action.")

    def test_choice_must_be_highest_probability(self) -> None:
        transport = FakeTransport(
            FakeResponse(
                response_body(
                    choice="wait",
                    probabilities={"wait": 0.2, "move": 0.8},
                )
            )
        )
        client = JevClient(1, 10_000, api_key="secret-key", opener=transport)

        with self.assertRaises(JevResponseError):
            client.choose({}, CRITERIA, "Choose an action.")

    def test_failed_call_reservation_exhausts_call_limit(self) -> None:
        transport = FakeTransport(FakeResponse(response_body()))
        client = JevClient(1, 10_000, api_key="secret-key", opener=transport)
        client.choose({}, CRITERIA, "Choose an action.")

        with self.assertRaises(JevBudgetExceededError):
            client.choose({}, CRITERIA, "Choose an action again.")
        self.assertEqual(client.calls_used, 1)
        self.assertEqual(len(transport.requests), 1)

    def test_http_failure_does_not_reveal_body_or_key(self) -> None:
        secret_body = b"internal server detail and secret-key"
        transport = FakeTransport(FakeResponse(secret_body, status=529))
        client = JevClient(1, 10_000, api_key="secret-key", opener=transport)

        with self.assertRaises(JevHTTPError) as raised:
            client.choose({}, CRITERIA, "Choose an action.")

        self.assertEqual(raised.exception.status, 529)
        self.assertNotIn(secret_body.decode(), str(raised.exception))
        self.assertNotIn("secret-key", str(raised.exception))
        self.assertEqual(client.calls_used, 1)

    def test_http_error_exception_body_is_opaque(self) -> None:
        transport = FakeTransport(
            urllib.error.HTTPError(
                "https://api.typesafe.ai/v1/systemone",
                401,
                "unauthorized",
                {},
                None,
            )
        )
        client = JevClient(1, 10_000, api_key="secret-key", opener=transport)

        with self.assertRaises(JevHTTPError) as raised:
            client.choose({}, CRITERIA, "Choose an action.")

        self.assertEqual(raised.exception.status, 401)
        self.assertNotIn("unauthorized", str(raised.exception))
        self.assertEqual(client.calls_used, 1)


if __name__ == "__main__":
    unittest.main()
