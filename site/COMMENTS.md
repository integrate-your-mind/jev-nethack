# Viewer comments

The public viewer has a plain-text discussion area. Display names are self-reported and unverified. Comments are separate from game recordings, decisions, and training data; they are never passed to the playing agent.

- Read comments: `GET /api/comments` (optional returned `cursor` for another page).
- Post a comment: same-origin `POST /api/comments` with JSON `displayName`, `body`, and an empty optional `website` honeypot field.
- Moderate one comment: `DELETE /api/comments/<id>` using the existing private ingest bearer credential. Never put this credential in browser code, comments, source control, or a public URL.

Names are limited to 40 characters and comments to 2,000 characters. Posting is limited to three accepted attempts per network address per fixed ten-minute window; people on a shared network share that limit. The server validates request size and origin and stores comments in a separate R2 namespace. Rate-limit keys contain a salted address hash, not the raw address. The browser renders visitor text using text nodes, without interpreting HTML or Markdown. Successful posts are public; visitors should not include private information.

The comment service does not require TypeSafe credits. Paused gameplay and comment availability are independent.

Run the API and viewer regression suite with `npm test`. Deployment verification should use one clearly labelled temporary system test comment, verify readback, delete only that test comment with the owner credential, and confirm its absence.
