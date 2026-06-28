# Before You Use — Agreement Required

Sentinel v1 asks you to read and agree to a short set of terms **before first use**. The web UI
shows an "I agree" acknowledgment before you can log incidents, and the server logs a notice
pointing here at startup (see the consent gate in `sentinel.policy`).

By installing and running this Software, you confirm that you have read and agree to:

1. **MIT License** — [`LICENSE`](LICENSE)
2. **Terms of Service** — [`docs/legal/TERMS_OF_SERVICE.md`](docs/legal/TERMS_OF_SERVICE.md)
3. **Acceptable Use Policy** — [`docs/legal/ACCEPTABLE_USE_POLICY.md`](docs/legal/ACCEPTABLE_USE_POLICY.md)
4. **Privacy Policy** — [`docs/legal/PRIVACY_POLICY.md`](docs/legal/PRIVACY_POLICY.md)
5. **AI Output & Warranty Disclaimer** — [`docs/legal/DISCLAIMER.md`](docs/legal/DISCLAIMER.md)

You specifically acknowledge that:

- ✅ You will use the Software **only for lawful purposes**, complying with **U.S. federal law
  and the laws of your own state and locality** (and any other law that applies to you).
- ✅ You will **not** use it to generate sexual content, nudity, explicit adult role-play, or
  any sexualization of minors, or for any other use prohibited by the Acceptable Use Policy.
  (Documenting real incidents — including violence, injury, theft, fire, and other crime — is
  the legitimate purpose and is fully supported.)
- ✅ You understand the **AI output may be wrong** and is **decision-support, not an
  underwriting or safety determination and not professional advice**; a human reviews and owns
  every consequential decision.
- ✅ The Software is provided **"as is", with no warranty**, and your use is at your own risk.
  Your incident data and uploaded evidence stay **local on your machine**.

**How acceptance is recorded:** the web dashboard shows a summary and an "I agree" checkbox you
must accept before submitting incidents (remembered in your browser's localStorage). For
CLI/automation use, `sentinel.policy.ensure_consent()` records acceptance (with timestamp and
policy version) to a local file in your Sentinel data directory (next to the SQLite store, or
`~/.sentinel`). You can withdraw consent by stopping use and deleting that file, your incident
store, and your evidence directory. Set `SENTINEL_ASSUME_CONSENT=1` only in automated/CI
contexts where you have already accepted these terms on behalf of the operator.
