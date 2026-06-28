# Terms of Service / End-User License Terms

> **DRAFT — pending attorney review.** Good-faith terms for the open-source release; **not
> legal advice.** The binding open-source grant is the **MIT `LICENSE`** in the repository
> root. These terms add use conditions and disclaimers; where they appear to conflict with the
> MIT License's grant of rights, the MIT License controls for the open-source distribution.

**Software:** Sentinel v1 ("the Software"), local AI risk & incident intelligence.
**Author / Licensor:** Ben Duske ("the Author").
**Last updated:** 2026-06-28.

## 1. Acceptance
By installing, copying, running, or modifying the Software, you ("you" / "the User") agree to
these Terms, the **Acceptable Use Policy** (`ACCEPTABLE_USE_POLICY.md`), and the **Disclaimer**
(`DISCLAIMER.md`). If you do not agree, do not install or use the Software.

## 2. License
The Software is licensed, not sold, under the **MIT License** (see `LICENSE`). That license
grants you broad rights to use, copy, modify, and distribute, subject to its notice
requirement. These Terms describe the conditions and limitations under which you exercise
those rights and the disclaimers that accompany them.

## 3. Acceptable use
Your use must comply with the **Acceptable Use Policy**, including the requirement to obey all
applicable **U.S. federal** law and **your own state and local** law, and the prohibitions on
sexual/illegal/harmful content. Violation terminates any permission these Terms grant
(the MIT License's own terms continue to govern redistribution).

## 4. AI output — no reliance
The Software produces **AI-generated output (severity scores, summaries, recommended actions)
that may be inaccurate, incomplete, biased, or fabricated.** It is **decision-support, not an
underwriting, claims, or safety determination, and not professional advice of any kind.** You
are solely responsible for reviewing, verifying, and deciding whether to act on any output.
See `DISCLAIMER.md`.

## 5. Your responsibilities
- You provide and run your own LLM endpoint (a local Ollama model by default) and any API keys.
- You are the **controller** of any incident data and evidence you put into the Software and of
  any deployment you run for others; you must meet the privacy/consent obligations that apply.
- You are responsible for securing your machine, keys, the SQLite store, and uploaded evidence.

## 6. Third-party services
The Software calls the LLM endpoint you configure (default: a local Ollama model; optionally any
other OpenAI-compatible endpoint). Your use of any third-party endpoint is governed by **its**
terms and model licenses. The Author is not a party to and is not responsible for those services.

## 7. No warranty
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT, AND
ACCURACY OF AI OUTPUT. See `LICENSE` and `DISCLAIMER.md`.

## 8. Limitation of liability
TO THE MAXIMUM EXTENT PERMITTED BY LAW, THE AUTHOR SHALL NOT BE LIABLE FOR ANY INDIRECT,
INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR EXEMPLARY DAMAGES, OR FOR ANY LOSS OF DATA, PROFITS, OR
GOODWILL, ARISING FROM OR RELATED TO THE SOFTWARE OR ITS USE, EVEN IF ADVISED OF THE
POSSIBILITY. THE AUTHOR'S TOTAL AGGREGATE LIABILITY SHALL NOT EXCEED THE GREATER OF (a) THE
AMOUNT YOU PAID THE AUTHOR FOR THE SOFTWARE (USD $0 for the open-source distribution) OR
(b) USD $100.

## 9. Indemnification
You agree to indemnify and hold the Author harmless from any claim, loss, or expense
(including reasonable legal fees) arising from your use of the Software, your deployments, your
data, or your violation of these Terms or applicable law.

## 10. Termination
Permission under these Terms ends automatically if you breach them. Sections 4, 7, 8, 9, and 11
survive termination.

## 11. Governing law
These Terms are governed by the laws of the **State of Texas, USA**, without regard to
conflict-of-laws rules, except where a mandatory consumer-protection law of your residence
applies. *(Operator note: a commercial offering should set its own forum/venue here with
counsel.)*

## 12. Changes
The Author may update these Terms; the version in the repository at the time you obtain the
Software governs your copy.
