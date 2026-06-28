# Contributing to Sentinel v1

Thanks for your interest in contributing! A few things to read first.

## Ground rules
- Be respectful — see [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
- Don't include anything that violates the
  [Acceptable Use Policy](docs/legal/ACCEPTABLE_USE_POLICY.md).
- Don't paste secrets, API keys, or real incident/personal data in issues, PRs, or fixtures.
- For security issues, **do not** open a public issue — use a private
  [security advisory](https://github.com/BenDuske/sentinel-v1/security/advisories/new)
  (see [`SECURITY.md`](SECURITY.md)).

## Developer Certificate of Origin (DCO)
Every commit must be signed off, certifying you wrote the code or have the right to submit it
under this project's license (the [DCO 1.1](https://developercertificate.org/)). Add a sign-off
line to each commit:

```
Signed-off-by: Your Name <your-email@example.com>
```

(use `git commit -s`). PRs without sign-off may be asked to amend.

## Contributor License Agreement (CLA) — why we ask
Sentinel is **open-core**: the code here is MIT-licensed and free, while the author
(Ben Duske) also offers separate **commercial/proprietary editions**. To keep that model
clean and lawful, contributions are accepted under the following inbound terms:

> By submitting a contribution (a pull request or patch), you grant **Ben Duske** a
> **perpetual, worldwide, non-exclusive, royalty-free, irrevocable license** to use,
> reproduce, modify, sublicense, and distribute your contribution — including the right to
> **relicense it under different terms** (such as in a commercial edition) and under the
> project's open-source license. You confirm you are legally entitled to grant this license,
> and that your contribution is your original work or you have the necessary rights to it.

This is a standard arrangement for open-core projects. It does **not** strip your rights to
your own code — you keep the copyright to your contribution; you simply also grant the author
the broad license above so the project can ship in both free and paid forms. If your employer
owns your work, get permission before contributing.

A formal signed CLA may be required for substantial contributions; see `[CLA URL]` *(to be
published)*. For small fixes, the DCO sign-off plus this notice applies.

## Workflow
1. Open an issue describing the change (use the templates) before large work.
2. Fork, branch, make the change with tests.
3. Run the suite: `python -m pytest -q` (the core policy tests run keyless / offline).
4. Open a PR with a clear description and a DCO `Signed-off-by` line.

By contributing you also agree your contribution is provided under the project's MIT
`LICENSE` to all downstream users.
