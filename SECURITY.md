# Security policy

Please report suspected vulnerabilities privately, not in a public issue.

## How to report

Use GitHub's private vulnerability reporting: on the repository page, open **Security →
Report a vulnerability**. Include the VectorView version, steps to reproduce and what you expected
to happen. Never include API keys.

You can expect an acknowledgement within 5 business days and an assessment within 15 business
days. Fixes are released as a patch version with a CHANGELOG entry and a GitHub security advisory.
Reporters are credited unless they prefer otherwise.

## What counts as a vulnerability

- API keys leaking into logs, reprs, error messages, the web pages or requests to the wrong service;
- the web app reading or serving files outside its data directory;
- a crafted request, LLM reply or VLM reply that makes VectorView execute code or SQL;
- a way around the per-key image cap of the quota ledger.

## Known limitations (not vulnerabilities)

- The web app has no authentication and no CSRF protection. It is meant for one user on their own
  machine and binds to 127.0.0.1 by default; do not expose it to a network.
- Query text is sent to the LLM, and images to the VLM provider you choose.
- Images and LLM output are untrusted; a crafted image could try to steer the VLM judge.

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | yes |
