# Security policy

## Reporting

Email security reports to smigolsmigol@protonmail.com. Do not open public issues for vulnerabilities.

Acknowledgment within 48 hours. Critical issues patched within 7 days.

## Supported versions

| Version | Supported |
|---------|-----------|
| latest  | Yes       |
| older   | No        |

## Architecture notes

pydantic-cal is pure Python. No network IO, no `eval`, no `exec`, no subprocess calls. Runtime dependencies are limited to numpy and pydantic; matplotlib and pydantic-evals are optional extras.

The calibration code path operates on numpy arrays of confidences and labels passed in by the caller. There is no deserialization of untrusted input inside the library.
