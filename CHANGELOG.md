## [unreleased]

### 💼 Other

- CSPRNG randomness with verifiable commit-reveal
- Preflight secrets check -- fail loudly on missing/malformed AWS_REGION or AWS_INSTANCE_ID
- Use renamed repo URL (Entropy-MCP)
- Stamp-mcp is on PyPI -- drop git+ URL (slim image has no git)

### ⚙️ Miscellaneous Tasks

- Auto-update CHANGELOG.md via shared git-cliff template
- Trigger changelog pipeline
- Inline changelog job (debug)
- Debug pipeline vars (temporary)
- Restore shared changelog include
- Inline changelog job (git-cliff, commits back on merge)
- Bisect pipeline-empty cause
- Test solo .post+rules job
- Changelog job on a real stage (.post-only pipelines never run)
