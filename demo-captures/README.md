# Sentinel demo captures (auto)

Flag-file gated screenshot cron for accumulating demo material passively.

## Toggle

- **Start capturing:** `New-Item -Force .enabled` (in this dir)
- **Stop capturing:** `Remove-Item .enabled`

Scheduled task `Sentinel-Demo-Snap` fires every 2 min. When the `.enabled` flag
is present, it saves `sentinel-YYYYMMDD-HHmmss.png` (primary monitor).

Auto-prunes files older than 7 days on each run.

Argo owns the task; edit via `tools/sentinel-snap.ps1` in the Argo workspace.
