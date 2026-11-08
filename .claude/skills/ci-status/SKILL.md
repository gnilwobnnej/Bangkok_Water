---
name: ci-status
description: Summarise the latest GitHub Actions runs of this repo's two workflows (Tests, Update data), with the failure reason of any failed run. Use when the user asks how CI or the daily data update is doing.
allowed-tools: Bash(gh run list *), Bash(gh run view *)
---
Report on both workflows of this repo. Read-only: don't re-run, cancel or change anything.

1. `gh run list --workflow update-data.yml --limit 7` and `gh run list --workflow ci.yml --limit 5`.
2. For each **failed** run among those, `gh run view <id>` to get the failing step and its annotations. The
   "Report a radar failure" / "Report a Traffy failure" steps put the error's last line in an annotation. Use
   `gh run view <id> --log-failed | tail -40` only if the annotation doesn't explain it.
3. Note when the latest Update data run is more than ~26 hours old (the schedule is daily at 02:30 UTC; GitHub
   often starts it hours late), or still running.

Reply with:
- one line per workflow: latest result, when it ran, and the streak (e.g. "3 green in a row");
- for each distinct failure cause: run ids, failing step, the decisive log line, and whether a later run fixed it;
- a suggestion to hand persistent Update data failures to the `pipeline-doctor` subagent.

Keep it under 15 lines.
