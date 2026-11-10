# Tier

**TIER: L1**

Git policy: main
PM mode: private
Deploys on push: no

The tier sets autonomy and rigor together; its ruleset is imported below. The git policy and
`Deploys on push` are read by `.claude/hooks/guard_commands.py`. Change any of these with
`/set-tier`, and the PM mode with `/pm-upgrade` — never by hand (CURRENT.md mirrors them).

@tiers/L1.md

## Exceptions

Per-axis overrides, one line each, agreed with the user and logged in the Decision Log. This is
the escape hatch, not a second dial: past ~3 lines the tier is wrong. `pm_check ignore:` lines
mark files that code depends on, so the PM checker leaves them alone.

<!-- - L3, but decide-and-log autonomy for isolated UI work in frontend/src/components/ -->
<!-- - pm_check ignore: project-management/reports/*_data-quality.md (written by src/cli.py) -->
