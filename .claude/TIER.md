# Tier

**TIER: L1**

The tier is the whole ruleset — it sets Claude's autonomy AND the engineering rigor. One line
above; switch it with `/set-tier` (which walks the implications) rather than editing by hand.

| Tier | Autonomy | Rigor | Typical use |
| --- | --- | --- | --- |
| **L1** | Highest — builds phase after phase without waiting; interviews you for design and features, decides the rest and logs it | Lean. Local-only posture, anti-overbuild | Starting a passion project from scratch |
| **L2** | Middle — phase plans wait for sign-off; decisions with real reversal cost stop | Beta. Pre-deployment hardening | Refining an app that still needs real progress |
| **L3** | Careful — architecture, schema, deps, auth, infra, prod data all stop for you | Production | Deployed apps that matter |

Rough mapping: L1 = build it. L2 = get it ready. L3 = don't break it.

**Switching is expected**, not exceptional — a passion project graduates L1 → L2 → L3.
Upgrading adds a hardening phase to the roadmap (auth, secrets, tests, CI gaps); `/set-tier`
proposes it. Downgrading an app that's already deployed doesn't remove the care its users need —
that's why L3 → anything requires a conscious conversation.

## Exceptions

Per-axis overrides, one line each, agreed with the user and logged in the Decision Log.
This is the escape hatch, not a second dial — if this list grows past ~3 lines, the tier is wrong.

<!-- Example:  L3, but decide-and-log autonomy for isolated UI work in frontend/src/components/ -->
