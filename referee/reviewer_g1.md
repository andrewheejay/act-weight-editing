Review Gate 1 for the task {{TASK}}. This is review round {{ROUND}}.

Gate 1 fixes three things before any sweep runs. They are the diagnosis of why the policy fails, the primary score, and the operating point. After this gate they cannot change, so this is the moment to catch a wrong turn.

{{PREVIOUS}}

Read these first
- SPEC.md, the sections "What we are testing", "Measurements" and "Stage 1"
- experiments/{{TASK}}/diagnosis.md
- experiments/{{TASK}}/stage1/decisions.json
- the per-seed files in experiments/{{TASK}}/stage1/
- DEVIATIONS.md and the tail of PROGRESS.md

Then try to refute each of these
1. The named target failure is the largest fixable group of failures. Recompute the failure classes from design_default.jsonl. Is another class bigger, or more clearly a near miss?
2. The tracking number really tracks that failure. Does it separate failing seeds from passing seeds in the data, or is it asserted?
3. The near-miss evidence is real. Is it measured across many seeds, or shown with a few hand-picked episodes?
4. The primary score and operating point follow the rules in SPEC.md. Run `python -m referee.recompute decisions --task {{TASK}}` and compare.
5. The strict handover rule is implemented as SPEC.md defines it. If the agent changed the definition, is the change logged in DEVIATIONS.md with a reason, and does it make the edit's job easier?
6. Nothing in DEVIATIONS.md changes a hypothesis, a pass rule, a seed set or a budget.
7. No validation seed was used yet. Run `python -m referee.recompute ledger`.

Return verdict "revise" only if a blocking issue remains. Otherwise return "pass".
