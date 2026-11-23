Review the freeze package for the task {{TASK}}. The gate is {{GATE}} and this is review round {{ROUND}}.

Once this package is frozen, the test seeds are drawn from a public randomness beacon and every arm runs once. Nothing can be changed afterwards. Your review is the last look before that.

{{PREVIOUS}}

Read these first
- SPEC.md, the sections "Rules of the game", "Stage 3", "Stage 4", "Controls and simple baselines" and "Statistics"
- referee/RULES.md
- experiments/{{TASK}}/hypotheses.md
- everything in experiments/{{TASK}}/freeze/
- experiments/{{TASK}}/stage3/, including the validation files
- DEVIATIONS.md and the tail of PROGRESS.md

Then try to refute each of these
1. The edit matches its written hypothesis. Does the head atlas in stage2/ support the story, and does the dose response move the tracking number in step?
2. The validation evidence is what the agent says it is. Run `python -m referee.recompute validation --task {{TASK}}`. If the edit missed the validation rule, the package must say so plainly.
3. The validation seeds were not overused. Run `python -m referee.recompute ledger`. Was any candidate tuned after seeing its validation result?
4. The simple baselines got a fair fight. Were they tuned on the same design seeds with comparable effort? Is any family set up to lose, for example with a narrow search range?
5. The random-edit rule in random_family.json is a fair match in count, layer type and scale, and it was written before any random edit was evaluated.
6. The auto-greedy edit was built by a script from the shared sweep, with no hand tuning.
7. No gradient or optimiser over more than 3 free numbers was used anywhere. Look at the code in awe/ and any scripts under experiments/.
8. The test manifest runs what the spec lists, and each arm's recipe is fully determined by files in this commit.
9. Nothing in DEVIATIONS.md changes a hypothesis, a pass rule, a seed set or a budget.

Do not judge whether the edit will win. A weak edit that followed the rules passes review. Return verdict "revise" only if a blocking issue remains. Otherwise return "pass".
