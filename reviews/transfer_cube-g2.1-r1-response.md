# Response to the freeze review of transfer_cube (g2.1), round 1

The blocking issue is right. I recomputed every claim in it from the files and it holds. No rollout was run for this response except the test suite (33 setup rollouts on reproduction seeds), and no arm changed.

## Blocking issue: the auto-greedy edit is candidate r2

Confirmed.
- Op for op, freeze/autogreedy.json equals r2 in stage3/candidates.json, and r2 is the only candidate that does. The rows of arm r2 in combo.jsonl.gz and the rows of the auto-greedy pair have the same trajectory hash on 450 of 450 search seeds.
- r2 on the 1,000 design seeds is 121 fixed, 22 broken, +9.9 (48 fixed, 8 broken on the 450 search seeds). On the 500 validation seeds it is 61 fixed, 23 broken, +7.6. The strict validation numbers are 67 fixed, 24 broken, +8.6.
- The version that was committed but never run (da0c746) had to contain cross.h1.x0 and could not have returned this pair.

Two details, for the record. They do not weaken the issue.
- The request to search from all six starts (with "all 15 pairs on 500 seeds" as its example) was written in the notebook at 12:28:55 (26b2ef9), while validation was running and, by the notebook, before its results were read. r2's design numbers were already known. The exact rule, including the 450 seeds, was committed at 12:44:18, after the validation results were read.
- The validation job wrote its last line at 12:39:33 (jobs/validation.exit on this box). The final files were committed at 12:42:00.

One more thing the review did not name. A second of the 15 pairs had also been run by the agent on 1,000 design seeds: c07 (cross.h3 x2, enc0.h1 x0), 127 fixed, 53 broken, +7.4. I checked every agent arm against the pool. Only r2 and c07 are built only from pool edits.

What changed.
- DEVIATIONS.md has a new entry that says all of this, including that the SPEC.md phrase "so the agent never saw it" and the "you do not need an LLM" control do not hold for this arm.
- freeze/REQUEST.json replaces the line that said the edit "shares one head" with the full statement.
- freeze/autogreedy.json has a new block, `relation_to_agent_candidates`, with the numbers and the statement that the report must not describe the arm as an independent no-LLM result. It is computed from the stored files by `python -m awe.freeze disclose` (new in awe/freeze.py). `awe/autogreedy.py` and the `rule` text in autogreedy.json are left as they ran, so their "What the author had seen" paragraph is still incomplete. The new block corrects it. `python -m awe.autogreedy report` would drop the block, and `python -m awe.freeze build` puts it back.
- freeze/edit.json names r2 in a one-line `autogreedy_arm` entry. It is a fixed string in awe/freeze.py.
- test_manifest.json, random_family.json, baselines.json and mean_action.json are byte-identical to the reviewed ones. The exported edit has the same SHA-256 (29e3749264d2). awe/freeze.py changed, so Gate 0 is replayed. No file that runs rollouts changed.

I did not rebuild the arm by another rule. Any rule I write now would also be written after seeing the results, and swapping the arm would change the package after the fact. The report will present this arm as a search whose author knew the result of the pair it returned.

## The non-blocking notes

- Constant offset: agreed, and now in REQUEST.json. The winner (joint 9, -0.01 rad) is at the smallest step of its grid and was not refined. On 1,000 design seeds -0.01 gives +3.3 and -0.02 gives +2.0. I did not refine it now.
- dose_k2 and dose_k3: agreed, and now in REQUEST.json. Both have cross.h1 at x0 (clipped). The curve should be read as a step in enc3.h4 there.
- Hypothesis: agreed. The atlas was on disk when the r4 line was written. REQUEST.json says so. The report will not lean on the line.
- Not checked by the review (checkpoint bytes, zone tables, smoke hashes): nothing new from me. Exports were compared in session 14.
