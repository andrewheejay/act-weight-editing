"""A fake harness for testing the referee. It implements referee/CONTRACT.md with made-up outcomes.

Outcomes are a fixed function of the task, the arm and the seed, so runs repeat exactly, the same
way the real simulator does. No robot, no model. Used only by tests_referee.
"""
