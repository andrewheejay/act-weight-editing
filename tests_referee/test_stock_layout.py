"""The referee reads every layout of LeRobot's eval_info.json, including the one 0.7.0 really writes."""

from __future__ import annotations

from referee.common import stock_per_episode

# what lerobot_eval.eval_policy_all saves in LeRobot 0.7.0, cut to three episodes
REAL_070 = {
    "per_task": [{
        "task_group": "aloha",
        "task_id": 0,
        "metrics": {"sum_rewards": [3.0, 410.0, 12.0], "max_rewards": [1.0, 4.0, 2.0], "successes": [False, True, False], "video_paths": [], "predicted_video_paths": []},
        "n_episodes": 3,
        "n_success": 1,
        "pc_success": 33.3,
    }],
    "per_group": {"aloha": {"n_episodes": 3}},
    "overall": {"n_episodes": 3},
}


def test_the_real_layout_is_read_from_the_start_seed():
    rows = stock_per_episode(REAL_070, start_seed=1000)
    assert [(r["seed"], r["success"], r["max_reward"]) for r in rows] == [(1000, False, 1.0), (1001, True, 4.0), (1002, False, 2.0)]


def test_the_real_layout_without_a_start_seed_fails_closed():
    assert stock_per_episode(REAL_070) == []


def test_several_tasks_without_seeds_fail_closed():
    two = {"per_task": [REAL_070["per_task"][0], REAL_070["per_task"][0]]}
    assert stock_per_episode(two, start_seed=1000) == []


def test_files_that_carry_seeds_are_read_by_their_seeds():
    old = {"per_episode": [{"seed": 1007, "success": True}], "aggregated": {}}
    assert stock_per_episode(old, start_seed=1000) == [{"seed": 1007, "success": True}]
    nested = {"per_task": [{"metrics": {"successes": [True], "per_episode": [{"seed": 1009, "success": True}]}}]}
    assert stock_per_episode(nested, start_seed=1000) == [{"seed": 1009, "success": True}]
