from pathlib import Path
import unittest

from anima_grpo.train import build_eval_jobs


class EvalCheckpointTests(unittest.TestCase):
    def test_build_eval_jobs_deterministic_six_column_groups(self):
        prompts = [f"eval_prompt_{i}" for i in range(120)]
        paths = [Path(f"eval_{i:03d}.png") for i in range(120)]
        jobs_a = build_eval_jobs(prompts, paths, 24, seed=10042)
        jobs_b = build_eval_jobs(prompts, paths, 24, seed=10042)
        self.assertEqual(jobs_a, jobs_b)
        self.assertEqual(len(jobs_a), 24)
        self.assertEqual([j["group_id"] for j in jobs_a[:7]], ["ev00"] * 6 + ["ev01"])
        self.assertEqual([j["id"] for j in jobs_a], sorted(j["id"] for j in jobs_a))


if __name__ == "__main__":
    unittest.main()
