import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "anima"))
from ppt_verifier_reward import (  # noqa: E402
    LETTER_VAL,
    bradley_terry_prob,
    extract_letter_scores,
    probabilistic_pivot_tournament,
)


class PptVerifierTests(unittest.TestCase):
    def test_extract_letter_scores_expectation(self):
        # Simulate token sequence around <score_A>P</score_A>\n<score_B>K</score_B>
        choice = {
            "message": {"content": "<score_A>P</score_A>\n<score_B>K</score_B>"},
            "logprobs": {
                "content": [
                    {"token": "<score_A>", "logprob": 0.0, "top_logprobs": []},
                    {
                        "token": "P",
                        "logprob": math.log(0.5),
                        "top_logprobs": [
                            {"token": "P", "logprob": math.log(0.5)},
                            {"token": "R", "logprob": math.log(0.5)},
                            {"token": "Z", "logprob": -0.1},  # out of A..T, ignored
                        ],
                    },
                    {"token": "</score_A>\n<score_B>", "logprob": 0.0, "top_logprobs": []},
                    {
                        "token": "K",
                        "logprob": 0.0,
                        "top_logprobs": [{"token": "K", "logprob": 0.0}],
                    },
                    {"token": "</score_B>", "logprob": 0.0, "top_logprobs": []},
                ]
            },
        }
        res = extract_letter_scores(choice)
        expected_a = 0.5 * LETTER_VAL["P"] + 0.5 * LETTER_VAL["R"]
        self.assertAlmostEqual(res["score_A"]["exp"], expected_a, places=6)
        self.assertAlmostEqual(res["score_B"]["exp"], LETTER_VAL["K"], places=6)

    def test_ppt_budget_and_ranking_order(self):
        # N=8, k=2 => N + k(N-k) + C(k,2) minus overlap <= 8 + 12 + 1 = 21 pairs (vs 28 round-robin)
        true_strength = [0.1, 0.9, 0.4, 0.85, 0.2, 0.6, 0.3, 0.7]
        calls = []

        def scorer(i, j):
            calls.append((i, j))
            return true_strength[i], true_strength[j]

        out = probabilistic_pivot_tournament(
            num_candidates=8,
            pivots=2,
            pair_scorer=scorer,
            seed=123,
            max_workers=1,
        )
        self.assertEqual(len(out["ring_edges"]), 8)
        self.assertLessEqual(len(calls), 8 + 2 * (8 - 2) + 1)
        # Every candidate appeared at least twice (once A, once B in ring)
        a_slots = [i for i, _ in [(e[0], e[1]) for e in out["ring_edges"]]]
        b_slots = [j for _, j in [(e[0], e[1]) for e in out["ring_edges"]]]
        self.assertEqual(sorted(a_slots), list(range(8)))
        self.assertEqual(sorted(b_slots), list(range(8)))
        best_idx = max(range(8), key=lambda idx: out["scores"][idx])
        self.assertEqual(best_idx, 1)  # 0.9 is strongest

    def test_ppt_ties_yield_equal_scores(self):
        out = probabilistic_pivot_tournament(
            num_candidates=4,
            pivots=2,
            pair_scorer=lambda i, j: (0.5, 0.5),
            seed=42,
            max_workers=2,
        )
        for s in out["scores"]:
            self.assertAlmostEqual(s, 0.5, places=6)

    def test_bradley_terry_symmetry(self):
        self.assertAlmostEqual(bradley_terry_prob(0.8, 0.3) + bradley_terry_prob(0.3, 0.8), 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
