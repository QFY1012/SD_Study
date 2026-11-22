from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "anima"))
from mimo_group_reward import (  # noqa: E402
    MiMoAnimaGroupGrader,
    combine_votes,
    compute_sample_reward,
    fill_invalid_with_group_mean,
    parse_pick,
    parse_single_audit,
    render_group_report,
    williams_orders,
)


class MiMoGroupRewardTests(unittest.TestCase):
    def test_williams_orders_position_and_adjacency_balance(self):
        orders = williams_orders(4, 4, seed=7)
        self.assertEqual(len(orders), 4)
        for col in range(4):
            self.assertEqual(sorted(row[col] for row in orders), [0, 1, 2, 3])
        adj = {(row[k], row[k + 1]) for row in orders for k in range(3)}
        self.assertEqual(len(adj), 12)  # all 4 * 3 ordered adjacencies covered once

    def test_combine_votes_denoising_rules(self):
        short = combine_votes([["G", "B", "-", "-"]], 4, min_ok=3)
        self.assertFalse(short["ok"])
        self.assertEqual(short["pick_norm"], [0.0, 0.0, 0.0, 0.0])

        ncd = combine_votes(
            [["N", "N", "N", "N"], ["N", "N", "N", "N"], ["G", "B", "-", "-"]],
            4,
            min_ok=3,
            ncd_zero_min=2,
        )
        self.assertTrue(ncd["ok"])
        self.assertTrue(ncd["ncd_zeroed"])
        self.assertEqual(ncd["pick_norm"], [0.0, 0.0, 0.0, 0.0])

        votes = [
            ["G", "B", "G", "-"],
            ["G", "B", "-", "-"],
            ["G", "B", "-", "-"],
            ["-", "-", "-", "-"],
        ]
        res = combine_votes(votes, 4, min_ok=3, net_deadzone=1)
        self.assertEqual(res["noise_zeroed"], [2])
        self.assertAlmostEqual(res["pick_norm"][0], 0.75)
        self.assertAlmostEqual(res["pick_norm"][1], -0.75)
        self.assertEqual(res["pick_norm"][2], 0.0)

        six = combine_votes(
            [
                ["N"] * 6,
                ["N"] * 6,
                ["G", "B", "-", "-", "G", "-"],
                ["G", "B", "-", "-", "G", "-"],
                ["G", "B", "-", "-", "G", "-"],
                ["G", "-", "-", "-", "G", "-"],
            ],
            6,
            min_ok=4,
        )
        self.assertFalse(six["ncd_zeroed"])
        self.assertGreater(six["pick_norm"][0], six["pick_norm"][2])
        self.assertGreater(six["pick_norm"][2], six["pick_norm"][1])

    def test_compute_sample_reward_five_binary_vetoes_and_pass_band(self):
        self.assertEqual(compute_sample_reward(0.8, {"prompt_ok": True, "severed_or_extra_limbs": True}), (0.0, "veto:severed_or_extra_limbs"))
        self.assertEqual(compute_sample_reward(1.0, {"prompt_ok": True, "digit_error": True}), (0.0, "veto:digit_error"))
        self.assertEqual(compute_sample_reward(1.0, {"prompt_ok": True, "paw_pad_error": True}), (0.0, "veto:paw_pad_error"))
        self.assertEqual(compute_sample_reward(1.0, {"prompt_ok": True, "identity_or_clothes_error": True}), (0.0, "veto:identity_or_clothes_error"))
        self.assertEqual(compute_sample_reward(0.9, {"prompt_ok": False}), (0.0, "veto:prompt_mismatch"))

        clean = {
            "prompt_ok": True,
            "severed_or_extra_limbs": False,
            "digit_error": False,
            "paw_pad_error": False,
            "identity_or_clothes_error": False,
        }
        r_worst, note_worst = compute_sample_reward(-1.0, clean)
        r_mid, _ = compute_sample_reward(0.0, clean)
        r_best, _ = compute_sample_reward(1.0, clean)
        self.assertIsNone(note_worst)
        self.assertAlmostEqual(r_worst, 0.60, places=6)
        self.assertAlmostEqual(r_mid, 0.80, places=6)
        self.assertAlmostEqual(r_best, 1.00, places=6)

    def test_grade_groups_runs_pick_only_on_two_or_more_survivors(self):
        grader = MiMoAnimaGroupGrader(api_key="dummy")
        ok_audit = {
            "prompt_ok": True,
            "severed_or_extra_limbs": False,
            "digit_error": False,
            "paw_pad_error": False,
            "identity_or_clothes_error": False,
            "reason": "clean",
        }
        bad_audit = {**ok_audit, "digit_error": True, "reason": "4 fingers"}
        audit_map = {"g1_0": ok_audit, "g1_1": bad_audit, "g1_2": ok_audit, "g1_3": bad_audit, "g2_0": ok_audit, "g2_1": bad_audit}
        pick_calls = []

        grader.audit_single = lambda prompt, path, deadline=None: audit_map[path]

        def fake_pick(prompt, paths, order, deadline=None):
            pick_calls.append((tuple(paths), tuple(order)))
            # Always vote survivor index 0 ("g1_0") good and survivor index 1 ("g1_2") bad
            pos0_orig = order[0]
            return {"good": [0], "bad": [1], "no_clear_difference": False, "reason": "ok", "order": order}

        grader.pick_round = fake_pick
        samples = [
            {"id": "g1_0", "group_id": "g1", "prompt": "p1", "image": "g1_0", "seed": 1},
            {"id": "g1_1", "group_id": "g1", "prompt": "p1", "image": "g1_1", "seed": 1},
            {"id": "g1_2", "group_id": "g1", "prompt": "p1", "image": "g1_2", "seed": 1},
            {"id": "g1_3", "group_id": "g1", "prompt": "p1", "image": "g1_3", "seed": 1},
            {"id": "g2_0", "group_id": "g2", "prompt": "p2", "image": "g2_0", "seed": 2},
            {"id": "g2_1", "group_id": "g2", "prompt": "p2", "image": "g2_1", "seed": 2},
        ]
        out = grader.grade_groups(samples, max_workers=2)
        # Only g1 had >=2 survivors ("g1_0", "g1_2"); g2 (1 survivor) skipped pick entirely
        self.assertEqual(len(pick_calls), 2)
        self.assertTrue(all(paths == ("g1_0", "g1_2") for paths, _ in pick_calls))
        score_map = {s["id"]: s["reward"] for s in out["scores"]}
        self.assertEqual(score_map["g1_0"], 1.0)
        self.assertEqual(score_map["g1_2"], 0.6)
        self.assertEqual(score_map["g1_1"], 0.0)
        self.assertEqual(score_map["g1_3"], 0.0)
        self.assertEqual(score_map["g2_0"], 0.8)  # sole survivor = neutral 0.8
        self.assertEqual(score_map["g2_1"], 0.0)

    def test_fill_invalid_with_group_mean_and_json_parsers(self):
        filled = fill_invalid_with_group_mean([0.8, None, 0.4, 0.6])
        self.assertAlmostEqual(filled[1], 0.6)
        self.assertEqual(fill_invalid_with_group_mean([None, None]), [0.5, 0.5])

        self.assertIsNone(parse_single_audit('{"prompt_ok": "yes"}'))
        audit = parse_single_audit('```json\n{"prompt_ok": true, "paw_pad_error": true}\n```')
        self.assertTrue(audit["prompt_ok"])
        self.assertTrue(audit["paw_pad_error"])

        pick = parse_pick('{"good": [0, 99, true], "bad": [2], "no_clear_difference": false}', n=4)
        self.assertEqual(pick["good"], [0])
        self.assertEqual(pick["bad"], [2])

    def test_render_group_report_creates_six_column_sheet(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            samples = []
            audits = {}
            rewards = {}
            for i in range(6):
                img_p = td / f"s_{i}.png"
                Image.new("RGB", (64, 64), (40 * i, 120, 200)).save(img_p)
                sid = f"0:{i}"
                samples.append({"id": sid, "group_id": "p0", "prompt": "demo", "image": str(img_p), "seed": i})
                audits[sid] = {
                    "prompt_ok": True,
                    "severed_or_extra_limbs": False,
                    "digit_error": i >= 3,
                    "paw_pad_error": False,
                    "identity_or_clothes_error": False,
                    "reason": f"sample {i}",
                }
                rewards[sid] = 0.8 if i < 3 else 0.0
            traces = {"p0": {"agg": {"pick_sum": [1, 0, -1, 0, 0, 0]}, "audits": audits, "rewards": rewards}}
            out = render_group_report(samples, traces, td / "mimo_report.png")
            self.assertIsNotNone(out)
            self.assertTrue(out.is_file())
            with Image.open(out) as rendered:
                self.assertEqual(rendered.size, (6 * 270, 74 + 360 + 48))


if __name__ == "__main__":
    unittest.main()
