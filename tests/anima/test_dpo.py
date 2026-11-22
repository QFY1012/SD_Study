import json
from pathlib import Path
import sys
import tempfile
import unittest
import torch

from anima_dpo.loss import dpo_loss, flow_noisy_latent, per_sample_mse, sample_flow_sigmas
from anima_dpo.train import load_pairs, temporary_param_override

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build_dpo_picker import build_picker_html, collect_groups_from_requests  # noqa: E402


class DpoTests(unittest.TestCase):
    def test_flow_sampling_and_noisy_target_identity(self):
        gen = torch.Generator().manual_seed(42)
        sigmas = sample_flow_sigmas(16, scheme="logit_normal", flow_shift=5.0, min_sigma=0.05, max_sigma=0.95, generator=gen)
        self.assertEqual(sigmas.shape, (16,))
        self.assertTrue((sigmas >= 0.05).all() and (sigmas <= 0.95).all())

        x0 = torch.randn(2, 4, 8, 8)
        noise = torch.randn_like(x0)
        s = torch.tensor([0.25, 0.75])
        xt, target = flow_noisy_latent(x0, noise, s)
        # Reconstruct x0 from x_t and exact velocity target: x_t - sigma * target == x0
        torch.testing.assert_close(xt - s.view(2, 1, 1, 1) * target, x0)

    def test_dpo_loss_at_reference_equals_log2_and_gradients_separate_win_lose(self):
        mse_ref_w = torch.tensor([0.2, 0.3])
        mse_ref_l = torch.tensor([0.25, 0.35])
        mse_pol_w = mse_ref_w.clone().requires_grad_(True)
        mse_pol_l = mse_ref_l.clone().requires_grad_(True)

        loss, stats = dpo_loss(mse_pol_w, mse_pol_l, mse_ref_w, mse_ref_l, beta=2000.0)
        self.assertAlmostEqual(stats["loss"], 0.693147, places=5)
        self.assertAlmostEqual(stats["reward_margin"], 0.0, places=6)

        loss.backward()
        # Reducing winner MSE reduces loss -> positive gradient wrt mse_pol_w; reducing loser MSE increases loss -> negative grad wrt mse_pol_l.
        self.assertTrue((mse_pol_w.grad > 0).all())
        self.assertTrue((mse_pol_l.grad < 0).all())

        better_loss, better_stats = dpo_loss(
            mse_ref_w - 0.002,
            mse_ref_l + 0.002,
            mse_ref_w,
            mse_ref_l,
            beta=2000.0,
        )
        self.assertLess(better_loss.item(), 0.2)
        self.assertEqual(better_stats["implicit_acc"], 1.0)
        self.assertGreater(better_stats["reward_margin"], 0.0)

    def test_temporary_param_override_preserves_parameter_identity_and_adam_state(self):
        p = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        opt = torch.optim.AdamW([p], lr=0.1)
        ref = [torch.tensor([5.0, 6.0])]

        with temporary_param_override([p], ref):
            torch.testing.assert_close(p.data, torch.tensor([5.0, 6.0]))
        torch.testing.assert_close(p.data, torch.tensor([1.0, 2.0]))

        (p.sum()).backward()
        opt.step()
        self.assertIn(p, opt.state)
        self.assertLess(p[0].item(), 1.0)

    def test_load_pairs_and_build_picker_html(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            w_img = td / "w.png"
            l_img = td / "l.png"
            Image.new("RGB", (32, 32), (0, 200, 0)).save(w_img)
            Image.new("RGB", (32, 32), (200, 0, 0)).save(l_img)

            req = td / "reward_request.json"
            req.write_text(
                json.dumps(
                    {
                        "samples": [
                            {"id": "0:0", "group_id": "g0", "prompt": "test prompt", "image": str(w_img)},
                            {"id": "0:1", "group_id": "g0", "prompt": "test prompt", "image": str(l_img)},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            groups = collect_groups_from_requests([req])
            self.assertEqual(len(groups), 1)
            self.assertEqual(len(groups[0]["candidates"]), 2)

            html_path = build_picker_html(groups, td / "picker.html")
            self.assertTrue(html_path.is_file())
            self.assertIn("test prompt", html_path.read_text(encoding="utf-8"))

            pairs_json = td / "pairs.json"
            pairs_json.write_text(
                json.dumps({"pairs": [{"prompt": "test prompt", "win_image": str(w_img), "lose_image": str(l_img)}]}),
                encoding="utf-8",
            )
            loaded = load_pairs(pairs_json)
            self.assertEqual(len(loaded), 1)


if __name__ == "__main__":
    unittest.main()
