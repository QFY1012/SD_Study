import json
from pathlib import Path
import sys
import tempfile
import unittest
import torch
from anima_grpo.core import initial_noise, transition, advantages, policy_loss
from anima_grpo.train import score


class CoreTests(unittest.TestCase):
    def test_initial_noise_sharing(self):
        a = initial_noise((1, 16, 1, 4, 4), seed=42, iteration=3, sample=0, same_noise=True)
        b = initial_noise((1, 16, 1, 4, 4), seed=42, iteration=3, sample=1, same_noise=True)
        torch.testing.assert_close(a, b)
        c = initial_noise((1, 16, 1, 4, 4), seed=42, iteration=3, sample=0, same_noise=False)
        d = initial_noise((1, 16, 1, 4, 4), seed=42, iteration=3, sample=1, same_noise=False)
        self.assertFalse(torch.equal(c, d))

    def test_transition_density_and_replay(self):
        x = torch.randn(2, 16, 1, 4, 4)
        v = torch.randn_like(x)
        y, lp = transition(x, v, 0.8, 0.6, 0.3)
        _, replay = transition(x, v, 0.8, 0.6, 0.3, next_x=y)
        torch.testing.assert_close(lp, replay)
        x0 = x - 0.8 * v
        drift = -(x - x0 * 0.2) / 0.8**2
        mean = x - 0.2 * v + 0.5 * 0.3**2 * drift * 0.2
        expected = torch.distributions.Normal(mean, 0.3 * (0.2**0.5)).log_prob(y).flatten(1).mean(1)
        torch.testing.assert_close(lp, expected)

    def test_equal_rewards_and_invalid_group(self):
        self.assertTrue((advantages(torch.ones(4)) == 0).all())
        for r in (torch.ones(1), torch.tensor([1., float('nan')])):
            with self.assertRaises(ValueError): advantages(r)

    def test_clipping_both_advantage_signs(self):
        ratio = torch.tensor([1.5, 0.5])
        loss, _ = policy_loss(ratio.log(), torch.zeros(2), torch.tensor([1., -1.]), clip=0.1)
        # Positive reward caps at 1.1; negative reward caps down at 0.9.
        torch.testing.assert_close(loss, torch.tensor((-1.1 + 0.9) / 2))

    def test_only_adapter_updates(self):
        torch.manual_seed(123)
        base = torch.nn.Linear(4, 4);base.requires_grad_(False)
        down, up = torch.nn.Linear(4, 2, bias=False), torch.nn.Linear(2, 4, bias=False)
        torch.nn.init.zeros_(up.weight)
        x = torch.randn(1, 4)
        def velocity(): return base(x) + up(down(x))
        with torch.no_grad():y, old = transition(x, velocity(), .8, .6, .3)
        _, new = transition(x, velocity(), .8, .6, .3, next_x=y)
        loss, ratio = policy_loss(new, old, torch.tensor([1.]))
        torch.testing.assert_close(ratio, torch.ones_like(ratio))
        loss.backward()
        self.assertIsNone(base.weight.grad)
        self.assertGreater(up.weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(up.weight.grad).all())

    def test_reward_protocol_orders_ids_and_rejects_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp); provider = d/'reward.py'
            provider.write_text('import json,sys;json.dump({"scores":[{"id":"b","reward":2},{"id":"a","reward":1}]},open(sys.argv[2],"w"))')
            jobs = [{'id':'a'}, {'id':'b'}]
            self.assertEqual(score([sys.executable,str(provider)],jobs,d),[1,2])
            provider.write_text('import json,sys;json.dump({"scores":[{"id":"b","reward":2},{"id":"a","reward":float("nan")}]},open(sys.argv[2],"w"))')
            with self.assertRaises(ValueError):score([sys.executable,str(provider)],jobs,d)

    def test_invalid_noise_schedule(self):
        for s, n, eta in ((0.,0.,.3),(.5,.5,.3),(.8,.6,0)):
            with self.assertRaises(ValueError):transition(torch.ones(1,4),torch.ones(1,4),s,n,eta)


if __name__ == '__main__':unittest.main()
