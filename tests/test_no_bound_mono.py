import math
import unittest
import torch
from src.no_bound_mono import chain_batches, objective


class NoBoundMonoTests(unittest.TestCase):
    def test_batches_no_cross_chain_edges_and_no_restored_percent(self):
        rows = [('a', 1), ('a%', -1), ('b', 2), ('%', -1)]
        chains = [dict(patterns=['a', 'a%']), dict(patterns=['b', '%'])]
        batches = chain_batches(rows, chains, 64, 0)
        self.assertEqual(len(batches), 1)
        b = batches[0]
        edges = {(b['queries'][i], b['queries'][j]) for i, j in zip(b['tight'], b['loose'])}
        self.assertEqual(edges, {('a', 'a%'), ('b', '%')})
        self.assertEqual(dict(zip(b['queries'], b['targets']))['%'], -1)
        self.assertEqual(batches, chain_batches(rows, chains, 64, 0))

    def test_inversion_gradients_and_tolerance_in_count_space(self):
        z = torch.tensor([math.log(101), math.log(11)], requires_grad=True)
        loss, sup, mono = objective(z, torch.tensor([-1., -1.]), [0], [1], 0., 1., .1)
        self.assertAlmostEqual(loss.item(), math.log(101/11/1.1), places=5)
        loss.backward()
        self.assertGreater(z.grad[0], 0)
        self.assertLess(z.grad[1], 0)
        self.assertEqual(sup.item(), 0)
        _, _, allowed = objective(torch.log(torch.tensor([11., 10.])), torch.tensor([-1., -1.]), [0], [1], 0., 1., .1)
        self.assertLess(allowed.item(), 1e-6)

    def test_pure_order_ignores_every_label_and_accepts_constant_solution(self):
        results = []
        for targets in [torch.tensor([1., 100.]), torch.tensor([999., -1.])]:
            z = torch.tensor([4., 2.], requires_grad=True)
            loss, _, _ = objective(z, targets, [0], [1], 0.)
            loss.backward(); results.append((loss.item(), z.grad.tolist()))
        self.assertEqual(results[0], results[1])
        loss, _, _ = objective(torch.tensor([8., 8.]), torch.tensor([-1., -1.]), [0], [1], 0.)
        self.assertEqual(loss.item(), 0.)

    def test_labeled_mask_and_zero_lambda_keep_unlabeled_rows(self):
        z = torch.tensor([2., 3., 4.], requires_grad=True)
        loss, sup, _ = objective(z, torch.tensor([1., -1., 99.]), [0, 1], [1, 2], 1., 0.)
        expected = ((2-math.log(2))**2 + (4-math.log(100))**2)/2
        self.assertAlmostEqual(loss.item(), expected, places=5)
        loss.backward();self.assertEqual(z.grad[1].item(), 0.)

    def test_reject_conflicting_labels_or_missing_queries(self):
        with self.assertRaises(ValueError):chain_batches([('a',1),('a',2)], [dict(patterns=['a'])], 64, 0)
        with self.assertRaises(ValueError):chain_batches([('a',1)], [dict(patterns=['b'])], 64, 0)

if __name__ == '__main__':unittest.main()
