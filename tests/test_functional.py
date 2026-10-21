import numpy as np
import needle as ndl
from needle import nn

def test_autograd_matmul():
    a = ndl.Tensor(np.arange(6, dtype=np.float32).reshape(2, 3), requires_grad=True)
    b = ndl.Tensor(np.ones((3, 4), dtype=np.float32), requires_grad=True)
    (a @ b).sum().backward()
    np.testing.assert_allclose(a.grad.numpy(), np.full((2, 3), 4.0))
    np.testing.assert_allclose(b.grad.numpy(), np.tile(np.array([[3], [5], [7]], dtype=np.float32), (1, 4)))

def test_nn_and_optimization():
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    x = ndl.Tensor(np.ones((5, 3), dtype=np.float32))
    loss = nn.SoftmaxLoss()(model(x), ndl.Tensor(np.array([0, 1, 0, 1, 0], dtype=np.float32)))
    loss.backward()
    opt = ndl.optim.SGD(model.parameters(), lr=0.01)
    norm = opt.clip_grad_norm(max_norm=0.1)
    assert np.isfinite(norm)
    assert sum(float((p.grad.numpy() ** 2).sum()) for p in model.parameters()) <= 0.011
    opt.step()
    opt.reset_grad()
    assert all(p.grad is None for p in model.parameters())
