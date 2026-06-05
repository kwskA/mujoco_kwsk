# optim/optimizer_factory.py

from optim.cmaes_optimizer import CMAESOptimizer


def create_optimizer(
    optimizer_type,
    **kwargs,
):
    """
    optimizer_typeに応じて最適化器を作成する。
    """

    if optimizer_type == "cmaes":
        return CMAESOptimizer(**kwargs)

    raise ValueError(
        f"Unknown optimizer_type: {optimizer_type}"
    )