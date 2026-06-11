# simulation/objectives/muscle_registry.py

from simulation.objectives import muscle_1018
from simulation.objectives import muscle_2354


def get_muscle_data_module(model_name):
    """
    model_nameに応じて筋パラメータファイルを返す。
    """

    name = model_name.lower()

    if "1018" in name or "10dof18" in name:
        return muscle_1018

    if "2354" in name or "23dof54" in name:
        return muscle_2354

    raise ValueError(
        f"Unknown muscle model: {model_name}. "
        "Expected model name containing '1018' or '2354'."
    )