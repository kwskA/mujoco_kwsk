# optim/optimization_logger.py

import os
import csv
import json
import matplotlib.pyplot as plt


class OptimizationLogger:
    """
    最適化中の世代ごとの評価値を記録・保存するクラス。
    """

    def __init__(self, result_dir):
        """
        ログ保存先ディレクトリを設定する。
        """

        self.result_dir = result_dir
        self.log_dir = os.path.join(result_dir, "optimization_logs")
        os.makedirs(self.log_dir, exist_ok=True)

        self.records = []

    def record(
        self,
        generation,
        min_cost,
        mean_cost,
        best_cost,
        details=None,
    ):
        """
        1世代分の最適化結果を記録する。
        """

        if details is None:
            details = {}

        record = {
            "generation": int(generation),
            "min_cost": float(min_cost),
            "mean_cost": float(mean_cost),
            "best_cost": float(best_cost),
            "details": dict(details),
        }

        self.records.append(record)

    def save_csv(self, filename="optimization_history.csv"):
        """
        世代ごとの評価値をCSVとして保存する。
        """

        path = os.path.join(self.log_dir, filename)

        detail_keys = self._collect_detail_keys()

        header = [
            "generation",
            "min_cost",
            "mean_cost",
            "best_cost",
        ] + detail_keys

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for record in self.records:
                row = [
                    record["generation"],
                    record["min_cost"],
                    record["mean_cost"],
                    record["best_cost"],
                ]

                details = record["details"]

                for key in detail_keys:
                    row.append(details.get(key, ""))

                writer.writerow(row)

        print(f"[OptimizationLogger] wrote {path}")

    def save_json(self, filename="optimization_history.json"):
        """
        世代ごとの評価値をJSONとして保存する。
        """

        path = os.path.join(self.log_dir, filename)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                self.records,
                f,
                indent=4,
                ensure_ascii=False,
            )

        print(f"[OptimizationLogger] wrote {path}")

    def plot_cost_history(self, filename="cost_history.png"):
        """
        min, mean, best cost の推移をグラフとして保存する。
        """

        if len(self.records) == 0:
            print("[OptimizationLogger] no records, skipped plot.")
            return

        path = os.path.join(self.log_dir, filename)

        generations = [
            record["generation"]
            for record in self.records
        ]

        min_costs = [
            record["min_cost"]
            for record in self.records
        ]

        mean_costs = [
            record["mean_cost"]
            for record in self.records
        ]

        best_costs = [
            record["best_cost"]
            for record in self.records
        ]

        plt.figure(figsize=(10, 5))

        plt.plot(generations, min_costs, label="generation_min")
        plt.plot(generations, mean_costs, label="generation_mean")
        plt.plot(generations, best_costs, label="global_best")

        plt.xlabel("Generation")
        plt.ylabel("Cost")
        plt.title("Optimization cost history")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(path)
        plt.close()

        print(f"[OptimizationLogger] wrote {path}")

    def save_all(self):
        """
        CSV, JSON, グラフをまとめて保存する。
        """

        self.save_csv()
        self.save_json()
        self.plot_cost_history()

    def _collect_detail_keys(self):
        """
        detailsに含まれる評価項目名を集める。
        """

        keys = []

        for record in self.records:
            for key in record["details"].keys():
                if key not in keys:
                    keys.append(key)

        return keys