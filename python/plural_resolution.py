from experiments.latest.system import LatestSystem


class PluralResolutionDemo:
    def __init__(self, iteration_num=1, demo_only=True):
        self.latest_system = LatestSystem(iteration_num, use_test_params=demo_only)

    def exe(self):
        self._run_latest()

    def _run_latest(self):
        self.latest_system.run()
