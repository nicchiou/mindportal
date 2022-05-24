from ray.tune import CLIReporter
from ray.tune.trial import Trial


class TrialTerminationReporter(CLIReporter):
    def __init__(self):
        super(TrialTerminationReporter, self).__init__()
        self.num_terminated = 0

    def should_report(self, trials, done=False):
        """Reports only on trial termination events."""
        old_num_terminated = self.num_terminated
        self.num_terminated = len(
            [t for t in trials if t.status == Trial.TERMINATED])
        return self.num_terminated > old_num_terminated
