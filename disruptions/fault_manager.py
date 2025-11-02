class FaultManager:
    """
    Manages multiple faults in the environment.
    """
    def __init__(self, faults):
        self.faults = faults

    def step(self, current_step):
        """Called once each env step."""
        if not self.faults:
            return
        for fault in self.faults:
            fault.step(current_step)
