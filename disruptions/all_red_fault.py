import traci


class AllRedFault:
    def __init__(self, tls_id, duration_steps: int, start_step: int):
        self.tls_id = tls_id
        self.duration_steps = duration_steps
        self.start_step = start_step

        self._timer = 0

    def step(self, current_step: int):
        if current_step == self.start_step:
            self._all_red_program()
            self._timer = self.duration_steps
        elif self._timer > 0:
            self._timer -= 1
            self._all_red_program()

    def _all_red_program(self):
        self.pattern_len = len(traci.trafficlight.getRedYellowGreenState(self.tls_id))
        traci.trafficlight.setRedYellowGreenState(self.tls_id, "r" * self.pattern_len)
