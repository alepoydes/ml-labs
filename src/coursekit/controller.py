"""Optional base for robot controllers with explicit sensor diagnostics."""


class Controller:
    def __init__(self):
        self._debug_observation = {}

    def act(self, observation, reset):
        """Call before processing inputs; keep detached copies for debug()."""
        self._debug_observation = {
            name: value.detach().clone() for name, value in observation.items()
        }

    def debug(self, index=0):
        values = {}
        for name, batch in self._debug_observation.items():
            if name == "rgb":
                for channel, colour in enumerate(("red", "green", "blue")):
                    values[f"sensor.{colour}"] = batch[index, :, channel]
            else:
                values[f"sensor.{name}"] = batch[index]
        return values
