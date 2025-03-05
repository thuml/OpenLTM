from models import timer, timer_xl, moirai, moment, gpt4ts, ttm


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "timer": timer,
            "timer_xl": timer_xl,
            "moirai": moirai,
            "moment": moment,
            "gpt4ts": gpt4ts,
            "ttm": ttm,
        }
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
