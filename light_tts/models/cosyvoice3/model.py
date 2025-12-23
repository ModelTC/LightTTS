from light_tts.models.cosyvoice2.model import CosyVoice2TpPartModel


class CosyVoice3TpPartModel(CosyVoice2TpPartModel):
    def __init__(self, kvargs):
        super().__init__(kvargs)
        self.eos_token = self.speech_token_size + 1
        self.fill_token = self.speech_token_size + 3
        self.stop_token_ids = [self.speech_token_size + i for i in range(200)]
        return
