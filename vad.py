import torch
import warnings

warnings.filterwarnings("ignore")


class VAD:

    def __init__(self):
        self.model = init_jit_model("silero_vad.jit")

    def is_speech(self, audio, threshold: float = 0.5, sampling_rate: int = 16000):
        if not torch.is_tensor(audio):
            try:
                audio = torch.Tensor(audio)
            except:
                raise TypeError("Audio cannot be casted to tensor. Cast it manually")
        speech_prob = self.model(audio, sampling_rate).item()
        return speech_prob >= threshold

    def reset_states(self):
        self.model.reset_states()


def init_jit_model(model_path: str, device=torch.device('cpu')):
    torch.set_grad_enabled(False)
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model
