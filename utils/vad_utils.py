from collections import deque
import torch
import torch.nn.functional as F


def validate(model, inputs: torch.Tensor):
    with torch.no_grad(): outs = model(inputs)
    return outs


def get_speech_ts(wav: torch.Tensor,
                  model,
                  trig_sum: float = 0.25,
                  neg_trig_sum: float = 0.07,
                  num_steps: int = 8,
                  batch_size: int = 200,
                  num_samples_per_window: int = 4000,
                  min_speech_samples: int = 10000, #samples
                  min_silence_samples: int = 500,
                  run_function=validate,
                  smoothed_prob_func='mean',
                  device='cpu'):

    assert smoothed_prob_func in ['mean', 'max'],  'smoothed_prob_func not in ["max", "mean"]'
    num_samples = num_samples_per_window
    assert num_samples % num_steps == 0
    step = int(num_samples / num_steps)  # stride / hop
    outs = []
    to_concat = []
    for i in range(0, len(wav), step):
        chunk = wav[i: i+num_samples]
        if len(chunk) < num_samples:
            chunk = F.pad(chunk, (0, num_samples - len(chunk)))
        to_concat.append(chunk.unsqueeze(0))
        if len(to_concat) >= batch_size:
            chunks = torch.Tensor(torch.cat(to_concat, dim=0)).to(device)
            out = run_function(model, chunks)
            outs.append(out)
            to_concat = []

    if to_concat:
        chunks = torch.Tensor(torch.cat(to_concat, dim=0)).to(device)
        out = run_function(model, chunks)
        outs.append(out)

    outs = torch.cat(outs, dim=0)
    buffer = deque(maxlen=num_steps)  # maxlen reached => first element dropped
    triggered = False
    speeches = []
    current_speech = {}
    speech_probs = outs[:, 1]  # this is very misleading
    temp_end = 0

    for i, predict in enumerate(speech_probs):  # add name
        buffer.append(predict)
        if smoothed_prob_func == 'mean': smoothed_prob = (sum(buffer) / len(buffer))
        elif smoothed_prob_func == 'max': smoothed_prob = max(buffer)

        if (smoothed_prob >= trig_sum) and temp_end: temp_end=0
        if (smoothed_prob >= trig_sum) and not triggered:
            triggered = True
            current_speech['start'] = step * max(0, i-num_steps)
            continue
        if (smoothed_prob < neg_trig_sum) and triggered:
            if not temp_end: temp_end = step * i
            if step * i - temp_end < min_silence_samples: continue
            else:
                current_speech['end'] = temp_end
                if (current_speech['end'] - current_speech['start']) > min_speech_samples: speeches.append(current_speech)
                temp_end = 0
                current_speech = {}
                triggered = False
                continue
    if current_speech:
        current_speech['end'] = len(wav)
        speeches.append(current_speech)
    return speeches