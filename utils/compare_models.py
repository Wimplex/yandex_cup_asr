import warnings
warnings.filterwarnings('ignore')

import os

import numpy as np

import torch

from modules.extractors import DeiT_Extractor, ResNet18_Extractor, AudioClassifier


def count_parameters(model):
    """ Counts number of learnable parameters in the model """
    return sum(p.numel() for p in model.parameters())


def measure_inference_timings(model, dummy_tensor, reps=300, device='cuda:0'):
    """ Measures mean inference time and FPS """
    model.to(device)
    dummy_tensor = dummy_tensor.to(device)

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    timings = np.zeros([reps, 1])

    # Warming up
    for _ in range(10): model(dummy_tensor)

    # Measuring
    with torch.no_grad():
        for rep in range(reps):
            starter.record()
            model(dummy_tensor)
            ender.record()

            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
        
    mean_inference_time = timings.mean()
    fps = 1000 / mean_inference_time
    return mean_inference_time, fps


def compare(model_archs: list, num_classes, embedding_size):
    input_shape = [3, 224, 224]
    print("Compare on tensor shape of", input_shape)
    net_kwargs = {'input_shape': input_shape, 'num_classes': num_classes, 'embed_size': embedding_size}
    models = {
        'ResNet18': ResNet18_Extractor(**net_kwargs),
        'DeiT-Ti': DeiT_Extractor(**net_kwargs),
    }
    
    models_to_compare = {m_name: model for m_name, model in models.items() if m_name in model_archs}
    for model_name, model in models_to_compare.items():
        print(f"[{model_name}]")
        model = AudioClassifier(model)

        # Check model size in Mb
        model_path = f'models/{model_name}.pth'
        torch.save(model.state_dict(), model_path)
        model_size = os.path.getsize(model_path) / 1000000
        os.remove(model_path)
        print(" --- Model size (Mb):", model_size)

        # Count parameters
        print(" --- Parameters count (millions):", count_parameters(model) / 1000000)

        # Measure timings
        dummy_tensor = torch.randn([1] + input_shape)
        inf_time, fps = measure_inference_timings(model, dummy_tensor, device='cuda:0')
        print(" --- Inference time (ms):", inf_time)
        print(" --- FPS:", fps)
