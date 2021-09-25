import warnings
warnings.filterwarnings('ignore')

import os

import torch

from networks import DeiT_Extractor, ResNet18_Extractor, AudioClassifier
from base_utils import count_parameters, measure_inference_timings
from config import Config


def main():
    input_shape = [6, 224, 224]
    net_kwargs = {'input_shape': input_shape, 'num_classes': Config.NUM_CLASSES, 'embed_size': Config.EMB_SIZE}
    models = {
        'ResNet18': ResNet18_Extractor(**net_kwargs),
        'DeiT-Ti': DeiT_Extractor(**net_kwargs),
    }
    
    for model_name, model in models.items():
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


if __name__ == '__main__':
    main()