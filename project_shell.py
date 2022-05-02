

import torch
import torchaudio
from torchvision.models import resnet34


import numpy as np


def load_samples(path):

    SAMPLE_RATE = 48000
    N_FFT = SAMPLE_RATE * 64 // 1000 + 4
    HOP_LENGTH = SAMPLE_RATE * 16 // 1000 + 4
    
    signal, _ = torchaudio.load(path)
    samples = get_samples(signal)

    specs = []

    for sample in samples:
        sample = prepare_sample(sample)

        spec = torch.stft(
            input=sample,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            normalized=True
        )

        real = spec[..., 0]
        img = spec[..., 1]
        specs.append(torch.cat([real, img], dim=0))

    return specs

def prepare_sample(waveform):

    waveform = waveform.numpy()
    current_len = waveform.shape[1]

    output = np.zeros((1, 165000), dtype='float32')
    output[0, -current_len:] = waveform[0, :165000]
    output = torch.from_numpy(output)

    return output

def get_samples(signal):

    ln = 18240 * 10
    samples = []
    waveform = signal.numpy()
    while len(waveform) > ln:
        samples.append(waveform[0:ln])
        waveform = waveform[ln+1:]
    
    return samples

def main():
    
    model = resnet34()
    model.conv1 = torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = torch.nn.Linear(in_features=512, out_features=1, bias=True)
    model = model.cpu()

    model.load_state_dict(torch.load('/content/drive/MyDrive/weigths_final.pth', map_location=torch.device('cpu')), strict=False)

    path = input('Введите путь до вашего файла: ')
    specs = load_samples(path)
    specs = [torch.unsqueeze(spec, 0) for spec in specs]

    noises = []
    
    for i in range(len(specs)):
        model.eval()
        with torch.no_grad():
            preds = model(specs[i])
            preds = np.around(preds[0])[0].item()
            if preds == 1:
                noises.append(i)
    
    if len(noises) == 0:
        print('Шумов нет')
    else:
        print('Таймкоды: ', end='')
        for time in noises:
            print(f'in {time // 60} misn {time % 60} secs', end=', ')

if __name__ == '__main__':
    main()

