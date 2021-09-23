import numpy as np
from scipy import signal
import glob
import os
from scipy.signal import chirp
import soundfile as sf
from collections import defaultdict


def rotate_ambisonics(signal, angle=0):
    rotation_mat = np.eye(9)
    rotation_mat[1, 1] = np.cos(angle * np.pi / float(180))
    rotation_mat[3, 1] = np.sin(angle * np.pi / float(180))
    rotation_mat[1, 3] = -1 * np.sin(angle * np.pi / float(180))
    rotation_mat[3, 3] = np.cos(angle * np.pi / float(180))
    rotation_mat[4, 4] = np.cos(2 * angle * np.pi / float(180))
    rotation_mat[4, 8] = -1 * np.sin(2 * angle * np.pi / float(180))
    rotation_mat[5:8, 5:8] = rotation_mat[1:4, 1:4]
    rotation_mat[8, 4] = np.sin(2 * angle * np.pi / float(180))
    rotation_mat[8, 8] = np.cos(2 * angle * np.pi / float(180))
    out_signal = np.matmul(signal, rotation_mat)
    return out_signal


def convolve_rir(rir, audio_clip, mode='same'):
    signal_rev = signal.fftconvolve(audio_clip[:, np.newaxis], rir, mode=mode)
    return signal_rev


def freq_sweep_signal(duration=0.5, fs=16000, index=0, **kwargs):
    # IR fs = 16000
    t = np.arange(0, int(fs * duration)) / float(fs)
    w = chirp(t, f0=20, f1=20000, t1=duration, method='linear')
    return w


class SemanticClips:
    def __init__(
            self,
            audio_dir='/glusterfs/spurushw/AUDIO/audio_clips/semantic_clips_small/'
    ):
        """TODO: Docstring for __init__.

        :clipname: TODO
        :duration: TODO
        :audio_dir: TODO
        :returns: TODO

        """
        self.audio_clips = {}
        self.audio_fs = {}
        self.rooms = [
            os.path.basename(r)
            for r in glob.glob(os.path.join(audio_dir, '*'))
        ]
        self.room_clip_paths = {}
        for r in self.rooms:
            self.room_clip_paths[r] = glob.glob(
                os.path.join(audio_dir, r, '*.wav'))

        self.room_clips = defaultdict(list)
        for r in self.rooms:
            if len(self.room_clip_paths[r]) == 0:
                self.room_clips[r].append(np.zeros(16000 * 10))

            for fname in self.room_clip_paths[r]:
                data, fs = sf.read(fname)
                if data.ndim == 2:
                    data = data.mean(1)
                data /= np.amax(np.abs(data))
                nonzero_ind = np.where(data > 0.05)[0][0]
                data = data[nonzero_ind:]
                ind = np.arange(0, len(data), int(float(fs) / 16000))
                data = data[ind]
                self.room_clips[r].append(data)

        self.room_clips['void'].append(np.zeros(16000 * 10))

    def __call__(self,
                 duration=3.0,
                 test=False,
                 normalize=False,
                 index=0,
                 meta=None,
                 **kwargs):
        """TODO: Docstring for __call__.

        :returns: TODO

        """
        room = meta['room'].replace('/', '_')
        if room not in self.room_clips.keys():
            print('Could not find room {}'.format(room))
            room = 'void'
        allclips = self.room_clips[room]
        if not test:
            ind = np.random.randint(len(allclips))
        else:
            ind = index % len(allclips)
        data = allclips[ind]
        data = data[:int(16000 * duration)]
        if data.shape[0] < int(duration * 16000):
            data = np.pad(data, (0, int(duration * 16000) - data.shape[0]))

        if not normalize:
            return data
        data = data / (np.amax(np.abs(data)) + 1e-10)
        return data


audio_loaders = {
    'freq_sweep_signal':
    freq_sweep_signal,
    'semantic_train_clips':
    SemanticClips(
        audio_dir='/glusterfs/spurushw/AUDIO/audio_clips/semantic_clips_train/'
    ),
    'semantic_val_clips':
    SemanticClips(
        audio_dir='/glusterfs/spurushw/AUDIO/audio_clips/semantic_clips_val/'),
    'semantic_test_clips':
    SemanticClips(
        audio_dir='/glusterfs/spurushw/AUDIO/audio_clips/semantic_clips_test/')
}
