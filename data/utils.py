import numba
import os
import math
import pickle
from typing import List, Optional, Tuple
import numpy as np
import torch
import cv2
import random
import moviepy.editor as mpy
import moviepy.audio as mpy_audio
from moviepy.audio.AudioClip import CompositeAudioClip

from PIL import Image
import soundfile as sf
import habitat.utils.visualizations.fog_of_war as fow
import habitat.utils.visualizations.maps as maps
import sklearn
import sklearn.metrics

import kornia.geometry.transform as korntransforms


def to_cuda(x):
    if isinstance(x, list):
        out = [to_cuda(xi) for xi in x]
    return x.cuda(non_blocking=True)


def prepare_batch(data, target, meta):
    audio, rgb, relpath = data[:3]
    meta['predictable_target'] = to_cuda(meta['predictable_target'])
    meta['semantic_target'] = to_cuda(meta['semantic_target'])
    rgb = rgb.cuda(non_blocking=True)
    audio = audio.cuda(non_blocking=True)
    relpath = relpath.cuda(non_blocking=True)

    if target.ndim == 4:
        target = target.view(-1, target.shape[2], target.shape[3])
        meta['predictable_target'] = meta['predictable_target'].view(
            -1, meta['predictable_target'].shape[2],
            meta['predictable_target'].shape[3])
        meta['semantic_target'] = meta['semantic_target'].view(
            -1, meta['semantic_target'].shape[2],
            meta['semantic_target'].shape[3])
    target = target.cuda(non_blocking=True)
    return audio, rgb, relpath, target, meta


def images_to_video_with_audio(images: List[np.ndarray],
                               output_dir: str,
                               video_name: str,
                               audios: List[str],
                               sr: int,
                               audio_duration: int = 2.5,
                               fps: float = 1.0,
                               quality: Optional[float] = 5,
                               **kwargs):
    r"""Calls imageio to run FFMPEG on a list of images. For more info on
    parameters, see https://imageio.readthedocs.io/en/stable/format_ffmpeg.html
    Args:
        images: The list of images. Images should be HxWx3 in RGB order.
        output_dir: The folder to put the video in.
        video_name: The name for the video.
        audios: raw audio files
        fps: Frames per second for the video. Not all values work with FFMPEG,
            use at your own risk.
        quality: Default is 5. Uses variable bit rate. Highest quality is 10,
            lowest is 0.  Set to None to prevent variable bitrate flags to
            FFMPEG so you can manually specify them using output_params
            instead. Specifying a fixed bitrate using ‘bitrate’ disables
            this parameter.
    """
    assert 0 <= quality <= 10
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"
    #assert len(images) == len(audios) * audio_duration * fps
    audio_clips = []
    temp_file_name = '/tmp/{}.wav'.format(random.randint(0, 10000))
    amplitude_scaling_factor = 1
    for i, audio in enumerate(audios):
        sf.write(temp_file_name, audio.T / amplitude_scaling_factor, sr)
        conv_log = os.popen(
            'wine64 /home/spurushw/AmbisonicBinauralizer.exe -i {} -o {} -a 0 '
            .format('/tmp', '/tmp/new')).read()
        bin_data, file_sr = sf.read(temp_file_name.replace('tmp/', 'tmp/new/'))
        audio_clip = mpy_audio.AudioClip.AudioArrayClip(bin_data, fps=sr)
        audio_clip = audio_clip.set_duration(audio_duration)
        audio_clip = audio_clip.set_start(i * audio_duration)
        audio_clips.append(audio_clip)
    composite_audio_clip = CompositeAudioClip(audio_clips)
    video_clip = mpy.ImageSequenceClip(images, fps=fps)
    video_with_new_audio = video_clip.set_audio(composite_audio_clip)
    video_with_new_audio.write_videofile(os.path.join(output_dir, video_name))
    os.remove(temp_file_name)


def IOU(output, target, balance=False):
    output_pred = output > 0
    target = target > 0
    poswt = 1.0 / float(len(target))
    negwt = 1.0 / float(len(target))
    if balance:
        poswt = (1.0 / (torch.sum(target.float()).item() + 1e-10))
        negwt = (1.0 / (torch.sum((~target).float()).item() + 1e-10))
    inters = (output_pred * target).sum(-1).sum(-1) * poswt
    union = inters + (output_pred * (~target)).sum(-1).sum(-1) * negwt + (
        (~output_pred) * target).sum(-1).sum(-1) * poswt
    iou = inters / (union + 1e-10)
    return iou.mean()


def AP(output, target, balance=False):
    target = target.numpy().flatten()
    output = output.numpy().flatten()
    sample_wt = np.ones_like(target) / float(len(target))
    if balance:
        n_pos = np.sum(target > 0)
        n_neg = len(target) - n_pos
        sample_wt[target > 0] = 0.5 / (float(n_pos) + 1e-10)
        sample_wt[target == 0] = 0.5 / (float(n_neg) + 1e-10)
    ap = sklearn.metrics.average_precision_score(target,
                                                 output,
                                                 sample_weight=sample_wt)
    return ap


def accuracy(output, target, topk=(1, ), balance=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        poswt = 1.0
        negwt = 1.0
        if balance:
            poswt = (0.5 * batch_size /
                     (torch.sum(target.float()).item() + 1e-10))
            negwt = (0.5 * batch_size /
                     (torch.sum((1 - target).float() + 1e-10)).item())

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        target = target.view(1, -1).expand_as(pred)
        correct = pred.eq(target)

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float()
            if balance:
                correct_k[target[0].view(-1) >
                          0] = correct_k[target[0].view(-1) > 0] * poswt
                correct_k[(1 - target[0]).view(-1) > 0] = correct_k[
                    (1 - target[0]).view(-1) > 0] * negwt
            correct_k = correct_k.sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def worker_init_fn(worker_id):
    seed = torch.initial_seed()
    print('Seeding worker {}: {}'.format(worker_id, int(seed)))
    np.random.seed(int(seed) % 2**30)


def double_data(x):
    if isinstance(x, list):
        if isinstance(x[0], str):
            return x + x
        return [double_data(xi) for xi in x]
    return torch.cat([x, x], 0)


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class TopDownFloorPlan(object):
    """Docstring for TopDownFloorPlan. """
    COORDINATE_EPSILON = 1e-6
    COORDINATE_MIN = -62.3241 - COORDINATE_EPSILON
    COORDINATE_MAX = 90.0399 + COORDINATE_EPSILON

    def __init__(self,
                 top_down_map,
                 level_heights,
                 clip_params,
                 downsample_factor=None):
        """TODO: to be defined1.

        :top_down_map: TODO
        :level_heights: TODO
        :clip_params: TODO

        """
        self._downsample_factor = downsample_factor
        if downsample_factor is not None:
            top_down_map = np.concatenate([
                cv2.resize(top_down_map[:, :, i],
                           (int(top_down_map.shape[1] / downsample_factor[1]),
                            int(top_down_map.shape[0] / downsample_factor[0])),
                           interpolation=cv2.INTER_NEAREST)[:, :, np.newaxis]
                for i in range(top_down_map.shape[2])
            ], 2)
        self._top_down_map = top_down_map
        self._level_heights = level_heights
        self._clip_params = clip_params
        assert (len(self._level_heights) == self._top_down_map.shape[2])

    def to_grid(
            self,
            realworld_x: float,
            realworld_y: float,
            realworld_z: float,
    ) -> Tuple[int, int]:
        r"""Return gridworld index of realworld coordinates assuming top-left corner
        is the origin. The real world coordinates of lower left corner are
        (coordinate_min, coordinate_min) and of top right corner are
        (coordinate_max, coordinate_max)
        """
        grid_size = (
            (self.COORDINATE_MAX - self.COORDINATE_MIN) /
            self._clip_params['map_resolution'][0],
            (self.COORDINATE_MAX - self.COORDINATE_MIN) /
            self._clip_params['map_resolution'][1],
        )
        grid_x = int((self.COORDINATE_MAX - realworld_x) / grid_size[0])
        grid_y = int((realworld_z - self.COORDINATE_MIN) / grid_size[1])
        #assert (grid_x <= self._clip_params['ind_x_max'])
        #assert (grid_y <= self._clip_params['ind_y_max'])

        grid_x = grid_x - (self._clip_params['ind_x_min'] -
                           self._clip_params['grid_delta'])
        grid_y = grid_y - (self._clip_params['ind_y_min'] -
                           self._clip_params['grid_delta'])
        if self._downsample_factor is not None:
            grid_x = int(grid_x / self._downsample_factor[0])
            grid_y = int(grid_y / self._downsample_factor[1])
        grid_z = np.argmin(np.abs(self._level_heights - realworld_y))
        return grid_x, grid_y, grid_z

    def from_grid(
            self,
            grid_x: int,
            grid_y: int,
            grid_z: int,
    ) -> Tuple[float, float]:
        r"""Inverse of _to_grid function. Return real world coordinate from
        gridworld assuming top-left corner is the origin. The real world
        coordinates of lower left corner are (coordinate_min, coordinate_min) and
        of top right corner are (coordinate_max, coordinate_max)
        """
        grid_size = (
            (self.COORDINATE_MAX - self.COORDINATE_MIN) /
            self._clip_params['map_resolution'][0],
            (self.COORDINATE_MAX - self.COORDINATE_MIN) /
            self._clip_params['map_resolution'][1],
        )
        if self._downsample_factor is not None:
            grid_x *= self._downsample_factor[0]
            grid_y *= self._downsample_factor[1]
        grid_x = grid_x + (self._clip_params['ind_x_min'] -
                           self._clip_params['grid_delta'])
        grid_y = grid_y + (self._clip_params['ind_y_min'] -
                           self._clip_params['grid_delta'])
        realworld_x = self.COORDINATE_MAX - grid_x * grid_size[0]
        realworld_z = self.COORDINATE_MIN + grid_y * grid_size[1]
        realworld_y = self._level_heights[grid_z]
        return realworld_x, realworld_y, realworld_z


def load_points(points_file: str, transform=True):
    points_data = np.loadtxt(points_file, delimiter="\t")
    if transform:
        points = list(
            zip(points_data[:, 1], points_data[:, 3] + -(1.4410627 + 0.111828),
                -points_data[:, 2]))
    else:
        points = list(
            zip(points_data[:, 1], points_data[:, 2], points_data[:, 3]))
    points_index = points_data[:, 0].astype(int)
    points_dict = dict(zip(points_index, points))
    assert list(points_index) == list(range(len(points)))
    return points_dict, points


def load_points_data(parent_folder, transform=False):
    points_file = os.path.join(parent_folder, 'points.txt')
    graph_file = os.path.join(parent_folder, 'graph_without_diagonal.pkl')

    _, points = load_points(points_file, transform=transform)
    if not os.path.exists(graph_file):
        raise FileExistsError(graph_file + ' does not exist!')
    else:
        with open(graph_file, 'rb') as fo:
            graph = pickle.load(fo)

    return points, graph


def get_ambisonics_file_path(ir_dir, source_id, receiver_id):
    fname = os.path.join(ir_dir, '{}_{}.wav'.format(receiver_id, source_id))
    if os.path.exists(fname):
        return fname
    return fname.replace('scratch', 'glusterfs')


def cart2spherical(xyz):
    ptsnew = np.zeros(xyz.shape)
    xy = xyz[:, 0]**2 + xyz[:, 1]**2
    ptsnew[:, 0] = np.sqrt(xy + xyz[:, 2]**2)
    ptsnew[:, 1] = np.arctan2(
        np.sqrt(xy), xyz[:, 2])  # for elevation angle defined from Z-axis down
    # ptsnew[:,1] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:, 2] = np.arctan2(xyz[:, 1], xyz[:, 0])
    return ptsnew


def cart2logspherical(xyz):
    ptsnew = cart2spherical(xyz)
    ptsnew[:, 0] = np.log(ptsnew)
    return ptsnew


def wallExistsBetween(top_down_map, pt1, pt2):
    count = 0
    for pt in fow.bresenham_supercover_line(pt1, pt2):
        x, y = pt
        if top_down_map[x, y] == maps.MAP_INVALID_POINT:
            count += 1
    if count > 5:
        return True
    return False


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    #cY, cX = center
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    translation = ((nH / 2) - cY, (nW / 2) - cX)
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH),
                          flags=cv2.INTER_NEAREST), translation


def translate_and_rotate(x, paths):
    paths = paths.view(-1, 4)
    x = x.float()
    rotation = paths[:, -1].float()
    translation = torch.flip(paths[:, :2], (1, )).float()
    center = (torch.tensor(x.shape[-2:]).float().to(rotation.device) //
              2).expand(x.shape[0], -1)
    x = korntransforms.translate(korntransforms.rotate(x,
                                                       angle=rotation,
                                                       center=center),
                                 translation=translation)
    return x


def rotate_origin_only(xy, degrees):
    """Only rotate a point around the origin (0, 0)."""
    radians = np.pi * (degrees / float(180))
    x, y = xy
    xx = x * math.cos(radians) + y * math.sin(radians)
    yy = -x * math.sin(radians) + y * math.cos(radians)

    return np.array([xx, yy])


def get_triangle_coords(centroid, angle, scale=64):
    front = rotate_origin_only(np.array([-1 * scale // 30, 0]),
                               angle) + centroid
    right = rotate_origin_only(np.array([scale // 30, scale // 60]),
                               angle) + centroid
    left = rotate_origin_only(np.array([scale // 30, -1 * scale // 60]),
                              angle) + centroid
    pts = np.vstack((front, left, right)).astype(np.int32).reshape((-1, 1, 2))
    return pts


def bresenham_supercover_line(pt1, pt2):
    r"""Line drawing algo based
    on http://eugen.dedu.free.fr/projects/bresenham/
    """

    ystep, xstep = 1, 1

    x, y = pt1
    dx, dy = pt2 - pt1
    if dy < 0:
        ystep *= -1
        dy *= -1
    if dx < 0:
        xstep *= -1
        dx *= -1

    line_pts = [[x, y]]
    ddx, ddy = 2 * dx, 2 * dy
    if ddx > ddy:
        errorprev = dx
        error = dx
        for _ in range(int(dx)):
            x += xstep
            error += ddy
            if error > ddx:
                y += ystep
                error -= ddx
                if error + errorprev < ddx:
                    line_pts.append([x, y - ystep])
                elif error + errorprev > ddx:
                    line_pts.append([x - xstep, y])
                else:
                    line_pts.append([x - xstep, y])
                    line_pts.append([x, y - ystep])
            line_pts.append([x, y])
            errorprev = error
    else:
        errorprev = dx
        error = dx
        for _ in range(int(dy)):
            y += ystep
            error += ddx
            if error > ddy:
                x += xstep
                error -= ddy
                if error + errorprev < ddy:
                    line_pts.append([x - xstep, y])
                elif error + errorprev > ddy:
                    line_pts.append([x, y - ystep])
                else:
                    line_pts.append([x - xstep, y])
                    line_pts.append([x, y - ystep])
            line_pts.append([x, y])
            errorprev = error

    return line_pts


def get_fow_line_values(mapdata, pt1, pt2):
    values = []
    for pt in bresenham_supercover_line(pt1, pt2):
        x, y = pt
        if x < 0 or x >= mapdata.shape[0]:
            break
        if y < 0 or y >= mapdata.shape[1]:
            break
        values.append(mapdata[int(x), int(y)])
    return np.array(values)
