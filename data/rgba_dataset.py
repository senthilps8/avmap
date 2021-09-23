import torch
import torch.nn.functional as F
import data.fog_of_war as fog_of_war
import torch.utils.data as data
import torchvision.transforms as transforms

from tqdm import tqdm
import copy
import os
import soundfile as sf
from .utils import *
from .audio_utils import *
import networkx as nx
import numpy as np
import pickle as pkl
import json
import scipy
from collections import defaultdict


class SequenceRGBAmbisonicsDataset(data.Dataset):
    """Docstring for SequenceRGBAmbisonicsDataset. """
    def __init__(self, cfg, test_set=False):
        """TODO: to be defined. """
        data.Dataset.__init__(self)

        self.test_set = test_set
        self.cfg = cfg
        self._audio_dir = cfg.audio_dir
        self._obs_dir = cfg.obs_dir

        self._output_gridsize = np.array(cfg.output_gridsize)
        self._disable_audio = cfg.disable_audio
        self._duration = cfg.duration
        self._target_nhood = cfg.out_nhood
        self._angle_range = np.array(cfg.angle_range)
        self._source_at_receiver = cfg.source_at_receiver
        self._n_steps = cfg.n_steps
        self._convolve_audio_clip = cfg.convolve_audio_clip

        with open(cfg.train_env_list_file, 'r') as f:
            train_envlist = f.read().splitlines()
        with open(cfg.val_env_list_file, 'r') as f:
            val_envlist = f.read().splitlines()

        train_augmentation = [
            transforms.Resize(cfg.rgb_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
        test_augmentation = [
            transforms.Resize(cfg.rgb_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
        train_rgb_transform = transforms.Compose(train_augmentation)
        test_rgb_transform = transforms.Compose(test_augmentation)
        audio_transform = None

        self.rgb_transform = train_rgb_transform
        self._env_list = train_envlist
        if test_set:
            self.rgb_transform = test_rgb_transform
            self._env_list = val_envlist
        self.audio_transform = audio_transform
        self._pool_steps = cfg.pool_steps
        self._batch_size = cfg.batch_size
        self._epoch_size = cfg.epoch_size
        self._convolution_mode = cfg.convolution_mode

        self._env_list = [
            e for e in self._env_list
            if os.path.exists(self._get_topdownmap_filepath(e))
        ]
        print(self._env_list)

        # Load and Store topdown maps
        print('Loading TopDown Maps...')
        angle_step = 30
        angle_choice = np.arange(self._angle_range[0], self._angle_range[1],
                                 angle_step)
        self.data = []
        self.env_points = {}
        self.source_graph = {}
        self.env_graphs = {}
        self.top_down_maps = {}
        self.semantic_top_down_maps = {}
        self.top_down_areas = {}
        self.semantic_top_down_maps = {}
        self.path_graphs = {}
        for env in self._env_list:
            map_data = pkl.load(open(self._get_topdownmap_filepath(env), 'rb'))

            # Erode to make the interior walls clearer
            for i in range(map_data['top_down_map'].shape[-1]):
                map_data['top_down_map'][:, :,
                                         i] = scipy.ndimage.binary_erosion(
                                             map_data['top_down_map'][:, :, i],
                                             iterations=1)

            self.top_down_maps[env] = TopDownFloorPlan(
                map_data['top_down_map'],
                map_data['level_height'],
                map_data['clip_params'],
                downsample_factor=None)
            self.top_down_areas[env] = (self.top_down_maps[env]._top_down_map >
                                        0).sum(0).sum(0)
            self.semantic_top_down_maps[env] = TopDownFloorPlan(
                map_data['semantic_map'] * map_data['top_down_map'],
                map_data['level_height'],
                map_data['clip_params'],
                downsample_factor=None)
            self.mp3d_categories = map_data['categories']

        # Retain frequent and unambiguous categories
        self.categories = [
            'bathroom', 'hallway', 'bedroom', 'stairs', 'kitchen',
            'living room', 'entryway/foyer/lobby', 'dining room', 'closet',
            'office', 'lounge', 'laundryroom/mudroom', 'workout/gym/exercise'
        ]
        self.remove_mp3dcats = [
            'balcony', 'porch/terrace/deck', 'junk', 'outdoor'
        ]
        self.remove_mp3dcat_ids = [
            self.mp3d_categories.index(c) + 1 for c in self.remove_mp3dcats
        ]
        self.combine_mp3dcats = [('toilet', 'bathroom'),
                                 ('familyroom/lounge', 'lounge')]
        self.mp3dcat_to_cat_mapping = np.zeros(len(self.mp3d_categories) + 1)
        for ci, cls in enumerate(self.categories):
            self.mp3dcat_to_cat_mapping[self.mp3d_categories.index(cls) +
                                        1] = self.categories.index(cls) + 1
        for cls in self.remove_mp3dcats:
            self.mp3dcat_to_cat_mapping[self.mp3d_categories.index(cls) +
                                        1] = 0
        for cls1, cls2 in self.combine_mp3dcats:
            self.mp3dcat_to_cat_mapping[self.mp3d_categories.index(cls1) +
                                        1] = self.categories.index(cls2) + 1

        # Remap categories
        self.n_categories = len(self.categories)

        # Clean up the graph and remove nodes which are outdoor
        for env in tqdm(self._env_list, desc='Environments'):
            self.env_points[env] = {}
            try:
                graph = pkl.load(open(self._get_graph_filepath(env), 'rb'))
            except Exception as e:
                continue
            self.env_graphs[env] = graph
            all_nodes = list(graph.nodes())
            for nr in all_nodes:
                pt_r = self.env_graphs[env].nodes[nr]['point']
                maploc_r = self._get_maploc_from_point(env, pt_r)
                td_pixel = self.top_down_maps[env]._top_down_map[
                    maploc_r[0], maploc_r[1], maploc_r[2]]
                semantic_td_pixel = self.semantic_top_down_maps[
                    env]._top_down_map[maploc_r[0], maploc_r[1], maploc_r[2]]
                if (not td_pixel > 0) or (
                        semantic_td_pixel in self.remove_mp3dcat_ids):
                    self.env_graphs[env].remove_node(nr)
                    continue
                self.env_points[env][nr] = np.array(pt_r)

        # Construct a graph with nodes as points+orientation in the environment
        # Edges between nodes that involve single rotation or single step forward
        for env in tqdm(self._env_list, desc='Constructing Path graphs'):
            path_graph = nx.DiGraph()
            graph = self.env_graphs[env]
            for n in graph.nodes():
                path_graph.add_nodes_from([(n, ang) for ang in angle_choice])
                for ang in angle_choice:
                    path_graph.add_edge((n, ang),
                                        (n, (ang - angle_step) % 360),
                                        movement='rotate')
                    path_graph.add_edge((n, ang),
                                        (n, (ang + angle_step) % 360),
                                        movement='rotate')
                for n2 in graph.neighbors(n):
                    map_loc1 = self._get_maploc_from_index(env, n)
                    map_loc2 = self._get_maploc_from_index(env, n2)
                    map_diff = map_loc2[:2] - map_loc1[:2]
                    angle = np.arctan2(map_diff[0], -1 * map_diff[1])
                    angle = angle * 180 / np.pi
                    if angle < 0:
                        angle = angle + 360
                    angle_diff = np.abs(angle_choice - angle)
                    best_angles = angle_choice[angle_diff <= (angle_step +
                                                              1e-2)]
                    for ang in best_angles:
                        path_graph.add_edge((n, ang), (n2, ang),
                                            movement='translate')
            self.path_graphs[env] = path_graph

        # Construct a graph for possible source locations in a neighborhood
        # "nearby" in paper
        for env in tqdm(self._env_list, desc='Constructing Source graphs'):
            graph = self.env_graphs[env]
            self.source_graph[env] = nx.Graph()
            self.source_graph[env].add_nodes_from(graph)
            for ns in graph.nodes():
                pt_s = graph.nodes[ns]['point']
                map_s = np.array(self.top_down_maps[env].to_grid(*pt_s))
                for nr in graph.nodes():
                    pt_r = graph.nodes[nr]['point']
                    map_r = np.array(self.top_down_maps[env].to_grid(*pt_r))
                    rs_diff = np.abs(map_s - map_r)
                    # Circle of radius nhood)
                    if (np.sqrt(rs_diff[0]**2 + rs_diff[1]**2) <
                            self._target_nhood // 2 - 1 and rs_diff[2] == 0):
                        self.source_graph[env].add_edge(nr, ns)

        self._orig_n_steps = max(self._n_steps, 2)
        step_maploc1 = self._get_maploc_from_point(self._env_list[0],
                                                   np.array([0, 1.5, 0]))
        step_maploc2 = self._get_maploc_from_point(self._env_list[0],
                                                   np.array([1, 1.5, 1]))
        step_size = self._get_sampleloc_from_maploc(self._env_list[0],
                                                    step_maploc1, step_maploc2,
                                                    0)
        self.step_size = np.amax(np.abs(step_size -
                                        self._output_gridsize // 2))
        print('Step size: {}'.format(self.step_size))
        self.padding = int(
            np.ceil(self.step_size * (self._n_steps - 1) * np.sqrt(2) +
                    (np.amax(self._output_gridsize) / 2.0) *
                    (np.sqrt(2) - 1)) * int(self._n_steps > 1))

        # Initialize audio clips
        # Only used if audio is turned on
        if self.test_set:
            self.audio_clip_cfg = cfg.test_audio_clip
            self.audio_clip_func = audio_loaders[self.audio_clip_cfg.func]
            self.audio_clip_args = self.audio_clip_cfg.args
        else:
            self.audio_clip_cfg = cfg.train_audio_clip
            self.audio_clip_func = audio_loaders[self.audio_clip_cfg.func]
            self.audio_clip_args = self.audio_clip_cfg.args

        self.n_sources = int(cfg.n_sources)
        # If sources > 1, source can not be at receiver (all room setting)
        assert (not (self.n_sources > 1 and self._source_at_receiver))

        # Collect room label -> list of node,level mappings
        # for all room setting
        self.source_room_labels = {}
        for env in self._env_list:
            self.source_room_labels[env] = defaultdict(list)
            nodes = list(self.source_graph[env].nodes())
            for ns in nodes:
                pt_s = self.env_graphs[env].nodes[ns]['point']
                maploc_s = self._get_maploc_from_point(env, pt_s)
                semantic_td_pixel = self.semantic_top_down_maps[
                    env]._top_down_map[maploc_s[0], maploc_s[1], maploc_s[2]]
                self.source_room_labels[env][int(semantic_td_pixel)].append(
                    (ns, maploc_s[2]))

    def _get_topdownmap_filepath(self, env):
        return os.path.join(self._obs_dir, env, 'topdownmap',
                            '{}_occant_semantic.pkl'.format(env))

    def _get_graph_filepath(self, env):
        return os.path.join(self._obs_dir, env, 'graph',
                            '{}_freeobjects.pkl'.format(env))

    def _get_rgb_filepath(self, env, nr, ang_r=0):
        return os.path.join(self._obs_dir, env, 'rgb',
                            '{:05d}_angle{:03d}.jpg'.format(nr, ang_r))

    def _get_agentinfo_filepath(self, env, nr, ang_r=0):
        return os.path.join(
            self._obs_dir, env, 'agent_info',
            '{:05d}_angle{:03d}_freeobjects.json'.format(nr, ang_r))

    def _get_maploc_from_point(self, env, pt_r):
        # Get top down coordinates from real world point
        map_r = np.array(self.top_down_maps[env].to_grid(*pt_r))
        return map_r

    def _get_point_from_maploc(self, env, maploc):
        real_pt = np.array(self.top_down_maps[env].from_grid(
            maploc[0], maploc[1], maploc[2]))
        return real_pt

    def _get_maploc_from_index(self, env, nr):
        # Get top down coordinates from node index
        pt = self.env_points[env][nr]
        return self._get_maploc_from_point(env, pt)

    def _get_maploc_from_location_in_sample(self, env, map_r, loc, ang_r=0):
        sample_rel_loc = loc - np.array(
            [self._output_gridsize[0] // 2, self._output_gridsize[1] // 2])
        scale_dim1 = self._output_gridsize[0] / self._target_nhood
        scale_dim2 = self._output_gridsize[1] / self._target_nhood
        sample_rel_loc[0] /= scale_dim1
        sample_rel_loc[1] /= scale_dim2

        theta = np.deg2rad(ang_r)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        rel_loc = np.dot(R, sample_rel_loc[:2])

        map_loc = np.array(
            [map_r[0] + rel_loc[0], map_r[1] + rel_loc[1], map_r[2]])
        return map_loc

    def _get_sampleloc_from_maploc(self, env, map_r, map_s, ang_r=0):
        rel_loc = map_s - map_r
        theta = np.deg2rad(-1 * ang_r)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        rotated_rel_loc = np.dot(R, rel_loc[:2])
        nhood = self._target_nhood
        scale_dim1 = self._output_gridsize[0] / nhood
        scale_dim2 = self._output_gridsize[1] / nhood
        out_loc = np.array(
            [rotated_rel_loc[0] * scale_dim1, rotated_rel_loc[1] * scale_dim2])
        out_loc = out_loc + np.array(
            [self._output_gridsize[0] // 2, self._output_gridsize[1] // 2])
        return out_loc

    def _get_random_path(self):
        env = np.random.choice(self._env_list)
        env_nodes = list(self.path_graphs[env].nodes)

        for overall_trial_i in range(100):
            start_node = env_nodes[np.random.randint(len(env_nodes))]
            path = [start_node]
            for stepi in range(self._orig_n_steps):
                edges = list(self.path_graphs[env].edges(path[-1], data=True))
                move_nbors = [
                    n[1] for n in edges if n[2]['movement'] == 'translate'
                ]
                rot_nbors = [
                    n[1] for n in edges if n[2]['movement'] == 'rotate'
                ]
                for trial_i in range(10):
                    if np.random.rand() > 0.1 and len(move_nbors):
                        nbors = move_nbors
                    else:
                        nbors = rot_nbors
                    sampled_nbor = nbors[np.random.randint(len(nbors))]
                    if sampled_nbor not in path:
                        break
                path.append(sampled_nbor)

            sem_gt1 = self._get_semantic_target(
                env, self._get_maploc_from_index(env, start_node[0]))
            sem_gt2 = self._get_semantic_target(
                env, self._get_maploc_from_index(env, path[-1][0]))
            # Ensure atleast 2 room types in neighborhood
            if not (len(np.unique(sem_gt1[0].numpy())) >= 2
                    or len(np.unique(sem_gt2[0].numpy())) >= 2):
                continue

            path_str = ['{}_{}'.format(*p) for p in path]
            if len(path_str) == len(np.unique(path_str)):
                break

        angles = [p[1] for p in path]
        path = [p[0] for p in path]
        path = path[:self._n_steps]
        angles = angles[:self._n_steps]
        level = self._get_maploc_from_index(env, start_node[0])[-1]

        # Select source locations -
        # List of sources for each step
        if self._source_at_receiver:
            allns = [[p] for p in path]
        elif self.n_sources < 20:
            sources = []
            for _ in range(self.n_sources):
                random_rec = path[np.random.randint(len(path))]
                nearby_sources = list(
                    self.source_graph[env].neighbors(random_rec))
                ns = nearby_sources[np.random.randint(len(nearby_sources))]
                sources.append(ns)
            allns = [sources] * len(path)
        else:  # (set n_sources>20 for allroom setting)
            sources = []
            for room_id, room_sources in self.source_room_labels[env].items():
                room_sources = [r[0] for r in room_sources if r[1] == level]
                if not len(room_sources):
                    continue
                sources.append(room_sources[np.random.randint(
                    len(room_sources))])
            allns = [sources] * len(path)
        return env, path, angles, allns

    def _get_rgb_obs(self, env, nr, ang_r=0):
        return pil_loader(self._get_rgb_filepath(env, nr, ang_r))

    def _get_single_step_signal(self, env, nr, allns, ang_r=0, index=0):
        # Identify rooms of sources
        maploc_s = [self._get_maploc_from_index(env, ns) for ns in allns]
        semantic_td_pixel = [
            self.semantic_top_down_maps[env].
            _top_down_map[maploc_s[i][0], maploc_s[i][1], maploc_s[i][2]]
            for i in range(len(maploc_s))
        ]
        source_rooms = [
            self.categories[int(self.mp3dcat_to_cat_mapping[int(pix)]) -
                            1] if pix > 0 else 'void'
            for pix in semantic_td_pixel
        ]

        # Load audio
        ir_paths = [
            get_ambisonics_file_path(os.path.join(self._audio_dir, env, 'irs'),
                                     nr, ns) for ns in allns
        ]
        allsignals = []
        for si, ir_path in enumerate(ir_paths):
            if self._disable_audio:
                signal = np.zeros((self._duration, 9))
                fs = 16000
            else:
                try:
                    [signal, fs] = sf.read(ir_path)
                except:
                    print('could not load {}'.format(ir_path))
                    signal = np.zeros((self._duration, 9))
                    fs = 16000
            signal = signal[:self._duration, :]
            signal = rotate_ambisonics(signal, -1 * ang_r)
            if signal.shape[0] < self._duration:
                signal = np.pad(signal, ((0, self._duration - signal.shape[0]),
                                         (0, 0)))
            if self._convolve_audio_clip:
                clip = self.audio_clip_func(
                    *self.audio_clip_args,
                    index=index,
                    meta={'room': source_rooms[si].replace(' ', '_')})
                signal = convolve_rir(signal,
                                      clip,
                                      mode=self._convolution_mode)
            allsignals.append(signal)
        allsignals = np.stack(allsignals, 2)
        audio_signal = torch.from_numpy(allsignals).float().permute(1, 0, 2)

        # Load RGB, Audio
        rgb = self._get_rgb_obs(env, nr, ang_r)
        if self.rgb_transform is not None:
            rgb = self.rgb_transform(rgb)
        if self.audio_transform is not None:
            audio_signal = self.audio_transform(audio_signal)
        return audio_signal, rgb

    def _get_signal(self, env, all_nr, all_ns, all_ang_r, index=0):
        out_signal = [
            self._get_single_step_signal(env,
                                         all_nr[i],
                                         all_ns[i],
                                         all_ang_r[i],
                                         index=index)
            for i in range(len(all_nr))
        ]
        out_signal = list(zip(*out_signal))
        out_signal = [torch.stack(signal) for signal in out_signal]
        out_signal[0] = out_signal[0].sum(-1)
        return out_signal

    def _get_top_down_map(self, env):
        return self.top_down_maps[env]._top_down_map

    def _get_semantic_top_down_map(self, env):
        return self.semantic_top_down_maps[env]._top_down_map

    def _get_target(self, env, map_r, ang_r=0):
        top_down_map = (self._get_top_down_map(env) > 0).astype(np.uint8)
        max_nhood = self._target_nhood

        pad_above = map_r[0] - max_nhood
        pad_below = top_down_map.shape[0] - (map_r[0] + max_nhood)
        pad_left = map_r[1] - max_nhood
        pad_right = top_down_map.shape[1] - (map_r[1] + max_nhood)
        pad_above = int(pad_above < 0) * pad_above * -1 + 1
        pad_below = int(pad_below < 0) * pad_below * -1 + 1
        pad_left = int(pad_left < 0) * pad_left * -1 + 1
        pad_right = int(pad_right < 0) * pad_right * -1 + 1
        top_down_map = np.pad(top_down_map, ((pad_above, pad_below),
                                             (pad_left, pad_right), (0, 0)))

        target = top_down_map[pad_above + map_r[0] - max_nhood:pad_above +
                              map_r[0] + max_nhood + 1, pad_left + map_r[1] -
                              max_nhood:pad_left + map_r[1] + max_nhood +
                              1, map_r[2]]
        target, trans = rotate_bound(target, ang_r)
        center = target.shape[0] // 2, target.shape[1] // 2
        target = target[center[0] - max_nhood // 2:center[0] + max_nhood // 2 +
                        max_nhood % 2, center[0] - max_nhood // 2:center[0] +
                        max_nhood // 2 + max_nhood % 2]
        center = target.shape[0] // 2, target.shape[1] // 2

        nhood = self._target_nhood
        this_target = torch.from_numpy(
            target[center[0] - nhood // 2:center[0] + nhood // 2 +
                   nhood % 2, center[0] - nhood // 2:center[0] + nhood // 2 +
                   nhood % 2]).float()
        this_target = F.interpolate(
            this_target.unsqueeze(0).unsqueeze(0),
            size=tuple(self._output_gridsize)).squeeze(0).squeeze(0)
        this_target = this_target.long()
        return this_target

    def _get_semantic_target(self, env, map_r, ang_r=0):
        top_down_map = self._get_semantic_top_down_map(env).astype(np.uint8)
        max_nhood = self._target_nhood
        pad_above = map_r[0] - max_nhood
        pad_below = top_down_map.shape[0] - (map_r[0] + max_nhood)
        pad_left = map_r[1] - max_nhood
        pad_right = top_down_map.shape[1] - (map_r[1] + max_nhood)
        pad_above = int(pad_above < 0) * pad_above * -1 + 1
        pad_below = int(pad_below < 0) * pad_below * -1 + 1
        pad_left = int(pad_left < 0) * pad_left * -1 + 1
        pad_right = int(pad_right < 0) * pad_right * -1 + 1
        top_down_map = np.pad(top_down_map, ((pad_above, pad_below),
                                             (pad_left, pad_right), (0, 0)))

        target = top_down_map[pad_above + map_r[0] - max_nhood:pad_above +
                              map_r[0] + max_nhood + 1, pad_left + map_r[1] -
                              max_nhood:pad_left + map_r[1] + max_nhood +
                              1, map_r[2]]
        target = self.mp3dcat_to_cat_mapping[target.astype(np.int32)]

        target, trans = rotate_bound(target, ang_r)
        center = target.shape[0] // 2, target.shape[1] // 2
        target = target[center[0] - max_nhood // 2:center[0] + max_nhood // 2 +
                        max_nhood % 2, center[0] - max_nhood // 2:center[0] +
                        max_nhood // 2 + max_nhood % 2]
        center = target.shape[0] // 2, target.shape[1] // 2
        nhood = self._target_nhood
        this_target = torch.from_numpy(
            target[center[0] - nhood // 2:center[0] + nhood // 2 +
                   nhood % 2, center[0] - nhood // 2:center[0] + nhood // 2 +
                   nhood % 2]).float()
        this_target = F.interpolate(
            this_target.unsqueeze(0).unsqueeze(0),
            size=tuple(self._output_gridsize)).squeeze(0).squeeze(0)
        this_target = this_target.long()
        return this_target

    def __getitem__(self, index):
        env, path, angles, allns = self._get_random_path()
        return self.getitem(env, path, angles, allns, index=index)

    def getitem(self, env, path, angles, allns, index):
        out_signal = list(
            self._get_signal(env, path, allns, angles, index=index))
        map_loc_rs = [self._get_maploc_from_index(env, nr) for nr in path]
        sample_loc_rs = [
            self._get_sampleloc_from_maploc(env, map_loc_rs[0], loc, angles[0])
            for loc in map_loc_rs
        ]
        relpath = np.stack([
            np.concatenate((sloc - sample_loc_rs[0], np.array([0]),
                            np.array([angles[li] - angles[0]])))
            for li, sloc in enumerate(sample_loc_rs)
        ])
        out_signal.append(relpath)

        # Create/Load Targets
        alltargets = torch.cat([
            self._get_target(env, map_loc_rs[ri],
                             angles[ri]).unsqueeze(0).unsqueeze(0)
            for ri in range(len(map_loc_rs))
        ], 0)
        semantic_alltargets = torch.cat([
            self._get_semantic_target(env, map_loc_rs[ri],
                                      angles[ri]).unsqueeze(0).unsqueeze(0)
            for ri in range(len(map_loc_rs))
        ], 0)

        # Pad, Translate, Align (and maybe pool)
        # Pad
        predictable_target = torch.ones_like(alltargets)
        alltargets = F.pad(alltargets, [self.padding] * 4)
        semantic_alltargets = F.pad(semantic_alltargets, [self.padding] * 4)
        predictable_target = F.pad(predictable_target, [self.padding] * 4)
        # Align
        aligned_alltargets = (translate_and_rotate(
            alltargets, torch.from_numpy(relpath)) > 0).long()
        aligned_semantic_alltargets = (translate_and_rotate(
            semantic_alltargets, torch.from_numpy(relpath))).long()
        predictable_target = (translate_and_rotate(
            predictable_target, torch.from_numpy(relpath)) > 0).long()

        # Pool?
        pooled_alltargets = aligned_alltargets.max(0).values.squeeze()
        pooled_semantic_alltargets = aligned_semantic_alltargets.max(
            0).values.squeeze()
        if self._pool_steps:
            target = pooled_alltargets
            semantic_target = pooled_semantic_alltargets
            predictable_target = predictable_target.max(0).values.squeeze()
        else:
            target = aligned_alltargets.squeeze()
            semantic_target = aligned_semantic_alltargets.squeeze()
            predictable_target = predictable_target.squeeze()

        # Other relevant signals
        map_loc_s = [[self._get_maploc_from_index(env, nsi) for nsi in ns]
                     for ns in allns]
        map_loc_s = [
            map_loc_si + [np.array(map_loc_si[-1])] *
            (self.n_sources - len(map_loc_si)) for map_loc_si in map_loc_s
        ]

        meta = {}
        meta['env'] = env
        meta['env_area'] = self.top_down_areas[env][
            map_loc_rs[0][2]] * np.amax(
                self._output_gridsize) / self._target_nhood
        meta['ns'] = [ns + [-1] * (self.n_sources - len(ns)) for ns in allns]
        meta['nr'] = path
        meta['maploc_r'] = map_loc_rs
        meta['maploc_s'] = map_loc_s
        meta['ang_r'] = angles
        meta['source_sample_loc'] = [[
            self._get_sampleloc_from_maploc(
                env,
                map_loc_rs[0],
                map_loc_s[ti][si],
                angles[0],
            ) + self.padding for si in range(len(map_loc_s[ti]))
        ] for ti in range(len(map_loc_s))]
        meta['rec_sample_loc'] = [
            self._get_sampleloc_from_maploc(
                env,
                map_loc_rs[0],
                map_loc_rs[ri],
                angles[0],
            ) + self.padding for ri in range(len(map_loc_rs))
        ]
        meta['angles'] = angles
        meta['step_size'] = self.step_size
        meta['sequence_padding'] = self.padding
        meta['predictable_target'] = predictable_target
        meta['semantic_target'] = semantic_target

        return out_signal, target, meta

    def __len__(self):
        return self._epoch_size * 2 * self._batch_size


class ValidationSequenceRGBAmbisonicsDataset(SequenceRGBAmbisonicsDataset):
    """Docstring for ValidationSequenceRGBAmbisonicsDataset. """
    def __init__(self, paths, cfg):
        """TODO: to be defined.

        :paths: TODO
        :cfg: TODO

        """
        SequenceRGBAmbisonicsDataset.__init__(self, cfg, test_set=True)
        self._paths = paths
        self._cfg = cfg
        n_steps = len(self._paths[0][1])
        cfg.n_steps = n_steps
        print('Creating Evaluationg Set with steps = {}'.format(n_steps))
        # Some paths were computed before erosion
        good_inds = []
        for pi, path in enumerate(self._paths):
            pts = self.env_points[path[0]]
            try:
                allpt = [pts[nr] for nr in path[1]]
                if isinstance(path[3], list):
                    pt = [pts[p] for p in path[3]]
                else:
                    pt = pts[path[3]]
                good_inds.append(pi)
            except:
                continue
        self._paths = [self._paths[pi] for pi in good_inds]

    def __getitem__(self, index):
        env, path, angles, ns = self._paths[index]
        if self._source_at_receiver:
            allns = [[p] for p in path]
        else:
            allns = [ns[:self.n_sources]] * len(path)
        return self.getitem(env, path, angles, allns, index=index)

    def __len__(self):
        return len(self._paths)


def getRGBAmbisonicsAreaDataset(origcfg):
    cfg = copy.deepcopy(origcfg)
    dataset_func = SequenceRGBAmbisonicsDataset
    trainset = dataset_func(cfg, test_set=False)
    all_valsets = None
    dataset_func = ValidationSequenceRGBAmbisonicsDataset
    paths = pkl.load(open(cfg.full_eval_path, 'rb'))
    for k in paths.keys():
        paths[k] = sorted(paths[k], key=lambda x: x[0])
    all_valsets = {}
    for k, v in paths.items():
        if k in cfg.full_eval_nsteps:
            cfg.n_steps = k
            valset = dataset_func(v, cfg)
            all_valsets[k] = valset
    return trainset, all_valsets
