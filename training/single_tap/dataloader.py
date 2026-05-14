from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union, cast
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler, Subset, random_split

SampleItemValue = Union[torch.Tensor, str, int]
SampleItem = Dict[str, SampleItemValue]

@dataclass(frozen=True)
class NeuroTacMeta:
    sensor_width: int = 320
    sensor_height: int = 320
    spatial_crop_x: int = 155
    spatial_crop_y: int = 155
    spatial_crop_radius: int = 75
    camera_period_us: int = 100


def _target_column_indices(target_mode: str) -> Tuple[List[int], List[str]]:
    target_mode = target_mode.lower()
    if target_mode == 'force_z':
        return [3], ['Force_Z']
    if target_mode == 'force_xy':
        return [1, 2], ['Force_X', 'Force_Y']
    if target_mode == 'force_xyz':
        return [1, 2, 3], ['Force_X', 'Force_Y', 'Force_Z']
    raise ValueError("target_mode must be one of: 'force_z', 'force_xy', 'force_xyz'")


def _duration_ms_to_num_bins(max_duration_ms: Optional[float], camera_period_us: int) -> Optional[int]:
    if max_duration_ms is None:
        return None

    if max_duration_ms <= 0:
        raise ValueError('max_duration_ms must be positive when provided.')
    if camera_period_us <= 0:
        raise ValueError('camera_period_us must be positive.')

    max_duration_us = float(max_duration_ms) * 1000.0
    max_bins = int(np.floor(max_duration_us / float(camera_period_us)))
    return max(1, max_bins)


class NeuroTacForceDataset(Dataset):
    """Lava-dl-compatible dataset for processed NeuroTac tactile events and force targets.

    Each sample constructs spike tensors as `[channels, height, width, time]` and
    returns them flattened to `[channels * height * width, time]`.
    """
    def __init__(
        self,
        processed_data_path: Union[str, Path],
        target_mode: str = 'force_z',
        use_uncropped: bool = False,
        binary_spikes: bool = True,
        dtype: torch.dtype = torch.float32,
        spike_dtype: torch.dtype = torch.uint8,
    ) -> None:
        
        self.processed_data_path = Path(processed_data_path)
        self.use_uncropped = use_uncropped
        self.meta = self._load_meta()
        self.binary_spikes = binary_spikes
        self.dtype = dtype
        self.spike_dtype = spike_dtype
        self.target_indices, self.target_names = _target_column_indices(target_mode)
        self.tactile_suffix = '_tactile_uncropped.npy' if self.use_uncropped else '_tactile.npy'

        self.sample_names = sorted(
            path.name.replace(self.tactile_suffix, '')
            for path in self.processed_data_path.glob(f'*{self.tactile_suffix}')
        )

        if len(self.sample_names) == 0:
            raise ValueError(f'No processed samples found in {self.processed_data_path}')

        self.x_min = max(0, self.meta.spatial_crop_x - self.meta.spatial_crop_radius)
        self.x_max = min(self.meta.sensor_width - 1, self.meta.spatial_crop_x + self.meta.spatial_crop_radius)
        self.y_min = max(0, self.meta.spatial_crop_y - self.meta.spatial_crop_radius)
        self.y_max = min(self.meta.sensor_height - 1, self.meta.spatial_crop_y + self.meta.spatial_crop_radius)

        self.output_width = self.x_max - self.x_min + 1
        self.output_height = self.y_max - self.y_min + 1
        self.input_shape = (2, self.output_height, self.output_width)
        self.sample_lengths = self._load_sample_lengths()

    def __len__(self) -> int:
        return len(self.sample_names)

    def __getitem__(self, index: int) -> SampleItem:
        sample_name = self.sample_names[index]
        tactile_path = self.processed_data_path / f'{sample_name}{self.tactile_suffix}'
        ft_path = self.processed_data_path / f'{sample_name}_ft.npy'
        ft_data = np.load(ft_path)

        if ft_data.ndim != 2 or ft_data.shape[1] < 4:
            raise ValueError(f'Unexpected FT shape for {sample_name}: {ft_data.shape}')

        num_time_bins = int(ft_data.shape[0])
        targets = torch.tensor(ft_data[:, self.target_indices].T, dtype=self.dtype)

        events = np.load(tactile_path, allow_pickle=True)
        spikes = self._events_to_spike_tensor(events, num_time_bins)

        return {
            'spikes': spikes.reshape(-1, num_time_bins),
            'target': targets,
            'timestamps': torch.tensor(ft_data[:, 0], dtype=self.dtype),
            'length': num_time_bins,
            'sample_name': sample_name,
        }

    def _load_sample_lengths(self) -> List[int]:
        cache_name = 'sample_lengths_uncropped.csv' if self.use_uncropped else 'sample_lengths.csv'
        cache_path = self.processed_data_path / cache_name

        if cache_path.exists():
            try:
                cache_df = pd.read_csv(cache_path)
                if {'sample_name', 'length'}.issubset(cache_df.columns):
                    cache_map = dict(zip(cache_df['sample_name'], cache_df['length']))
                    if all(sample_name in cache_map for sample_name in self.sample_names):
                        return [int(cache_map[sample_name]) for sample_name in self.sample_names]
            except Exception:
                pass

        lengths = []
        for sample_name in self.sample_names:
            ft_path = self.processed_data_path / f'{sample_name}_ft.npy'
            ft_data = np.load(ft_path, mmap_mode='r')
            if ft_data.ndim != 2 or ft_data.shape[1] < 4:
                raise ValueError(f'Unexpected FT shape for {sample_name}: {ft_data.shape}')
            lengths.append(int(ft_data.shape[0]))

        try:
            cache_df = pd.DataFrame({'sample_name': self.sample_names, 'length': lengths})
            cache_df.to_csv(cache_path, index=False)
        except Exception:
            pass

        return lengths

    def _events_to_spike_tensor(self, events: np.ndarray, num_ts: int) -> torch.Tensor:
        spikes = torch.zeros(self.input_shape + (num_ts,), dtype=self.spike_dtype)

        if events.ndim != 2 or events.shape[0] == 0:
            return spikes
        if events.shape[1] < 4:
            raise ValueError(f'Unexpected tactile event shape: {events.shape}')

        x_events = events[:, 0].astype(np.int64, copy=False)
        y_events = events[:, 1].astype(np.int64, copy=False)
        p_events = events[:, 2].astype(np.int64, copy=False)
        t_events = events[:, 3].astype(np.float64, copy=False)

        time_bins = np.floor(t_events / float(self.meta.camera_period_us)).astype(np.int64)

        valid_mask = (
            (p_events >= 0)
            & (p_events < self.input_shape[0])
            & (time_bins >= 0)
            & (time_bins < num_ts)
            & (x_events >= self.x_min)
            & (x_events <= self.x_max)
            & (y_events >= self.y_min)
            & (y_events <= self.y_max)
        )

        dx = x_events.astype(np.int64, copy=False) - int(self.meta.spatial_crop_x)
        dy = y_events.astype(np.int64, copy=False) - int(self.meta.spatial_crop_y)
        valid_mask &= (dx * dx + dy * dy) <= int(self.meta.spatial_crop_radius) ** 2

        if not np.any(valid_mask):
            return spikes

        x_valid = x_events[valid_mask] - self.x_min
        y_valid = y_events[valid_mask] - self.y_min
        p_valid = p_events[valid_mask]
        t_valid = time_bins[valid_mask]

        spikes[p_valid, y_valid, x_valid, t_valid] = 1

        return spikes


    def _load_meta(self) -> NeuroTacMeta:
        meta_path = Path(self.processed_data_path) / 'meta.csv'
        if not meta_path.exists():
            return NeuroTacMeta()

        meta_df = pd.read_csv(meta_path)
        if 'Parameter' not in meta_df.columns or 'Value' not in meta_df.columns:
            return NeuroTacMeta()

        meta_map = dict(zip(meta_df['Parameter'], meta_df['Value']))

        def _get_int(name: str, default: int) -> int:
            try:
                return int(float(meta_map.get(name, default)))
            except (TypeError, ValueError):
                return default

        pooling_enabled = _get_int('pooling_enabled', 0) == 1 and not self.use_uncropped

        if pooling_enabled:
            sensor_width = _get_int('pooled_sensor_width', _get_int('sensor_width', 320))
            sensor_height = _get_int('pooled_sensor_height', _get_int('sensor_height', 320))
            spatial_crop_x = _get_int('pooled_spatial_crop_x', _get_int('spatial_crop_x', 155))
            spatial_crop_y = _get_int('pooled_spatial_crop_y', _get_int('spatial_crop_y', 155))
            spatial_crop_radius = _get_int('pooled_spatial_crop_radius', _get_int('spatial_crop_radius', 75))
            camera_period_us = _get_int('pooled_camera_period_us', _get_int('camera_period_us', 100))
        else:
            sensor_width = _get_int('sensor_width', 320)
            sensor_height = _get_int('sensor_height', 320)
            spatial_crop_x = _get_int('spatial_crop_x', 155)
            spatial_crop_y = _get_int('spatial_crop_y', 155)
            spatial_crop_radius = _get_int('spatial_crop_radius', 75)
            camera_period_us = _get_int('camera_period_us', 100)

        return NeuroTacMeta(
            sensor_width=sensor_width,
            sensor_height=sensor_height,
            spatial_crop_x=spatial_crop_x,
            spatial_crop_y=spatial_crop_y,
            spatial_crop_radius=spatial_crop_radius,
            camera_period_us=camera_period_us
        )


class TemporalClipDataset(Dataset):
    """Dataset wrapper that clips each sample to a maximum temporal duration."""

    def __init__(self, dataset: Dataset, max_time_bins: Optional[int] = None) -> None:
        self.dataset = dataset
        self.max_time_bins = int(max_time_bins) if max_time_bins is not None else None

        if isinstance(dataset, Subset):
            parent_lengths = getattr(dataset.dataset, 'sample_lengths', None)
            if parent_lengths is None:
                raise AttributeError('Subset parent dataset must expose sample_lengths for temporal clipping.')
            base_lengths = [int(parent_lengths[int(idx)]) for idx in dataset.indices]
        else:
            base_lengths = getattr(dataset, 'sample_lengths', None)
            if base_lengths is None:
                raise AttributeError('Wrapped dataset must expose sample_lengths for temporal clipping.')

        if self.max_time_bins is None:
            self.sample_lengths = [int(length) for length in base_lengths]
        else:
            if self.max_time_bins <= 0:
                raise ValueError('max_time_bins must be positive when provided.')
            self.sample_lengths = [min(int(length), self.max_time_bins) for length in base_lengths]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> SampleItem:
        item = cast(SampleItem, self.dataset[index])

        if self.max_time_bins is None:
            return item

        clipped_length = min(int(item['length']), self.max_time_bins)
        if clipped_length == int(item['length']):
            return item

        clipped_item = dict(item)
        clipped_item['spikes'] = cast(torch.Tensor, item['spikes'])[..., :clipped_length]
        clipped_item['target'] = cast(torch.Tensor, item['target'])[..., :clipped_length]
        clipped_item['timestamps'] = cast(torch.Tensor, item['timestamps'])[:clipped_length]
        clipped_item['length'] = clipped_length
        return clipped_item


class BucketBatchSampler(BatchSampler):
    """Batch sampler that groups similar sequence lengths together.

    Supports single-process loading and rank-local sharding for DDP.
    """

    def __init__(
        self,
        indices: Sequence[int],
        lengths: Sequence[int],
        batch_size: int,
        shuffle: bool,
        drop_last: bool = False,
        bucket_size_multiplier: int = 50,
        seed: int = 42,
        num_replicas: int = 1,
        rank: int = 0,
        sort_by_length: bool = False,
    ) -> None:
        self.indices = list(indices)
        self.lengths = lengths
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.bucket_size_multiplier = max(1, int(bucket_size_multiplier))
        self.seed = int(seed)
        self.num_replicas = max(1, int(num_replicas))
        self.rank = int(rank)
        self.sort_by_length = bool(sort_by_length)
        self.epoch = 0

        if self.batch_size <= 0:
            raise ValueError('batch_size must be positive')
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError('rank must be in [0, num_replicas)')
        if self.sort_by_length and self.shuffle:
            raise ValueError('sort_by_length=True requires shuffle=False')

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _ordered_indices(self) -> List[int]:
        if not self.indices:
            return []

        if self.sort_by_length:
            return sorted(self.indices, key=lambda idx: (self.lengths[idx], idx))

        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            permutation = torch.randperm(len(self.indices), generator=generator).tolist()
            ordered = [self.indices[idx] for idx in permutation]
        else:
            ordered = list(self.indices)

        return ordered

    def __iter__(self):
        ordered = self._ordered_indices()
        if not ordered:
            return

        global_batch_size = self.batch_size * self.num_replicas
        if self.drop_last:
            usable_size = (len(ordered) // global_batch_size) * global_batch_size
            ordered = ordered[:usable_size]
        else:
            total_size = int(np.ceil(len(ordered) / global_batch_size)) * global_batch_size
            if len(ordered) < total_size:
                ordered.extend(ordered[:total_size - len(ordered)])

        bucket_size = global_batch_size * self.bucket_size_multiplier
        global_batches: List[List[int]] = []

        for start in range(0, len(ordered), bucket_size):
            bucket = ordered[start:start + bucket_size]
            bucket.sort(key=lambda idx: self.lengths[idx])

            for batch_start in range(0, len(bucket), global_batch_size):
                batch = bucket[batch_start:batch_start + global_batch_size]
                if len(batch) == global_batch_size or not self.drop_last:
                    global_batches.append(batch)

        if self.shuffle and global_batches:
            generator = torch.Generator().manual_seed(self.seed + self.epoch + 10_000)
            permutation = torch.randperm(len(global_batches), generator=generator).tolist()
            global_batches = [global_batches[idx] for idx in permutation]

        for global_batch in global_batches:
            rank_start = self.rank * self.batch_size
            rank_end = rank_start + self.batch_size
            local_batch = global_batch[rank_start:rank_end]
            if len(local_batch) == self.batch_size or not self.drop_last:
                yield local_batch

    def __len__(self) -> int:
        ordered_count = len(self._ordered_indices())
        global_batch_size = self.batch_size * self.num_replicas
        if self.drop_last:
            return ordered_count // global_batch_size
        return (ordered_count + global_batch_size - 1) // global_batch_size


def _resolve_sampler_space(split_dataset: Dataset, full_dataset: NeuroTacForceDataset) -> Tuple[List[int], List[int]]:
    if hasattr(split_dataset, 'sample_lengths'):
        local_lengths = [int(length) for length in getattr(split_dataset, 'sample_lengths')]
        local_indices = list(range(len(local_lengths)))
        return local_indices, local_lengths

    if isinstance(split_dataset, Subset):
        subset_indices = [int(idx) for idx in split_dataset.indices]
        local_indices = list(range(len(subset_indices)))
        local_lengths = [int(full_dataset.sample_lengths[idx]) for idx in subset_indices]
        return local_indices, local_lengths

    if isinstance(split_dataset, NeuroTacForceDataset):
        local_indices = list(range(len(split_dataset)))
        local_lengths = [int(length) for length in full_dataset.sample_lengths]
        return local_indices, local_lengths

    raise TypeError(f'Unsupported dataset type for bucket batching: {type(split_dataset)!r}')


def _build_batch_sampler(
    split_dataset: Dataset,
    full_dataset: NeuroTacForceDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_replicas: int,
    rank: int,
    sort_by_length: bool = False,
) -> BucketBatchSampler:
    sampler_indices, sampler_lengths = _resolve_sampler_space(split_dataset, full_dataset)
    return BucketBatchSampler(
        indices=sampler_indices,
        lengths=sampler_lengths,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        bucket_size_multiplier=20,
        seed=seed,
        num_replicas=num_replicas,
        rank=rank,
        sort_by_length=sort_by_length,
    )


def neurotac_pad_collate(batch: Sequence[SampleItem]) -> Dict[str, Union[torch.Tensor, List[str]]]:
    max_length = max(int(item['length']) for item in batch)

    spike_batch = []
    target_batch = []
    timestamp_batch = []
    lengths = []
    sample_names = []

    for item in batch:
        spikes = cast(torch.Tensor, item['spikes'])
        target = cast(torch.Tensor, item['target'])
        timestamps = cast(torch.Tensor, item['timestamps'])
        length = int(item['length'])
        pad_amount = max_length - length

        if pad_amount > 0:
            spikes = F.pad(spikes, (0, pad_amount))
            target = F.pad(target, (0, pad_amount))
            last_timestamp = float(timestamps[-1].item()) if length > 0 else 0.0
            timestamp_padding = torch.full((pad_amount,), last_timestamp, dtype=timestamps.dtype)
            timestamps = torch.cat((timestamps, timestamp_padding), dim=0)

        spike_batch.append(spikes)
        target_batch.append(target)
        timestamp_batch.append(timestamps)
        lengths.append(length)
        sample_names.append(str(item['sample_name']))

    return {
        'spikes': torch.stack(spike_batch, dim=0),
        'target': torch.stack(target_batch, dim=0),
        'timestamps': torch.stack(timestamp_batch, dim=0),
        'lengths': torch.tensor(lengths, dtype=torch.long),
        'sample_names': sample_names,
    }


def create_lava_dataloaders(
    processed_data_path: Union[str, Path],
    batch_size: int = 8,
    val_split: float = 0.2,
    test_split: float = 0.0,
    shuffle_train: bool = True,
    seed: int = 42,
    num_workers: int = 0,
    target_mode: str = 'force_z',
    use_uncropped: bool = False,
    binary_spikes: bool = True,
    pin_memory: bool = False,
    spike_dtype: torch.dtype = torch.uint8,
    distributed_world_size: int = 1,
    distributed_rank: int = 0,
    train_max_duration_ms: Optional[float] = None,
    val_max_duration_ms: Optional[float] = None,
    test_max_duration_ms: Optional[float] = None,
    val_batch_size: Optional[int] = None,
    test_batch_size: Optional[int] = None,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], NeuroTacForceDataset]:
    """Create Lava-dl-ready dataloaders from processed NeuroTac data.

    Returns train/val/test dataloaders along with the full dataset object. Batched spike
    tensors are shaped `[batch, channels * height * width, time]`.
    """

    dataset = NeuroTacForceDataset(
        processed_data_path=processed_data_path,
        target_mode=target_mode,
        use_uncropped=use_uncropped,
        binary_spikes=binary_spikes,
        spike_dtype=spike_dtype,
    )

    train_max_time_bins = _duration_ms_to_num_bins(train_max_duration_ms, dataset.meta.camera_period_us)
    val_max_time_bins = _duration_ms_to_num_bins(val_max_duration_ms, dataset.meta.camera_period_us)
    test_max_time_bins = _duration_ms_to_num_bins(test_max_duration_ms, dataset.meta.camera_period_us)

    dataset_size = len(dataset)
    test_size = int(dataset_size * test_split)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size - test_size

    if train_size <= 0:
        raise ValueError('Train split is empty. Reduce val_split/test_split or add more samples.')

    generator = torch.Generator().manual_seed(seed)

    if val_size == 0 and test_size == 0:
        train_dataset = dataset
        val_dataset = None
        test_dataset = None
    else:
        splits = [train_size]
        if val_size > 0:
            splits.append(val_size)
        if test_size > 0:
            splits.append(test_size)

        split_datasets = random_split(dataset, splits, generator=generator)
        split_index = 0
        train_dataset = split_datasets[split_index]
        split_index += 1
        val_dataset = split_datasets[split_index] if val_size > 0 else None
        split_index += 1 if val_size > 0 else 0
        test_dataset = split_datasets[split_index] if test_size > 0 else None

    train_dataset = TemporalClipDataset(train_dataset, max_time_bins=train_max_time_bins)
    if val_dataset is not None:
        val_dataset = TemporalClipDataset(val_dataset, max_time_bins=val_max_time_bins)
    if test_dataset is not None:
        test_dataset = TemporalClipDataset(test_dataset, max_time_bins=test_max_time_bins)

    effective_val_batch_size = int(val_batch_size) if val_batch_size is not None else int(batch_size)
    effective_test_batch_size = int(test_batch_size) if test_batch_size is not None else int(batch_size)
    if effective_val_batch_size <= 0:
        raise ValueError('val_batch_size must be positive when provided.')
    if effective_test_batch_size <= 0:
        raise ValueError('test_batch_size must be positive when provided.')

    train_batch_sampler = _build_batch_sampler(
        train_dataset,
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        seed=seed,
        num_replicas=distributed_world_size,
        rank=distributed_rank,
    )
    train_loader_kwargs = {
        'batch_sampler': train_batch_sampler,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'persistent_workers': num_workers > 0,
        'collate_fn': neurotac_pad_collate,
    }
    if num_workers > 0:
        train_loader_kwargs['prefetch_factor'] = 2

    train_loader = DataLoader(
        train_dataset,
        **train_loader_kwargs,
    )

    val_loader = None
    if val_dataset is not None:
        val_batch_sampler = _build_batch_sampler(
            val_dataset,
            dataset,
            batch_size=effective_val_batch_size,
            shuffle=False,
            seed=seed,
            num_replicas=distributed_world_size,
            rank=distributed_rank,
            sort_by_length=True,
        )
        val_loader_kwargs = {
            'batch_sampler': val_batch_sampler,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'persistent_workers': num_workers > 0,
            'collate_fn': neurotac_pad_collate,
        }
        if num_workers > 0:
            val_loader_kwargs['prefetch_factor'] = 2

        val_loader = DataLoader(
            val_dataset,
            **val_loader_kwargs,
        )

    test_loader = None
    if test_dataset is not None:
        test_batch_sampler = _build_batch_sampler(
            test_dataset,
            dataset,
            batch_size=effective_test_batch_size,
            shuffle=False,
            seed=seed,
            num_replicas=distributed_world_size,
            rank=distributed_rank,
        )
        test_loader_kwargs = {
            'batch_sampler': test_batch_sampler,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'persistent_workers': num_workers > 0,
            'collate_fn': neurotac_pad_collate,
        }
        if num_workers > 0:
            test_loader_kwargs['prefetch_factor'] = 2

        test_loader = DataLoader(
            test_dataset,
            **test_loader_kwargs,
        )

    return train_loader, val_loader, test_loader, dataset


def main() -> None:
    processed_data_path = '/media/george/T7 Shield/Neuromorphic Data/George/sigma_delta/processed_data_aligned'
    train_loader, val_loader, test_loader, dataset = create_lava_dataloaders(
        processed_data_path=processed_data_path,
        batch_size=5,
        val_split=0.2,
        test_split=0.1,
        target_mode='force_z',
    )

    first_batch = next(iter(train_loader))
    print(f'Dataset size: {len(dataset)}')
    print(f'Input shape (C,H,W): {dataset.input_shape}')
    print(f'Spike dtype: {first_batch["spikes"].dtype}')
    print(f"Spike batch shape: {tuple(first_batch['spikes'].shape)}")
    print(f"Target batch shape: {tuple(first_batch['target'].shape)}")
    print(f"Sequence lengths: {first_batch['lengths'][:4].tolist()}")
    print(f"Validation loader present: {val_loader is not None}")
    print(f"Test loader present: {test_loader is not None}")


if __name__ == '__main__':
    main()
