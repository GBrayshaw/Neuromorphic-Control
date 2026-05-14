from __future__ import annotations

import csv
import os
import random
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import lava.lib.dl.slayer as slayer
import lava.lib.dl.slayer.axon.delta as slayer_delta
import lava.lib.dl.slayer.utils.recurrent as slayer_recurrent
import lava.lib.dl.slayer.block.base as slayer_block_base
from training.single_tap.dataloader import create_lava_dataloaders

try:
    from torch.amp import GradScaler, autocast
    _USE_NEW_AMP_API = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    _USE_NEW_AMP_API = False


def _patch_lava_delta_bool_subtraction() -> None:
    """Patch Lava's cumulative-error delta path for modern PyTorch bool semantics."""
    if getattr(slayer_delta._DeltaUnit, '_bool_subtraction_patch_applied', False):
        return

    def _patched_forward(
        ctx,
        input,
        threshold,
        pre_state,
        residual_state,
        error_state,
        cum_error,
        tau_grad,
        scale_grad,
    ):
        output = torch.zeros_like(input)
        delta_input = torch.zeros_like(input)
        error = error_state

        if cum_error is True:
            for t in range(input.shape[-1]):
                delta = input[..., t] - pre_state + residual_state
                delta_input[..., t] = delta
                error += delta
                output[..., t] = torch.where(
                    torch.abs(error) >= threshold,
                    delta,
                    0 * delta,
                ).to(input.dtype)
                spike_mask = torch.abs(output[..., t]) > 0
                error *= torch.logical_not(spike_mask).to(error.dtype)
                residual_state = (delta - output[..., t]).detach()
                pre_state = input[..., t].detach()
        else:
            for t in range(input.shape[-1]):
                delta = input[..., t] - pre_state + residual_state
                delta_input[..., t] = delta
                output[..., t] = torch.where(
                    torch.abs(delta) >= threshold,
                    delta,
                    0 * delta,
                ).to(input.dtype)
                residual_state = (delta - output[..., t]).detach()
                pre_state = input[..., t].detach()

        ctx.save_for_backward(
            delta_input,
            threshold,
            torch.autograd.Variable(
                torch.tensor(
                    tau_grad,
                    device=input.device,
                    dtype=input.dtype,
                ),
                requires_grad=False,
            ),
            torch.autograd.Variable(
                torch.tensor(
                    scale_grad,
                    device=input.device,
                    dtype=input.dtype,
                ),
                requires_grad=False,
            ),
        )

        return output, residual_state, error

    slayer_delta._DeltaUnit.forward = staticmethod(_patched_forward)
    slayer_delta._DeltaUnit._bool_subtraction_patch_applied = True


_patch_lava_delta_bool_subtraction()


def _patch_lava_recurrent_backward() -> None:
    """Patch Lava recurrent backward to retain graph across timestep backprop calls."""
    if getattr(slayer_recurrent.CustomRecurrent, '_retain_graph_patch_applied', False):
        return

    def _patched_backward(ctx, grad_x):
        grad_z = torch.zeros_like(grad_x).to(grad_x.device)
        grad_neuron = None
        grad_spike = 0

        for time in range(grad_x.shape[-1])[::-1]:
            grad_spike = grad_spike + grad_x[..., time: time + 1]
            torch.autograd.backward(
                ctx.spikes[time],
                grad_spike,
                retain_graph=(time > 0),
            )
            grad_dend_sum = ctx.dend_sums[time].grad
            grad_feedback = grad_dend_sum
            grad_dendrite = grad_dend_sum

            grad_spike = torch.unsqueeze(
                torch.matmul(grad_feedback, ctx.recurrent_mat), dim=-1
            )
            grad_z[..., time] = grad_dendrite

        grad_output = (
            grad_z[..., 1:].transpose(0, 1).reshape(grad_dendrite.shape[1], -1)
        )
        input = ctx.x[..., :-1].transpose(1, 2).reshape(-1, ctx.x.shape[1])
        grad_recurrent_mat = torch.matmul(grad_output, input)

        return grad_z, grad_neuron, grad_recurrent_mat

    slayer_recurrent.CustomRecurrent.backward = staticmethod(_patched_backward)
    slayer_recurrent.CustomRecurrent._retain_graph_patch_applied = True


_patch_lava_recurrent_backward()


def _patch_lava_recurrent_forward() -> None:
    """Use Lava's ground-truth recurrent implementation to avoid custom autograd failures."""
    if getattr(slayer_recurrent, '_custom_recurrent_forward_patch_applied', False):
        return

    slayer_recurrent.custom_recurrent = slayer_recurrent.custom_recurrent_ground_truth_2
    setattr(slayer_recurrent, '_custom_recurrent_forward_patch_applied', True)


_patch_lava_recurrent_forward()


def _patch_lava_step_delay() -> None:
    """Detach persistent delay buffers so state does not retain previous-batch graphs."""
    if getattr(slayer_block_base, '_step_delay_patch_applied', False):
        return

    original_delay = slayer_block_base.delay

    def _patched_step_delay(module, x):
        if hasattr(module, 'delay_buffer') is False:
            module.delay_buffer = None
        persistent_state = hasattr(module, 'neuron') and module.neuron.persistent_state is True
        if module.delay_buffer is not None and module.delay_buffer.shape[0] != x.shape[0]:
            module.delay_buffer = None
        if persistent_state:
            delay_buffer = 0 if module.delay_buffer is None else module.delay_buffer
            module.delay_buffer = x[..., -1].detach().clone()
        x = original_delay(x, 1)
        if persistent_state:
            if isinstance(delay_buffer, torch.Tensor):
                x[..., 0] = delay_buffer.detach()
            else:
                x[..., 0] = delay_buffer
        return x

    slayer_block_base.step_delay = _patched_step_delay
    setattr(slayer_block_base, '_step_delay_patch_applied', True)


_patch_lava_step_delay()


def _setup_distributed():
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    distributed = world_size > 1

    if distributed:
        os.environ.setdefault('NCCL_P2P_DISABLE', '1')
        os.environ.setdefault('NCCL_IB_DISABLE', '1')
        os.environ.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1')
        os.environ.setdefault('NCCL_DEBUG', 'WARN')

    if torch.cuda.is_available():
        if distributed:
            torch.cuda.set_device(local_rank)
            device = torch.device('cuda', local_rank)
        else:
            torch.cuda.set_device(0)
            device = torch.device('cuda', 0)
    else:
        device = torch.device('cpu')

    if distributed and not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(seconds=180))

    return distributed, rank, world_size, local_rank, device


def _cleanup_distributed(distributed: bool):
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def _is_main_process(rank: int) -> bool:
    return rank == 0


def _rank0_print(is_main_process: bool, *args, **kwargs):
    if is_main_process:
        print(*args, **kwargs)


def _rank_print(rank: int, *args, **kwargs):
    print(f'[rank{rank}]', *args, **kwargs, flush=True)


def _debug_model_state(net: torch.nn.Module, rank: int, local_rank: int, device: torch.device):
    named_params = list(net.named_parameters())
    total_params = len(named_params)
    trainable_params = sum(1 for _, param in named_params if param.requires_grad)
    total_numel = sum(int(param.numel()) for _, param in named_params)
    state_keys = list(net.state_dict().keys())
    sample_param_names = [name for name, _ in named_params[:8]]
    sample_state_keys = state_keys[:8]

    _rank_print(
        rank,
        f'Pre-DDP model debug | local_rank={local_rank} | device={device} | '
        f'param_tensors={total_params} | trainable_tensors={trainable_params} | '
        f'total_numel={total_numel} | state_dict_keys={len(state_keys)}'
    )
    _rank_print(rank, f'Pre-DDP sample param names: {sample_param_names}')
    _rank_print(rank, f'Pre-DDP sample state_dict keys: {sample_state_keys}')


def _debug_distributed_cuda_collective(rank: int, device: torch.device, distributed: bool):
    if not distributed:
        return

    tensor = torch.tensor([rank + 1.0], dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    _rank_print(rank, f'Pre-DDP NCCL all_reduce ok | reduced_value={tensor.item():.1f} on {device}')


def _cuda_mem_mb(device: torch.device) -> Tuple[float, float]:
    if device.type != 'cuda' or not torch.cuda.is_available():
        return 0.0, 0.0
    allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
    return allocated_mb, reserved_mb


def _debug_rank_batch_state(
    rank: int,
    local_rank: int,
    device: torch.device,
    phase_name: str,
    batch_index: int,
    spikes: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    loss: Optional[torch.Tensor] = None,
) -> None:
    allocated_mb, reserved_mb = _cuda_mem_mb(device)
    pieces = [
        f'{phase_name} batch {batch_index}',
        f'pid={os.getpid()}',
        f'local_rank={local_rank}',
        f'current_device={torch.cuda.current_device() if device.type == "cuda" and torch.cuda.is_available() else "cpu"}',
        f'device={device}',
        f'spikes_device={spikes.device}',
        f'target_device={target.device}',
        f'lengths_device={lengths.device}',
        f'spikes_shape={tuple(spikes.shape)}',
        f'target_shape={tuple(target.shape)}',
        f'lengths_shape={tuple(lengths.shape)}',
        f'cuda_allocated_mb={allocated_mb:.1f}',
        f'cuda_reserved_mb={reserved_mb:.1f}',
    ]
    if output is not None:
        pieces.extend([
            f'output_device={output.device}',
            f'output_shape={tuple(output.shape)}',
        ])
    if loss is not None:
        pieces.extend([
            f'loss_device={loss.device}',
            f'loss_value={float(loss.detach().item()):.6f}',
        ])
    _rank_print(rank, ' | '.join(pieces))


def _reduce_average(value: float, device: torch.device, distributed: bool) -> float:
    if not distributed:
        return float(value)

    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return float(tensor.item())


def _set_loader_epoch(loader, epoch: int):
    if loader is None:
        return
    batch_sampler = getattr(loader, 'batch_sampler', None)
    if batch_sampler is not None and hasattr(batch_sampler, 'set_epoch'):
        batch_sampler.set_epoch(epoch)


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f'{hours:d}h{minutes:02d}m{secs:02d}s'
    if minutes > 0:
        return f'{minutes:d}m{secs:02d}s'
    return f'{secs:d}s'


def _dataset_length_summary(split_dataset) -> Tuple[Optional[float], Optional[int]]:
    if split_dataset is None:
        return None, None

    sample_lengths = getattr(split_dataset, 'sample_lengths', None)
    if not sample_lengths:
        return None, None

    lengths_array = np.asarray(sample_lengths, dtype=np.float64)
    return float(lengths_array.mean()), int(lengths_array.max())


def _assert_batch_lengths_within_limit(
    lengths: torch.Tensor,
    expected_max_seq_len: Optional[int],
    phase_name: str,
    batch_index: int,
) -> None:
    if expected_max_seq_len is None:
        return

    batch_max_seq_len = int(lengths.max().item())
    if batch_max_seq_len > expected_max_seq_len:
        raise RuntimeError(
            f'{phase_name} batch {batch_index} exceeded expected max_seq_len: '
            f'{batch_max_seq_len} > {expected_max_seq_len}. '
            'This indicates the active dataloader is not applying the intended temporal clipping.'
        )


def _build_curriculum_schedule(
    initial_train_max_duration_ms: Optional[float],
    curriculum_train_increment_ms: Optional[float],
    curriculum_train_final_max_duration_ms: Optional[float],
) -> List[Optional[float]]:
    if initial_train_max_duration_ms is None:
        return [None]

    initial_value = float(initial_train_max_duration_ms)
    if initial_value <= 0:
        raise ValueError('initial_train_max_duration_ms must be positive when provided.')

    if curriculum_train_final_max_duration_ms is None:
        return [initial_value]

    final_value = float(curriculum_train_final_max_duration_ms)
    if final_value < initial_value:
        raise ValueError('curriculum_train_final_max_duration_ms must be >= initial_train_max_duration_ms.')
    if np.isclose(final_value, initial_value):
        return [initial_value]

    if curriculum_train_increment_ms is None or curriculum_train_increment_ms <= 0:
        raise ValueError('curriculum_train_increment_ms must be positive when the final curriculum duration exceeds the initial duration.')

    increment_value = float(curriculum_train_increment_ms)
    schedule: List[float] = [initial_value]
    current_value = initial_value

    while current_value + increment_value < final_value:
        current_value += increment_value
        schedule.append(float(current_value))

    if not np.isclose(float(schedule[-1]), final_value):
        schedule.append(final_value)

    return [float(value) for value in schedule]


def _curriculum_batch_size_for_stage(
    initial_batch_size: int,
    initial_train_max_duration_ms: Optional[float],
    current_train_max_duration_ms: Optional[float],
    min_batch_size: int = 2,
    ) -> int:

    if initial_batch_size <= 0:
        raise ValueError('initial_batch_size must be positive.')
    if min_batch_size <= 0:
        raise ValueError('min_batch_size must be positive.')

    if initial_train_max_duration_ms is None or current_train_max_duration_ms is None:
        return initial_batch_size

    initial_duration_value = float(initial_train_max_duration_ms)
    current_duration_value = float(current_train_max_duration_ms)
    if initial_duration_value <= 0:
        raise ValueError('initial_train_max_duration_ms must be positive when provided.')
    if current_duration_value <= 0:
        raise ValueError('current_train_max_duration_ms must be positive when provided.')

    proportional_batch_size = int(np.floor(initial_batch_size * (initial_duration_value / current_duration_value)))
    return max(min_batch_size, min(initial_batch_size, proportional_batch_size))


def _curriculum_duration_for_stage(
    initial_duration_ms: Optional[float],
    curriculum_schedule_ms: Sequence[Optional[float]],
    curriculum_stage_index: int,
) -> Optional[float]:
    if initial_duration_ms is None:
        return None

    if not curriculum_schedule_ms:
        return float(initial_duration_ms)

    schedule_start_ms = curriculum_schedule_ms[0]
    schedule_current_ms = curriculum_schedule_ms[curriculum_stage_index]
    if schedule_start_ms is None or schedule_current_ms is None:
        return None

    duration_delta_ms = float(schedule_current_ms) - float(schedule_start_ms)
    return float(initial_duration_ms) + duration_delta_ms


def _save_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    base_net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    best_val_loss: float,
    run_seed: int,
    history: List[Dict],
    current_train_max_duration_ms: Optional[float],
    current_val_max_duration_ms: Optional[float],
    current_test_max_duration_ms: Optional[float],
    curriculum_schedule_ms: Sequence[Optional[float]],
    curriculum_stage_index: int,
    initial_batch_size: int,
    current_batch_size: int,
    curriculum_stage_start_epoch: int,
    curriculum_stage_best_loss: float,
    curriculum_epochs_since_improvement: int,
) -> None:
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': base_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_loss': best_val_loss,
            'run_seed': run_seed,
            'history': history,
            'current_train_max_duration_ms': current_train_max_duration_ms,
            'current_val_max_duration_ms': current_val_max_duration_ms,
            'current_test_max_duration_ms': current_test_max_duration_ms,
            'curriculum_schedule_ms': curriculum_schedule_ms,
            'curriculum_stage_index': curriculum_stage_index,
            'initial_batch_size': initial_batch_size,
            'current_batch_size': current_batch_size,
            'curriculum_stage_start_epoch': curriculum_stage_start_epoch,
            'curriculum_stage_best_loss': curriculum_stage_best_loss,
            'curriculum_epochs_since_improvement': curriculum_epochs_since_improvement,
        },
        checkpoint_path,
    )


def _find_latest_checkpoint(logs_folder: Path) -> Optional[Path]:
    checkpoint_paths = sorted(logs_folder.glob('checkpoint_*.pt'))
    if not checkpoint_paths:
        return None
    return checkpoint_paths[-1]


def _create_run_seed(distributed: bool, rank: int, device: torch.device) -> int:
    max_seed = 2**31 - 1
    seed = int(time.time_ns() % max_seed)

    if not distributed:
        return seed

    if rank != 0:
        seed = 0

    seed_tensor = torch.tensor([seed], dtype=torch.int64, device=device)
    dist.broadcast(seed_tensor, src=0)
    return int(seed_tensor.item())


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_loss_plot(history: List[Dict], output_path: Path) -> None:
    if not history:
        return

    epochs_axis = [entry['epoch'] + 1 for entry in history]
    train_losses = [entry['train_loss'] for entry in history]
    val_losses = [entry['val_loss'] for entry in history]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_axis, train_losses, label='Train Loss', linewidth=2)

    if any(loss is not None for loss in val_losses):
        plt.plot(epochs_axis, val_losses, label='Validation Loss', linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def _prepare_masked_sequence_tensors(
    output: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    while output.dim() > target.dim() and 1 in output.shape:
        squeeze_dim = next(idx for idx, size in enumerate(output.shape) if size == 1)
        output = output.squeeze(squeeze_dim)
    if output.dim() == 2:
        output = output.unsqueeze(1)
    if target.dim() == 2:
        target = target.unsqueeze(1)

    if output.shape != target.shape:
        raise ValueError(f'Output shape {tuple(output.shape)} does not match target shape {tuple(target.shape)}')

    time_axis = output.shape[-1]
    mask = (torch.arange(time_axis, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(1)
    return output, target, mask.to(dtype=output.dtype)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def masked_force_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
    spikes: torch.Tensor,
    point_loss_weight: float = 1.0,
    derivative_loss_weight: float = 0.3,
    hold_loss_weight: float = 0.05,
    hold_fit_loss_weight: float = 0.2,
    point_huber_delta: float = 0.1,
    derivative_huber_delta: float = 0.05,
    hold_activity_alpha: float = 4.0,
) -> torch.Tensor:
    """Masked composite loss for force prediction on variable-length spike sequences.

    Combines:
    - pointwise Huber fit to the target force,
    - Huber loss on temporal derivatives,
    - activity-weighted hold regularization that discourages drift when input activity is sparse,
    - activity-weighted hold fit that anchors predictions to the target during sparse activity.

    Padded timesteps are excluded from all terms.
    """
    output, target, mask = _prepare_masked_sequence_tensors(output, target, lengths)

    pointwise_loss = F.huber_loss(output, target, reduction='none', delta=point_huber_delta)
    pointwise_loss = _masked_mean(pointwise_loss, mask)

    pair_mask = mask[..., 1:] * mask[..., :-1]
    pred_delta = output[..., 1:] - output[..., :-1]
    target_delta = target[..., 1:] - target[..., :-1]
    derivative_loss = F.huber_loss(
        pred_delta,
        target_delta,
        reduction='none',
        delta=derivative_huber_delta,
    )
    derivative_loss = _masked_mean(derivative_loss, pair_mask)

    if spikes.dim() < 3:
        raise ValueError(f'Spike tensor must include feature and time dimensions, got shape {tuple(spikes.shape)}')

    activity_dims = tuple(range(1, spikes.dim() - 1))
    event_activity = spikes.abs().sum(dim=activity_dims)
    valid_time_mask = mask.squeeze(1)
    mean_activity = (event_activity * valid_time_mask).sum(dim=-1, keepdim=True) / valid_time_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    normalized_activity = event_activity / mean_activity.clamp_min(1e-6)
    sparse_activity_weight = torch.exp(-hold_activity_alpha * normalized_activity).unsqueeze(1)
    hold_fit_weight = sparse_activity_weight * mask
    hold_fit_loss = F.huber_loss(output, target, reduction='none', delta=point_huber_delta)
    hold_fit_loss = (hold_fit_loss * hold_fit_weight).sum() / hold_fit_weight.sum().clamp_min(1.0)

    hold_weight = 0.5 * (sparse_activity_weight[..., 1:] + sparse_activity_weight[..., :-1])
    hold_weight = hold_weight * pair_mask
    hold_stability_loss = (pred_delta.abs() * hold_weight).sum() / hold_weight.sum().clamp_min(1.0)

    return (
        point_loss_weight * pointwise_loss
        + derivative_loss_weight * derivative_loss
        + hold_loss_weight * hold_stability_loss
        + hold_fit_loss_weight * hold_fit_loss
    )

########################################################################################################################
# Network
########################################################################################################################
class Network(torch.nn.Module):
    def __init__(self, in_shape):
        super(Network, self).__init__()
        
        cuba_params = { # cuba neuron parameters
            "threshold": 1.05,  # Previously 1.25
            "current_decay": 0.15,  # Preivously 0.25
            "voltage_decay": 0.02,  # Previously 0.03
            "tau_grad": 0.03,
            "scale_grad": 3,
            "requires_grad": False,
        }
        cuba_dense_params = {
            **cuba_params,
            'dropout': slayer.neuron.Dropout(p=0.2),
        }

        sdnn_params = { # sigma-delta neuron parameters
                'threshold'     : 0.18,    # delta unit threshold
                'tau_grad'      : 0.5,    # delta unit surrogate gradient relaxation parameter
                'scale_grad'    : 1,      # delta unit surrogate gradient scale parameter
                'requires_grad' : False,   # trainable threshold
                'cum_error'     : True,   # cumulative error feedback
                'shared_param'  : True,   # layer wise threshold
                'activation'    : F.relu, # activation function
        }
        sdnn_cnn_params = { # conv layer has additional mean only batch norm
                **sdnn_params,                                 # copy all sdnn_params
                'norm' : slayer.neuron.norm.MeanOnlyBatchNorm, # mean only quantized batch normalizaton
        }
        sdnn_dense_params = { # dense layers have additional dropout units enabled
                **sdnn_cnn_params,                        
                'dropout' : slayer.neuron.Dropout(p=0.25), # neuron dropout
        }

        # Feature extraction layers
        self.blocks = torch.nn.ModuleList([
            slayer.block.cuba.Conv(cuba_params, 2, 16, 3, stride=2, padding=1, weight_scale=3),
            slayer.block.cuba.Conv(cuba_params, 16, 32, 3, stride=2, padding=1, weight_scale=3),
            slayer.block.cuba.Conv(cuba_params, 32, 64, 3, stride=2, padding=1, weight_scale=3),
            slayer.block.cuba.Flatten(),  # Flatten spatial dimensions before recurrent layers
        ])

        # Calculate flattened size based on input shape and conv layers in self.blocks
        flattened_size, _ = self._calculate_flattened_size(in_shape)
        
        # Network head for force regression
        # TODO: Change recurrancy here to maintain spiking activity for longer scale tasks
        self.head = torch.nn.ModuleList([
            # slayer.block.cuba.Dense(cuba_dense_params, flattened_size, 250, weight_scale=2, weight_norm=False),
            # slayer.block.cuba.Dense(cuba_dense_params, 750, 100, weight_scale=2, weight_norm=False),
            # slayer.block.cuba.Dense(cuba_dense_params, 250, 200, weight_scale=2, weight_norm=True),
            # Recurrent layers for temporal processing with feedback
            slayer.block.cuba.Recurrent(cuba_dense_params, flattened_size, 256, weight_scale=2, weight_norm=False),
            # slayer.block.cuba.Recurrent(cuba_dense_params, 256, 100, weight_scale=2, weight_norm=False),
            # slayer.block.cuba.Recurrent(cuba_dense_params, 100, 100, weight_scale=2, weight_norm=False),
            # slayer.block.cuba.Recurrent(cuba_dense_params, 100, 100, weight_scale=2, weight_norm=False),

            # Sigma-delta encoding and output
            slayer.block.sigma_delta.Input(sdnn_params),
            slayer.block.sigma_delta.Output(sdnn_dense_params,  256,   1, weight_scale=2, weight_norm=False)
        ])

    def forward(self, x):
        for block in self.blocks: 
            # forward computation is as simple as calling the blocks in a loop
            x = block(x)
        for block in self.head: 
            x = block(x)
        return x

    def grad_flow(self, path):
        # helps monitor the gradient flow
        grad = [b.synapse.grad_norm for b in self.blocks if hasattr(b, 'synapse')]

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()

        return grad
    
    def export_hdf5(self, filename):
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{i}'))

    def _calculate_conv_output_size(self, input_size, kernel_size, stride, padding):
        """
        Calculate output size after a convolution operation.
        
        Args:
            input_size: Input spatial dimension (height or width)
            kernel_size: Size of convolution kernel
            stride: Stride of convolution
            padding: Padding applied
            
        Returns:
            Output spatial dimension
        """
        return (input_size + 2 * padding - kernel_size) // stride + 1

    def _calculate_flattened_size(self, input_shape):
        channels, height, width = input_shape
        
        # Extract conv layer parameters from self.blocks
        for block in self.blocks:
            # Check if it's a Conv layer (CUBA or sigma-delta)
            if hasattr(block, 'synapse') and hasattr(block.synapse, 'out_channels'):
                # Extract parameters from the conv layer
                out_channels = block.synapse.out_channels
                kernel_size = block.synapse.kernel_size[0] if isinstance(block.synapse.kernel_size, tuple) else block.synapse.kernel_size
                stride = block.synapse.stride[0] if isinstance(block.synapse.stride, tuple) else block.synapse.stride
                padding = block.synapse.padding[0] if isinstance(block.synapse.padding, tuple) else block.synapse.padding
                
                # Calculate output dimensions
                height = self._calculate_conv_output_size(height, kernel_size, stride, padding)
                width = self._calculate_conv_output_size(width, kernel_size, stride, padding)
                channels = out_channels
            # Stop at Flatten layer since that's where we need the size
            elif 'Flatten' in block.__class__.__name__:
                break
        
        flattened_size = channels * height * width
        return flattened_size, (channels, height, width)


def main():
    distributed, rank, world_size, local_rank, device = _setup_distributed()
    is_main_process = _is_main_process(rank)
    run_seed = _create_run_seed(distributed=distributed, rank=rank, device=device)
    _seed_everything(run_seed)

    ########################################################################################################################
    # Training params
    ########################################################################################################################
    processed_data_path = '/media/george/T7 Shield/Neuromorphic Data/George/sigma_delta/processed_data_multitap_no_release/'
    experiment_root = Path('/home/george/Documents/NeuroTac_Force_Estimation/models/')
    trained_folder = experiment_root / 'trained'
    logs_folder = experiment_root / 'logs'
    trained_folder.mkdir(parents=True, exist_ok=True)
    logs_folder.mkdir(parents=True, exist_ok=True)

    lr = 1e-4 #1e-3
    epochs = 60
    train_batch_size = 4
    val_batch_size = 2
    test_batch_size = 2
    num_workers = 4
    minimum_batch_size = 2
    val_split = 0.1
    test_split = 0.1
    steps = [15, 30, 45]
    weight_decay = 1e-5
    max_grad_norm = 1.0
    target_mode = 'force_z'
    point_loss_weight = 1.0
    derivative_loss_weight = 0.15
    hold_loss_weight = 0.05
    hold_fit_loss_weight = 0.3
    point_huber_delta = 0.1
    derivative_huber_delta = 0.05
    hold_activity_alpha = 6.0
    enable_rank_batch_debug = True
    use_amp = False  # Lava Slayer CUBA kernels expect float32 and currently fail under autocast/float16
    train_max_duration_ms = 800.0
    val_max_duration_ms = None
    test_max_duration_ms = None
    curriculum_enabled = False
    curriculum_train_increment_ms = 250.0
    curriculum_train_final_max_duration_ms = 8000.0
    curriculum_stage_epochs = 20
    curriculum_plateau_patience = 10
    curriculum_plateau_min_delta = 1e-4


    ########################################################################################################################
    # Init dataloaders, model, optimizer, scheduler
    ########################################################################################################################

    curriculum_schedule_ms = _build_curriculum_schedule(
        initial_train_max_duration_ms=train_max_duration_ms,
        curriculum_train_increment_ms=curriculum_train_increment_ms,
        curriculum_train_final_max_duration_ms=curriculum_train_final_max_duration_ms,
    )
    curriculum_stage_index = 0
    if not curriculum_enabled:
        curriculum_schedule_ms = [None]
        current_train_max_duration_ms = None
        current_val_max_duration_ms = None
        current_test_max_duration_ms = None
    else:
        current_train_max_duration_ms = curriculum_schedule_ms[0]
        current_val_max_duration_ms = _curriculum_duration_for_stage(
            initial_duration_ms=val_max_duration_ms,
            curriculum_schedule_ms=curriculum_schedule_ms,
            curriculum_stage_index=curriculum_stage_index,
        )
        current_test_max_duration_ms = _curriculum_duration_for_stage(
            initial_duration_ms=test_max_duration_ms,
            curriculum_schedule_ms=curriculum_schedule_ms,
            curriculum_stage_index=curriculum_stage_index,
        )
    initial_batch_size = train_batch_size
    current_batch_size = _curriculum_batch_size_for_stage(
        initial_batch_size=initial_batch_size,
        initial_train_max_duration_ms=train_max_duration_ms,
        current_train_max_duration_ms=current_train_max_duration_ms,
        min_batch_size=minimum_batch_size,
    )
    curriculum_stage_start_epoch = 0
    curriculum_stage_best_loss = float('inf')
    curriculum_epochs_since_improvement = 0

    def _build_active_dataloaders(
        active_train_max_duration_ms: Optional[float],
        active_val_max_duration_ms: Optional[float],
        active_test_max_duration_ms: Optional[float],
        active_batch_size: int,
    ):
        active_train_loader, active_val_loader, active_test_loader, active_dataset = create_lava_dataloaders(
            processed_data_path=processed_data_path,
            batch_size=active_batch_size,
            val_split=val_split,
            test_split=test_split,
            seed=run_seed,
            num_workers=num_workers,
            target_mode=target_mode,
            binary_spikes=True,
            pin_memory=torch.cuda.is_available(),
            distributed_world_size=world_size,
            distributed_rank=rank,
            train_max_duration_ms=active_train_max_duration_ms,
            val_max_duration_ms=active_val_max_duration_ms,
            test_max_duration_ms=active_test_max_duration_ms,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
        )
        active_train_avg_seq_len, active_train_max_seq_len = _dataset_length_summary(active_train_loader.dataset)
        active_val_avg_seq_len, active_val_max_seq_len = _dataset_length_summary(active_val_loader.dataset if active_val_loader is not None else None)
        active_test_avg_seq_len, active_test_max_seq_len = _dataset_length_summary(active_test_loader.dataset if active_test_loader is not None else None)
        return (
            active_train_loader,
            active_val_loader,
            active_test_loader,
            active_dataset,
            active_train_avg_seq_len,
            active_train_max_seq_len,
            active_val_avg_seq_len,
            active_val_max_seq_len,
            active_test_avg_seq_len,
            active_test_max_seq_len,
        )

    (
        train_loader,
        val_loader,
        test_loader,
        dataset,
        train_avg_seq_len,
        train_max_seq_len,
        val_avg_seq_len,
        val_max_seq_len,
        test_avg_seq_len,
        test_max_seq_len,
    ) = _build_active_dataloaders(
        current_train_max_duration_ms,
        current_val_max_duration_ms,
        current_test_max_duration_ms,
        current_batch_size,
    )

    num_gpus = torch.cuda.device_count()
    _rank0_print(is_main_process, f'Using device: {device}')
    _rank0_print(is_main_process, f'Number of GPUs available: {num_gpus}')
    _rank0_print(is_main_process, f'Dataset size: {len(dataset)}')
    _rank0_print(is_main_process, f'Input shape (C,H,W): {dataset.input_shape}')
    _rank0_print(is_main_process, f'Train batch size (initial): {initial_batch_size}')
    _rank0_print(is_main_process, f'Train batch size (active): {current_batch_size}')
    _rank0_print(is_main_process, f'Validation batch size: {val_batch_size}')
    _rank0_print(is_main_process, f'Test batch size: {test_batch_size}')
    _rank0_print(is_main_process, f'Minimum train batch size: {minimum_batch_size}')
    _rank0_print(is_main_process, 'Batch size schedule: proportional to train max duration ratio')
    _rank0_print(is_main_process, f'DataLoader workers: {num_workers}')
    _rank0_print(is_main_process, f'Max gradient norm: {max_grad_norm}')
    _rank0_print(
        is_main_process,
        f'Loss weights: point={point_loss_weight} derivative={derivative_loss_weight} '
        f'hold_smooth={hold_loss_weight} hold_fit={hold_fit_loss_weight}',
    )
    _rank0_print(is_main_process, f'Huber deltas: point={point_huber_delta} derivative={derivative_huber_delta}')
    _rank0_print(is_main_process, f'Hold activity alpha: {hold_activity_alpha}')
    _rank0_print(is_main_process, f'Enable per-rank batch debug: {enable_rank_batch_debug}')
    _rank0_print(is_main_process, f'Curriculum enabled: {curriculum_enabled}')
    _rank0_print(is_main_process, f'Curriculum train schedule (ms): {curriculum_schedule_ms}')
    _rank0_print(is_main_process, f'Curriculum stage epochs: {curriculum_stage_epochs}')
    _rank0_print(is_main_process, f'Curriculum plateau patience: {curriculum_plateau_patience}')
    _rank0_print(is_main_process, f'Train max duration (ms): {current_train_max_duration_ms}')
    _rank0_print(is_main_process, f'Validation max duration (ms): {current_val_max_duration_ms}')
    _rank0_print(is_main_process, f'Test max duration (ms): {current_test_max_duration_ms}')
    if train_avg_seq_len is not None and train_max_seq_len is not None:
        _rank0_print(is_main_process, f'Train samples avg_seq_len={train_avg_seq_len:.1f} max_seq_len={train_max_seq_len}')
    if val_avg_seq_len is not None and val_max_seq_len is not None:
        _rank0_print(is_main_process, f'Validation samples avg_seq_len={val_avg_seq_len:.1f} max_seq_len={val_max_seq_len}')
    if test_avg_seq_len is not None and test_max_seq_len is not None:
        _rank0_print(is_main_process, f'Test samples avg_seq_len={test_avg_seq_len:.1f} max_seq_len={test_max_seq_len}')
    _rank0_print(is_main_process, f'Run seed: {run_seed}')
    if distributed:
        _rank0_print(is_main_process, f'Using DistributedDataParallel with world_size={world_size}')
    elif num_gpus > 1:
        _rank0_print(is_main_process, 'Multiple GPUs detected. Launch with torchrun to enable DistributedDataParallel scaling.')

    if is_main_process:
        seed_log_path = logs_folder / 'run_seed.txt'
        seed_log_path.write_text(f'{run_seed}\n', encoding='utf-8')

    net = Network(dataset.input_shape)
    net = net.to(device)
    if distributed:
        dist.barrier()
        try:
            _debug_distributed_cuda_collective(rank=rank, device=device, distributed=distributed)
            dist.barrier()
            net = DDP(
                net,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )
        except Exception as exc:
            _rank_print(rank, f'DDP construction failed: {type(exc).__name__}: {exc}')
            _debug_model_state(net, rank=rank, local_rank=local_rank, device=device)
            raise

    optimizer = torch.optim.RAdam(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=0.2)
    amp_enabled = bool(use_amp and torch.cuda.is_available())
    if _USE_NEW_AMP_API:
        scaler = GradScaler('cuda', enabled=amp_enabled)
    else:
        scaler = GradScaler(enabled=amp_enabled)

    _rank0_print(is_main_process, f'Automatic mixed precision enabled: {amp_enabled}')

    base_net = net.module if isinstance(net, DDP) else net
    start_epoch = 0
    best_val_loss = float('inf')
    history = []
    resume_checkpoint_path = _find_latest_checkpoint(logs_folder)

    if resume_checkpoint_path is not None:
        if distributed:
            dist.barrier()
        checkpoint = torch.load(resume_checkpoint_path, map_location=device)
        base_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if amp_enabled and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        best_val_loss = float(checkpoint.get('best_val_loss', best_val_loss))
        history = list(checkpoint.get('history', history))
        start_epoch = int(checkpoint.get('epoch', 0))
        current_train_max_duration_ms = checkpoint.get('current_train_max_duration_ms', current_train_max_duration_ms)
        curriculum_schedule_ms = list(checkpoint.get('curriculum_schedule_ms', curriculum_schedule_ms))
        curriculum_stage_index = int(checkpoint.get('curriculum_stage_index', curriculum_stage_index))
        current_val_max_duration_ms = checkpoint.get(
            'current_val_max_duration_ms',
            _curriculum_duration_for_stage(
                initial_duration_ms=val_max_duration_ms,
                curriculum_schedule_ms=curriculum_schedule_ms,
                curriculum_stage_index=curriculum_stage_index,
            ),
        )
        current_test_max_duration_ms = checkpoint.get(
            'current_test_max_duration_ms',
            _curriculum_duration_for_stage(
                initial_duration_ms=test_max_duration_ms,
                curriculum_schedule_ms=curriculum_schedule_ms,
                curriculum_stage_index=curriculum_stage_index,
            ),
        )
        if not curriculum_enabled:
            curriculum_schedule_ms = [None]
            curriculum_stage_index = 0
            current_train_max_duration_ms = None
            current_val_max_duration_ms = None
            current_test_max_duration_ms = None
        initial_batch_size = int(checkpoint.get('initial_batch_size', initial_batch_size))
        current_batch_size = int(checkpoint.get(
            'current_batch_size',
            _curriculum_batch_size_for_stage(
                initial_batch_size=initial_batch_size,
                initial_train_max_duration_ms=train_max_duration_ms,
                current_train_max_duration_ms=current_train_max_duration_ms,
                min_batch_size=minimum_batch_size,
            ),
        ))
        if not curriculum_enabled:
            current_batch_size = _curriculum_batch_size_for_stage(
                initial_batch_size=initial_batch_size,
                initial_train_max_duration_ms=train_max_duration_ms,
                current_train_max_duration_ms=current_train_max_duration_ms,
                min_batch_size=minimum_batch_size,
            )
        curriculum_stage_start_epoch = int(checkpoint.get('curriculum_stage_start_epoch', curriculum_stage_start_epoch))
        curriculum_stage_best_loss = float(checkpoint.get('curriculum_stage_best_loss', curriculum_stage_best_loss))
        curriculum_epochs_since_improvement = int(checkpoint.get('curriculum_epochs_since_improvement', curriculum_epochs_since_improvement))
        (
            train_loader,
            val_loader,
            test_loader,
            dataset,
            train_avg_seq_len,
            train_max_seq_len,
            val_avg_seq_len,
            val_max_seq_len,
            test_avg_seq_len,
            test_max_seq_len,
        ) = _build_active_dataloaders(
            current_train_max_duration_ms,
            current_val_max_duration_ms,
            current_test_max_duration_ms,
            current_batch_size,
        )
        _rank0_print(is_main_process, f'Resuming from checkpoint {resume_checkpoint_path.name} at epoch {start_epoch}')
        _rank0_print(is_main_process, f'Resumed curriculum stage {curriculum_stage_index + 1}/{len(curriculum_schedule_ms)} with train max duration {current_train_max_duration_ms} ms')
        _rank0_print(is_main_process, f'Resumed validation max duration: {current_val_max_duration_ms} ms')
        _rank0_print(is_main_process, f'Resumed test max duration: {current_test_max_duration_ms} ms')
        _rank0_print(is_main_process, f'Resumed active batch size: {current_batch_size}')

    epoch_durations: List[float] = []

    def run_epoch(loader, training: bool, epoch_index: int):
        if loader is None:
            return None

        net.train(mode=training)
        running_loss = 0.0
        num_batches = 0
        phase_name = 'train' if training else 'eval'
        total_batches = len(loader)
        epoch_start_time = time.perf_counter()
        progress_interval = max(1, min(100, total_batches // 20))
        last_iteration_end = epoch_start_time

        _rank0_print(is_main_process, f'Starting {phase_name} epoch pass with {total_batches} local batches')

        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch_index, batch in enumerate(loader, start=1):
                iteration_start_time = time.perf_counter()
                data_wait_time = iteration_start_time - last_iteration_end
                spikes = batch['spikes'].reshape(batch['spikes'].shape[0], *dataset.input_shape, batch['spikes'].shape[-1])

                spikes = spikes.to(device, non_blocking=True).float()

                target = batch['target'].to(device, non_blocking=True)
                lengths = batch['lengths'].to(device, non_blocking=True)

                if training:
                    _assert_batch_lengths_within_limit(
                        lengths=lengths,
                        expected_max_seq_len=train_max_seq_len,
                        phase_name='train',
                        batch_index=batch_index,
                    )

                if training:
                    optimizer.zero_grad(set_to_none=True)

                if amp_enabled:
                    if _USE_NEW_AMP_API:
                        autocast_context = autocast(device_type='cuda', enabled=True)
                    else:
                        autocast_context = autocast(enabled=True)
                else:
                    autocast_context = nullcontext()

                with autocast_context:
                    output = net(spikes)
                    loss = masked_force_loss(
                        output,
                        target,
                        lengths,
                        spikes,
                        point_loss_weight=point_loss_weight,
                        derivative_loss_weight=derivative_loss_weight,
                        hold_loss_weight=hold_loss_weight,
                        hold_fit_loss_weight=hold_fit_loss_weight,
                        point_huber_delta=point_huber_delta,
                        derivative_huber_delta=derivative_huber_delta,
                        hold_activity_alpha=hold_activity_alpha,
                    )

                if enable_rank_batch_debug and batch_index == 1:
                    _debug_rank_batch_state(
                        rank=rank,
                        local_rank=local_rank,
                        device=device,
                        phase_name=phase_name,
                        batch_index=batch_index,
                        spikes=spikes,
                        target=target,
                        lengths=lengths,
                        output=output,
                        loss=loss,
                    )

                if training:
                    if amp_enabled:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
                        optimizer.step()

                loss_value = float(loss.detach().item())
                running_loss += loss_value
                num_batches += 1
                step_time = time.perf_counter() - iteration_start_time
                last_iteration_end = time.perf_counter()

                if is_main_process and (batch_index == 1 or batch_index % progress_interval == 0 or batch_index == total_batches):
                    elapsed = time.perf_counter() - epoch_start_time
                    avg_batch_time = elapsed / batch_index
                    epoch_eta_seconds = avg_batch_time * max(0, total_batches - batch_index)
                    estimated_epoch_total = avg_batch_time * total_batches
                    remaining_epochs_after_current = max(0, epochs - (epoch_index + 1))
                    reference_epoch_duration = (
                        (sum(epoch_durations) / len(epoch_durations)) if epoch_durations else estimated_epoch_total
                    )
                    training_eta_seconds = epoch_eta_seconds + remaining_epochs_after_current * reference_epoch_duration
                    mean_seq_len = float(lengths.float().mean().item())
                    max_seq_len = int(lengths.max().item())
                    _rank0_print(
                        is_main_process,
                        f'[{phase_name}] batch {batch_index}/{total_batches} '
                        f'loss={loss_value:.6f} avg_loss={running_loss / num_batches:.6f} '
                        f'data_wait={data_wait_time:.2f}s step_time={step_time:.2f}s avg_batch_time={avg_batch_time:.2f}s '
                        f'epoch_eta={_format_duration(epoch_eta_seconds)} training_eta={_format_duration(training_eta_seconds)} '
                        f'mean_seq_len={mean_seq_len:.1f} max_seq_len={max_seq_len}'
                    )

        epoch_loss = running_loss / max(1, num_batches)
        return _reduce_average(epoch_loss, device=device, distributed=distributed)

    final_test_loss = None

    for epoch in range(start_epoch, epochs):
        epoch_wall_start = time.perf_counter()
        _set_loader_epoch(train_loader, epoch)
        _set_loader_epoch(val_loader, epoch)
        _set_loader_epoch(test_loader, epoch)

        _rank0_print(is_main_process, "Performing training pass...")
        train_loss = run_epoch(train_loader, training=True, epoch_index=epoch)
        _rank0_print(is_main_process, "Performing validation pass...")
        val_loss = run_epoch(val_loader, training=False, epoch_index=epoch)
        scheduler.step()
        epoch_duration = time.perf_counter() - epoch_wall_start
        epoch_durations.append(epoch_duration)

        current_lr = optimizer.param_groups[0]['lr']
        monitor_loss = val_loss if val_loss is not None else train_loss
        if is_main_process:
            avg_epoch_duration = sum(epoch_durations) / len(epoch_durations)
            remaining_epochs = max(0, epochs - (epoch + 1))
            training_eta_seconds = remaining_epochs * avg_epoch_duration
            history.append(
                {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'lr': current_lr,
                    'train_max_duration_ms': current_train_max_duration_ms,
                    'val_max_duration_ms': current_val_max_duration_ms,
                    'test_max_duration_ms': current_test_max_duration_ms,
                    'batch_size': current_batch_size,
                    'curriculum_stage_index': curriculum_stage_index,
                }
            )

            status = f"Epoch [{epoch + 1:03d}/{epochs}] train={train_loss:.6f}"
            if val_loss is not None:
                status += f" val={val_loss:.6f}"
            status += (
                f" lr={current_lr:.6e} epoch_time={_format_duration(epoch_duration)}"
                f" training_eta={_format_duration(training_eta_seconds)}"
            )
            _rank0_print(is_main_process, status)

            if monitor_loss is not None and monitor_loss < best_val_loss:
                best_val_loss = monitor_loss
                torch.save(base_net.state_dict(), trained_folder / 'network.pt')
                _rank0_print(is_main_process, f'Saved new best model with loss {best_val_loss:.6f}')

            if (epoch + 1) % 10 == 0 or epoch == 0:
                _save_checkpoint(
                    checkpoint_path=logs_folder / f'checkpoint_{epoch + 1:03d}.pt',
                    epoch=epoch + 1,
                    base_net=base_net,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    best_val_loss=best_val_loss,
                    run_seed=run_seed,
                    history=history,
                    current_train_max_duration_ms=current_train_max_duration_ms,
                    current_val_max_duration_ms=current_val_max_duration_ms,
                    current_test_max_duration_ms=current_test_max_duration_ms,
                    curriculum_schedule_ms=curriculum_schedule_ms,
                    curriculum_stage_index=curriculum_stage_index,
                    initial_batch_size=initial_batch_size,
                    current_batch_size=current_batch_size,
                    curriculum_stage_start_epoch=curriculum_stage_start_epoch,
                    curriculum_stage_best_loss=curriculum_stage_best_loss,
                    curriculum_epochs_since_improvement=curriculum_epochs_since_improvement,
                )

            if (epoch + 1) % 2 == 0:
                base_net.grad_flow(str(trained_folder) + '/')
                loss_plot_path = logs_folder / 'loss_curve.png'
                _save_loss_plot(history, loss_plot_path)
                _rank0_print(is_main_process, f'Saved loss plot to {loss_plot_path}')

            with open(logs_folder / 'training_history.csv', 'w', newline='') as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=[
                        'epoch',
                        'train_loss',
                        'val_loss',
                        'lr',
                        'train_max_duration_ms',
                        'val_max_duration_ms',
                        'test_max_duration_ms',
                        'batch_size',
                        'curriculum_stage_index',
                    ],
                )
                writer.writeheader()
                writer.writerows(history)

        if monitor_loss is not None:
            if monitor_loss < (curriculum_stage_best_loss - curriculum_plateau_min_delta):
                curriculum_stage_best_loss = float(monitor_loss)
                curriculum_epochs_since_improvement = 0
            else:
                curriculum_epochs_since_improvement += 1

        curriculum_stage_epochs_completed = (epoch + 1) - curriculum_stage_start_epoch
        stage_complete = curriculum_stage_epochs_completed >= curriculum_stage_epochs
        plateau_reached = curriculum_epochs_since_improvement >= curriculum_plateau_patience
        can_advance_curriculum = curriculum_enabled and curriculum_stage_index < (len(curriculum_schedule_ms) - 1)

        if can_advance_curriculum and (stage_complete or plateau_reached):
            transition_reason = 'plateau' if plateau_reached else 'stage_complete'
            if is_main_process:
                _save_checkpoint(
                    checkpoint_path=logs_folder / f'checkpoint_stage_{curriculum_stage_index + 1:02d}_epoch_{epoch + 1:03d}.pt',
                    epoch=epoch + 1,
                    base_net=base_net,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    best_val_loss=best_val_loss,
                    run_seed=run_seed,
                    history=history,
                    current_train_max_duration_ms=current_train_max_duration_ms,
                    current_val_max_duration_ms=current_val_max_duration_ms,
                    current_test_max_duration_ms=current_test_max_duration_ms,
                    curriculum_schedule_ms=curriculum_schedule_ms,
                    curriculum_stage_index=curriculum_stage_index,
                    initial_batch_size=initial_batch_size,
                    current_batch_size=current_batch_size,
                    curriculum_stage_start_epoch=curriculum_stage_start_epoch,
                    curriculum_stage_best_loss=curriculum_stage_best_loss,
                    curriculum_epochs_since_improvement=curriculum_epochs_since_improvement,
                )

            previous_train_max_duration_ms = current_train_max_duration_ms
            previous_val_max_duration_ms = current_val_max_duration_ms
            previous_test_max_duration_ms = current_test_max_duration_ms
            previous_batch_size = current_batch_size
            curriculum_stage_index += 1
            current_train_max_duration_ms = curriculum_schedule_ms[curriculum_stage_index]
            current_val_max_duration_ms = _curriculum_duration_for_stage(
                initial_duration_ms=val_max_duration_ms,
                curriculum_schedule_ms=curriculum_schedule_ms,
                curriculum_stage_index=curriculum_stage_index,
            )
            current_test_max_duration_ms = _curriculum_duration_for_stage(
                initial_duration_ms=test_max_duration_ms,
                curriculum_schedule_ms=curriculum_schedule_ms,
                curriculum_stage_index=curriculum_stage_index,
            )
            current_batch_size = _curriculum_batch_size_for_stage(
                initial_batch_size=initial_batch_size,
                initial_train_max_duration_ms=train_max_duration_ms,
                current_train_max_duration_ms=current_train_max_duration_ms,
                min_batch_size=minimum_batch_size,
            )
            curriculum_stage_start_epoch = epoch + 1
            curriculum_stage_best_loss = float('inf')
            curriculum_epochs_since_improvement = 0
            (
                train_loader,
                val_loader,
                test_loader,
                dataset,
                train_avg_seq_len,
                train_max_seq_len,
                val_avg_seq_len,
                val_max_seq_len,
                test_avg_seq_len,
                test_max_seq_len,
            ) = _build_active_dataloaders(
                current_train_max_duration_ms,
                current_val_max_duration_ms,
                current_test_max_duration_ms,
                current_batch_size,
            )

            _rank0_print(
                is_main_process,
                f'Advancing curriculum from {previous_train_max_duration_ms} ms to {current_train_max_duration_ms} ms '
                f'after epoch {epoch + 1} due to {transition_reason}.',
            )
            _rank0_print(
                is_main_process,
                f'Updated validation max duration from {previous_val_max_duration_ms} ms to {current_val_max_duration_ms} ms',
            )
            _rank0_print(
                is_main_process,
                f'Updated test max duration from {previous_test_max_duration_ms} ms to {current_test_max_duration_ms} ms',
            )
            _rank0_print(
                is_main_process,
                f'Updated active batch size from {previous_batch_size} to {current_batch_size}',
            )
            if train_avg_seq_len is not None and train_max_seq_len is not None:
                _rank0_print(is_main_process, f'Updated train samples avg_seq_len={train_avg_seq_len:.1f} max_seq_len={train_max_seq_len}')
            if val_avg_seq_len is not None and val_max_seq_len is not None:
                _rank0_print(is_main_process, f'Validation samples avg_seq_len={val_avg_seq_len:.1f} max_seq_len={val_max_seq_len}')
            if test_avg_seq_len is not None and test_max_seq_len is not None:
                _rank0_print(is_main_process, f'Test samples avg_seq_len={test_avg_seq_len:.1f} max_seq_len={test_max_seq_len}')

    best_model_path = trained_folder / 'network.pt'
    if distributed:
        dist.barrier()
    if best_model_path.exists():
        state_dict = torch.load(best_model_path, map_location=device)
        base_net.load_state_dict(state_dict)

    _rank0_print(is_main_process, "Performing final test pass...")
    final_test_loss = run_epoch(test_loader, training=False, epoch_index=epochs)
    if final_test_loss is not None:
        _rank0_print(is_main_process, f'Final test loss: {final_test_loss:.6f}')

    if history:
        if is_main_process:
            loss_plot_path = logs_folder / 'loss_curve.png'
            _save_loss_plot(history, loss_plot_path)
            _rank0_print(is_main_process, f'Saved loss plot to {loss_plot_path}')

    if is_main_process and best_model_path.exists():
        base_net.export_hdf5(trained_folder / 'network.net')
        _rank0_print(is_main_process, f'Exported best model to {trained_folder / "network.net"}')

    _cleanup_distributed(distributed)

if __name__ == "__main__":
    main()