import torch
import warnings
import inspect

from typing import Union, Tuple, List

import numpy as np
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerTopkLoss import nnUNetTrainerDiceTopK10Loss

import os
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.run.run_training import maybe_load_checkpoint as nnunet_maybe_load_checkpoint
from typing import Union, List, Tuple, Optional
import torch.nn as nn
import math
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
import mlflow
import numpy as np 
from torch import distributed as dist
from nnunetv2.utilities.collate_outputs import collate_outputs

class nnUNetSequentialLR(SequentialLR):
    """
    Custom wrapper to make SequentialLR compatible with nnU-Net.
    nnU-Net calls .step(epoch), but SequentialLR only wants .step().
    """
    def step(self, epoch=None):
        # We ignore the 'epoch' argument passed by nnU-Net
        super().step()
class nnUNetTrainer_5epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 5


class nnUNetTrainer_1epoch(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1


class nnUNetTrainer_10epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 10


class nnUNetTrainer_20epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 20


class nnUNetTrainer_50epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 50


class nnUNetTrainer_100epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainer_250epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250


class nnUNetTrainer_500epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500


class nnUNetTrainer_750epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 750


class nnUNetTrainer_2000epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2000


class nnUNetTrainer_4000epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 4000


class nnUNetTrainer_8000epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 8000

class nnUNetTrainer_CustomStrongerAug(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 300
        self.initial_lr = 1e-2
        self.warmup = 0.02
        self.warmup_start = 0.3
        self.head_weight = 1
        self.finetune_run = False
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("Shoulder_Segmentation_Experiments")


    def initialize(self):
        if not self.was_initialized:
            ## DDP batch size and oversampling can differ between workers and needs adaptation
            # we need to change the batch size in DDP because we don't use any of those distributed samplers
            self._set_batch_size_and_oversample()

            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            sig = inspect.signature(self.build_network_architecture)
            if 'plans_manager' in sig.parameters:
                self.network = self.build_network_architecture(
                    self.plans_manager,
                    self.configuration_manager,
                    self.num_input_channels,
                    self.label_manager.num_segmentation_heads,
                    self.enable_deep_supervision
                ).to(self.device)
            else:
                warnings.warn(
                    f"Trainer {self.__class__.__name__} uses the old build_network_architecture signature. "
                    "Please update to the new signature: "
                    "build_network_architecture(plans_manager, configuration_manager, "
                    "num_input_channels, num_output_channels, enable_deep_supervision). "
                    "The old signature will be removed in a future version.",
                    DeprecationWarning, stacklevel=2,
                )
                self.network = self.build_network_architecture(
                    self.configuration_manager.network_arch_class_name,
                    self.configuration_manager.network_arch_init_kwargs,
                    self.configuration_manager.network_arch_init_kwargs_req_import,
                    self.num_input_channels,
                    self.label_manager.num_segmentation_heads,
                    self.enable_deep_supervision
                ).to(self.device)
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)
            
            fine_tune = self.finetune_run 

            if fine_tune:
                head_params = []
                backbone_params = []
                for name, param in self.network.named_parameters():
                    if not param.requires_grad:
                        continue
                    # Targeting the segmentation heads for the higher LR
                    if 'seg_layers' in name or 'output' in name:
                        head_params.append(param)
                    else:
                        backbone_params.append(param)

                # Head gets 10x the initial_lr (1e-2 if initial is 1e-3)
                # Backbone/Adapters get initial_lr (1e-3)
                param_groups = [
                    {'params': backbone_params, 'lr': self.initial_lr},
                    {'params': head_params, 'lr': self.initial_lr *self.head_weight }
                ]
                
                self.optimizer = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)

                # 3. Setup Sequential Scheduler: 5%-epoch Warmup + Cosine Decay 
                total_epochs = self.num_epochs
                if total_epochs:
                    warmup_epochs = int( 0.05 * total_epochs)
                else: warmup_epochs = 10 
                

                warmup_scheduler = LinearLR(
                    self.optimizer, 
                    start_factor=0.3, 
                    total_iters=warmup_epochs
                )
                
                cosine_scheduler = CosineAnnealingLR(
                    self.optimizer, 
                    T_max=(total_epochs - warmup_epochs), 
                    eta_min=1e-6
                )
                
                self.lr_scheduler = SequentialLR(
                    self.optimizer, 
                    schedulers=[warmup_scheduler, cosine_scheduler], 
                    milestones=[warmup_epochs]
                )
            else: 
                self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
                

                total_epochs = self.num_epochs
                if total_epochs:
                    warmup_epochs = int(self.warmup * total_epochs)
                    
                else: warmup_epochs = 10 
                remaining_epochs = total_epochs - warmup_epochs
                warmup_scheduler = LinearLR(
                    self.optimizer, 
                    start_factor=0.3, 
                    total_iters=warmup_epochs
                )
                
                poly_scheduler = PolyLRScheduler(
                    self.optimizer, 
                    self.initial_lr, 
                    remaining_epochs
                )
                
                self.lr_scheduler = SequentialLR(
                    self.optimizer, 
                    schedulers=[warmup_scheduler, poly_scheduler], 
                    milestones=[warmup_epochs]
                )
                
                
                
                
                

            
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()

            

            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    
    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
        transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False  # , mode_seg='nearest'
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.15),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.
        ))
        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5, benchmark=True
            ), apply_probability=0.2
        ))
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.65, 1.5)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=0.25
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.3
        ))
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(
                MirrorTransform(
                    allowed_axes=mirror_axes
                )
            )

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )
        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 5),
                        p_per_label=1
                    ), apply_probability=0.5
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.25,
                        p_per_label=1
                    ), apply_probability=0.4
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)
    
    def on_train_start(self): 
        super().on_train_start()
        if mlflow.active_run(): 
            mlflow.end_run()
        if self.local_rank == 0:
            run_name = f"Fold_{self.fold}_{self.configuration_name}"
            mlflow.start_run(run_name=run_name, log_system_metrics=True)

            mlflow.log_params({
                "fold": self.fold,
                "config": self.configuration_name,
                "lr" : self.initial_lr,
                "epochs": self.num_epochs, 
                'warmup': self.warmup,
                'finetuning' : self.finetune_run
            })

    def on_train_epoch_start(self):
        self.network.train()
        self.lr_scheduler.step()
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

    def get_dataloaders(self):
        # 1. Get the standard loaders (which are MultiThreadedAugmenters)
        train_loader, val_loader = super().get_dataloaders()

        # 2. Check if it's the right class before modifying
        if isinstance(train_loader, MultiThreadedAugmenter):
            print(f"Old Cache Size: {train_loader.num_cached_per_thread}")
            
            # 3. FORCE the buffer depth (The 180GB Fix)
            # This works because the threads haven't started yet!
            train_loader.num_cached_per_thread = 6
            
            print(f"New Cache Size: {train_loader.num_cached_per_thread}")
        else:
            print("Warning: Loader is not MultiThreadedAugmenter? Check your setup.")

        return train_loader, val_loader
    
    def on_epoch_end(self):
        """Logs metrics at the end of every epoch using the MetaLogger interface"""
        super().on_epoch_end()
        
        # Ensure we only log from the main process and if MLflow is active
        if self.local_rank == 0 and mlflow.active_run():
            
            # Helper to get the last element of a logged list safely
            def get_last(key):
                # self.logger is the MetaLogger instance
                # get_value(key, step=None) returns the full list from local_logger
                vals = self.logger.get_value(key, step=None)
                return vals[-1] if len(vals) > 0 else None

            log_metrics = {}

            # 1. Grab Train and Val Losses
            train_loss = get_last('train_losses')
            val_loss = get_last('val_losses')
            
            if train_loss is not None: log_metrics['avg_train_loss_epoch'] = train_loss
            if val_loss is not None: log_metrics['avg_val_loss_epoch'] = val_loss

            # 2. Log Learning Rates (from optimizer directly is safer)
            for i, param_group in enumerate(self.optimizer.param_groups):
                log_metrics[f'lr_group_{i}'] = param_group['lr']

            # 3. Log Foreground Dice
            mean_dice = get_last('mean_fg_dice')
            ema_dice = get_last('ema_fg_dice')
            
            if mean_dice is not None: log_metrics['val_mean_fg_dice'] = mean_dice
            if ema_dice is not None: log_metrics['val_ema_fg_dice'] = ema_dice
            
            # 4. Epoch Timing
            start_ts = get_last('epoch_start_timestamps')
            end_ts = get_last('epoch_end_timestamps')
            
            if start_ts is not None and end_ts is not None:
                log_metrics['epoch_time'] = np.round(end_ts - start_ts, decimals=2)

            # Send to MLflow
            mlflow.log_metrics(log_metrics, step=self.current_epoch)
            

    def on_train_end(self):
        """Finalize the run and log the model"""
        # Always call the super first so nnU-Net finishes its internal work
        super().on_train_end()
        
        if self.local_rank == 0 and mlflow.active_run():
            print(f"Logging Fold {self.fold} model to MLflow...")
            try:
                # We use a timeout or a specific check here if needed
                mlflow.pytorch.log_model(
                    pytorch_model=self.network, 
                    artifact_path="model_final"
                )
                print(f"Successfully logged Fold {self.fold} model.")
            except Exception as e:
                # We catch the error, print it, but DO NOT raise it
                print("="*30)
                print(f"WARNING: MLflow model logging failed for Fold {self.fold}!")
                print(f"Error: {e}")
                print("Continuing to next fold anyway...")
                print("="*30)
            finally:
                # Ensure the run is closed so the next fold doesn't conflict
                try:
                    mlflow.end_run()
                except:
                    pass

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        if self.is_ddp:
            world_size = dist.get_world_size()

            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated['loss'])

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
        metrics_dict = {f"Dice_Class_{i}": score for i, score in enumerate(global_dc_per_class)}
        mlflow.log_metrics(metrics_dict, step=self.current_epoch)
        mean_fg_dice = np.nanmean(global_dc_per_class)
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        Safe override for the 'NoneType' decoder error.
        """
        if self.network is None:
            return

        # 1. Direct access (Standard)
        if hasattr(self.network, 'decoder') and self.network.decoder is not None:
            self.network.decoder.deep_supervision = enabled
        # 2. DDP/Module wrapping
        elif hasattr(self.network, 'module') and hasattr(self.network.module, 'decoder'):
            self.network.module.decoder.deep_supervision = enabled
        # 3. If it's a list (Standard nnU-Net multi-head behavior)
        elif isinstance(self.network, (list, tuple)):
            for m in self.network:
                if m is not None and hasattr(m, 'decoder'):
                    m.decoder.deep_supervision = enabled
        else:
            self.print_to_log_file("WARNING: Could not set deep_supervision. 'decoder' not found.")


    def perform_actual_validation(self, save_probabilities: bool = False):
        # 1. Build the network architecture if it's missing
        if not self.was_initialized:
            self.initialize()
        
        # 2. Check if weights are loaded. If not, load them from the expected output folder.
        # nnU-Net usually stores the best checkpoint as 'checkpoint_final.pth' or 'checkpoint_best.pth'
        checkpoint_path = os.path.join(self.output_folder, 'checkpoint_final.pth')
        
        if self.network is not None and os.path.exists(checkpoint_path):
            print(f"Loading weights for validation from: {checkpoint_path}")
            self.load_checkpoint(checkpoint_path)
        elif not os.path.exists(checkpoint_path):
            print(f"WARNING: No checkpoint found at {checkpoint_path}. Validating with random weights!")

        # 3. Now proceed with the standard validation
        super().perform_actual_validation(save_probabilities)



class nnUNetTrainer_500epochs_StrongerAug(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 500
        self.initial_lr = 1e-2
        self.warmup = 0.02
        self.warmup_start = 0.3
        self.head_weight = 1
        self.finetune_run = False
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("Spine_V3_lowres")


    def initialize(self):
        if not self.was_initialized:
            ## DDP batch size and oversampling can differ between workers and needs adaptation
            # we need to change the batch size in DDP because we don't use any of those distributed samplers
            self._set_batch_size_and_oversample()

            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision
            ).to(self.device)
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)
            
            fine_tune = self.finetune_run 

            if fine_tune:
                head_params = []
                backbone_params = []
                for name, param in self.network.named_parameters():
                    if not param.requires_grad:
                        continue
                    # Targeting the segmentation heads for the higher LR
                    if 'seg_layers' in name or 'output' in name:
                        head_params.append(param)
                    else:
                        backbone_params.append(param)

                # Head gets 10x the initial_lr (1e-2 if initial is 1e-3)
                # Backbone/Adapters get initial_lr (1e-3)
                param_groups = [
                    {'params': backbone_params, 'lr': self.initial_lr},
                    {'params': head_params, 'lr': self.initial_lr *self.head_weight }
                ]
                
                self.optimizer = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)

                # 3. Setup Sequential Scheduler: 5%-epoch Warmup + Cosine Decay 
                total_epochs = self.num_epochs
                if total_epochs:
                    warmup_epochs = int( 0.05 * total_epochs)
                else: warmup_epochs = 10 
                

                warmup_scheduler = LinearLR(
                    self.optimizer, 
                    start_factor=0.3, 
                    total_iters=warmup_epochs
                )
                
                cosine_scheduler = CosineAnnealingLR(
                    self.optimizer, 
                    T_max=(total_epochs - warmup_epochs), 
                    eta_min=1e-6
                )
                
                self.lr_scheduler = SequentialLR(
                    self.optimizer, 
                    schedulers=[warmup_scheduler, cosine_scheduler], 
                    milestones=[warmup_epochs]
                )
            else: 
                self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
                

                total_epochs = self.num_epochs
                if total_epochs:
                    warmup_epochs = int(self.warmup * total_epochs)
                    
                else: warmup_epochs = 10 
                remaining_epochs = total_epochs - warmup_epochs
                warmup_scheduler = LinearLR(
                    self.optimizer, 
                    start_factor=0.3, 
                    total_iters=warmup_epochs
                )
                
                poly_scheduler = PolyLRScheduler(
                    self.optimizer, 
                    self.initial_lr, 
                    remaining_epochs
                )
                
                self.lr_scheduler = SequentialLR(
                    self.optimizer, 
                    schedulers=[warmup_scheduler, poly_scheduler], 
                    milestones=[warmup_epochs]
                )
                
                
                
                
                

            
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()

            

            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    
    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
        transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False  # , mode_seg='nearest'
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.15),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.
        ))
        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5, benchmark=True
            ), apply_probability=0.2
        ))
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.65, 1.5)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=0.25
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.3
        ))
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(
                MirrorTransform(
                    allowed_axes=mirror_axes
                )
            )

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )
        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 5),
                        p_per_label=1
                    ), apply_probability=0.5
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.25,
                        p_per_label=1
                    ), apply_probability=0.4
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)
    
    def on_train_start(self): 
        super().on_train_start()
        if mlflow.active_run(): 
            mlflow.end_run()
        if self.local_rank == 0:
            run_name = f"Fold_{self.fold}_{self.configuration_name}"
            mlflow.start_run(run_name=run_name)

            mlflow.log_params({
                "fold": self.fold,
                "config": self.configuration_name,
                "lr" : self.initial_lr,
                "epochs": self.num_epochs, 
                'warmup': self.warmup,
                'finetuning' : self.finetune_run
            })

    def on_train_epoch_start(self):
        self.network.train()
        self.lr_scheduler.step()
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

    def get_dataloaders(self):
        # 1. Get the standard loaders (which are MultiThreadedAugmenters)
        train_loader, val_loader = super().get_dataloaders()

        # 2. Check if it's the right class before modifying
        if isinstance(train_loader, MultiThreadedAugmenter):
            print(f"Old Cache Size: {train_loader.num_cached_per_thread}")
            
            # 3. FORCE the buffer depth (The 180GB Fix)
            # This works because the threads haven't started yet!
            train_loader.num_cached_per_thread = 6
            
            print(f"New Cache Size: {train_loader.num_cached_per_thread}")
        else:
            print("Warning: Loader is not MultiThreadedAugmenter? Check your setup.")

        return train_loader, val_loader
    
    def on_epoch_end(self):
        """Logs metrics at the end of every epoch"""
        super().on_epoch_end()
        
        if self.local_rank == 0 and mlflow.active_run():
            # Grab latest metrics from nnUNet's internal logger
            log_metrics = {
                'avg_train_loss_epoch': self.logger.my_fantastic_logging['train_losses'][-1],
                'avg_val_loss_epoch': self.logger.my_fantastic_logging['val_losses'][-1],
            }
            for i, param_group in enumerate(self.optimizer.param_groups):
                log_metrics[f'lr_group_{i}'] = param_group['lr']
            mlflow.log_metrics(log_metrics, step=self.current_epoch)
            
            if 'mean_fg_dice' in self.logger.my_fantastic_logging:
                mean_fg_dice = self.logger.my_fantastic_logging['mean_fg_dice'][-1]
                mlflow.log_metric('val_mean_fg_dice', mean_fg_dice, step=self.current_epoch)
            if 'epoch_end_timestamps' in self.logger.my_fantastic_logging and 'epoch_start_timestamps' in self.logger.my_fantastic_logging:
                epoch_time = np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)
                mlflow.log_metric('epoch_time', epoch_time, step = self.current_epoch)
    
    
    def on_train_end(self):
        """Finalize the run and log the model"""
        # Always call the super first so nnU-Net finishes its internal work
        super().on_train_end()
        
        if self.local_rank == 0 and mlflow.active_run():
            print(f"Logging Fold {self.fold} model to MLflow...")
            try:
                # Fix deprec argument before handover 
                mlflow.pytorch.log_model(
                    pytorch_model=self.network, 
                    artifact_path="model_final"
                )
                print(f"Successfully logged Fold {self.fold} model.")
            except Exception as e:
                # We catch the error, print it, but DO NOT raise it
                print("="*30)
                print(f"WARNING: MLflow model logging failed for Fold {self.fold}!")
                print(f"Error: {e}")
                print("Continuing to next fold anyway...")
                print("="*30)
            finally:
                # Ensure the run is closed so the next fold doesn't conflict
                try:
                    mlflow.end_run()
                except:
                    pass

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        if self.is_ddp:
            world_size = dist.get_world_size()

            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated['loss'])

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
        metrics_dict = {f"Dice_Class_{i}": score for i, score in enumerate(global_dc_per_class)}
        mlflow.log_metrics(metrics_dict, step=self.current_epoch)
        mean_fg_dice = np.nanmean(global_dc_per_class)
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        Safe override for the 'NoneType' decoder error.
        """
        if self.network is None:
            return

        # 1. Direct access (Standard)
        if hasattr(self.network, 'decoder') and self.network.decoder is not None:
            self.network.decoder.deep_supervision = enabled
        # 2. DDP/Module wrapping
        elif hasattr(self.network, 'module') and hasattr(self.network.module, 'decoder'):
            self.network.module.decoder.deep_supervision = enabled
        # 3. If it's a list (Standard nnU-Net multi-head behavior)
        elif isinstance(self.network, (list, tuple)):
            for m in self.network:
                if m is not None and hasattr(m, 'decoder'):
                    m.decoder.deep_supervision = enabled
        else:
            self.print_to_log_file("WARNING: Could not set deep_supervision. 'decoder' not found.")


    def perform_actual_validation(self, save_probabilities: bool = False):
        # 1. Build the network architecture if it's missing
        if not self.was_initialized:
            self.initialize()
        
        # 2. Check if weights are loaded. If not, load them from the expected output folder.
        # nnU-Net usually stores the best checkpoint as 'checkpoint_final.pth' or 'checkpoint_best.pth'
        checkpoint_path = os.path.join(self.output_folder, 'checkpoint_final.pth')
        
        if self.network is not None and os.path.exists(checkpoint_path):
            print(f"Loading weights for validation from: {checkpoint_path}")
            self.load_checkpoint(checkpoint_path)
        elif not os.path.exists(checkpoint_path):
            print(f"WARNING: No checkpoint found at {checkpoint_path}. Validating with random weights!")

        # 3. Now proceed with the standard validation
        super().perform_actual_validation(save_probabilities)