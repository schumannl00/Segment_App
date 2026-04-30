import warnings
import os
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.run.run_training import maybe_load_checkpoint as nnunet_maybe_load_checkpoint
import math
from typing import Union, List, Tuple, Optional
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
# -------------------------------------------------------------------------
# 1. Custom Adapter Layer (SiLU + Parallel Conv)
# -------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import mlflow
import numpy as np
try:
    from torch._dynamo.eval_frame import OptimizedModule
except Exception:
    OptimizedModule = None

from batchgeneratorsv2.helpers.scalar_type import RandomScalar
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

from torch import distributed as dist
from nnunetv2.utilities.collate_outputs import collate_outputs
class LoRAConv3d(nn.Module):
    """
    nnU-Net–compatible LoRA Conv3D:
    - Preserves original weight identity
    - No nested Conv3d modules
    - Spatial LoRA (same kernel as base)
    """

    def __init__(self, conv: nn.Conv3d, r=16, alpha=32, dropout=0.3):
        super().__init__()

        # steal parameters (important!) also use parameter not buffer as we do not need custom logic 
        self.weight = nn.Parameter(conv.weight.data.clone(), requires_grad=False)
        self.bias = (
            nn.Parameter(conv.bias.data.clone(), requires_grad=False)
            if conv.bias is not None else None
        )

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups


        self.r = r
        self.scale = alpha / r

        # LoRA A: spatial
        self.lora_A = nn.Parameter(
            torch.zeros(
                r, self.in_channels // self.groups, *self.kernel_size
            )
        )

        # Normalization in bottleneck
        self.norm = nn.InstanceNorm3d(r, affine=True)

        self.act = nn.SiLU(inplace=True)

        # LoRA B: 1×1×1
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_channels, r, 1, 1, 1)
        )

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()   
        
    def forward(self, x):
        out = F.conv3d(
            x, self.weight, self.bias,
            self.stride, self.padding,
            self.dilation, self.groups
        )

        lora = F.conv3d(
            x, self.lora_A,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )

        #lora = self.norm(lora)
        lora = self.act(lora)
        lora = self.lora_dropout(lora)
        lora = F.conv3d(lora, self.lora_B, stride=1, padding=0) * self.scale

        return out + lora


# -------------------------------------------------------------------------
# 2. Universal Trainer Class (Selective Injection)
# -------------------------------------------------------------------------
class nnUNetTrainerUniversalAdapter(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):

        super().__init__(plans, configuration, fold, dataset_json, device)
        self.fold =fold


        #chnage if usingthat for any other training loop, hardcoding was easier than fighting with the pth loading in nnunet 
        self.weight_paths = {
            "3d_lowres": fr"D:\nnUNet_results\Dataset217_Spine\nnUNetTrainer_CustomStrongerAug__nnUNetPlans__3d_lowres\fold_{self.fold}\checkpoint_best.pth",
            "3d_fullres": r"D:\nnUNet_results\Dataset117_Spine\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_0\checkpoint_final.pth" ,         # Standard fullres
            "3d_cascade_fullres": fr"D:\nnUNet_results\Dataset217_Spine\nnUNetTrainer_CustomStrongerAug__nnUNetPlans__3d_cascade_fullres\fold_{self.fold}\checkpoint_best.pth"  # Explicit cascade name
        }

        #keep mlflow server at 500, there should be nothing competing with it, run it in wsl, it makes it easier in tmux so you do not close it by accident, use cloudflar quick tunnel for off site monitoring 
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("Scoliosis_LoRA_V3_cascade")
        # Adapter Hyperparameters
        self.adapter_rank = 32
        self.adapter_alpha = 64
        self.adapter_dropout = 0.3
        
        # How many encoder stages to SKIP (Freeze without adapting)?
        # Stage 0 is the highest resolution (input). Stage 5 is the bottleneck.
        # Default: Skip Stage 0 only.
        self.stages_to_skip = 0
        self.num_epochs = 200
        self.warmup = 0.1
        self.head_weight = 5 
        self.initial_lr = 1e-4
        self.loss = None

    def initialize(self):
    # Determine if we are starting fresh or resuming
        output_filename = os.path.join(self.output_folder, "checkpoint_latest.pth")
        need_pretrained_weights = not os.path.isfile(output_filename)
        
        if not self.was_initialized:
            # 1. Build Network with LoRA
            self.num_input_channels = determine_num_input_channels(
                self.plans_manager, self.configuration_manager, self.dataset_json
            )
            label_manager = self.plans_manager.get_label_manager(self.dataset_json)
            n_outputs = label_manager.num_segmentation_heads
            
            self.network = self.build_network_architecture(
                self.plans_manager, self.dataset_json, self.configuration_manager,
                self.num_input_channels, n_outputs, self.enable_deep_supervision
            ).to(self.device)
           
            # 2. Setup Optimizer with LR Disparity (Head vs Adapters)
            head_params = []
            adapter_params = []
            for name, param in self.network.named_parameters():
                if not param.requires_grad:
                    continue
                if 'seg_layers' in name or 'output' in name:
                    print(f"{name} sorted into head" )
                    head_params.append(param)
                else:
                    print(f"Sorted into Adapter: {name}")
                    adapter_params.append(param)

            # KEPT THE LR THE SAME AFTER RUNNING A FFT, OTHERWISE MAKE HEAD MORE AGGRESSIVE
            param_groups = [
                {'params': adapter_params, 'lr': self.initial_lr},
                {'params': head_params, 'lr': self.initial_lr * self.head_weight }
            ]
            
            self.optimizer = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)

            # 3. Setup Sequential Scheduler: 10%-epoch Warmup + Cosine Decay
            total_epochs = self.num_epochs
            warmup_epochs = int(self.num_epochs * self.warmup)
            
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

            # 4. Final Training Components
            self.loss = self._build_loss()
            if self.device.type == 'cuda':
                self.grad_scaler = torch.cuda.amp.GradScaler() 
            else:
                self.grad_scaler = None
            
            self.was_initialized = True

        # 5. Load Weights (Surgery vs Resume)
        if need_pretrained_weights:
            print("DEBUG: Fresh training. Loading & splicing pretrained weights...")
            self._smart_load_weights()
        else:
            print("DEBUG: Resuming training from checkpoint.")

        # Sanity check for stabilization, sometimes cache cleans not quickly enough automatically 
        torch.cuda.empty_cache()

        # Debug outputs
        if hasattr(self.network.decoder, 'seg_layers'):
            for i, m in enumerate(self.network.decoder.seg_layers):
                print(
                    f"Seg layer {i}:",
                    m.weight.abs().mean().item(),
                    m.bias.abs().mean().item() if m.bias is not None else None
                )
        
        # Sanity checks – run once
        if not hasattr(self, "_init_sanity_done"):
            trainable = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.network.parameters())
            print(f"DEBUG: Params: {trainable:,} trainable / {total:,} total")

            # Assert LoRA params are on CUDA during training
            if self.device.type == "cuda":
                for n, p in self.network.named_parameters():
                    if "lora_" in n:
                        assert p.is_cuda, f"LoRA param not on CUDA: {n}"

            # ---- Forward-pass sanity ----
            # Use randn not zeros — zeros through InstanceNorm gives degenerate output.
            # Also check argmax distribution: if the network is working correctly the
            # predicted class distribution should be non-uniform (not all one class).
            self.network.eval()
            with torch.no_grad():
                ps = self.configuration_manager.patch_size
                torch.manual_seed(42)
                dummy = torch.randn(1, self.num_input_channels, *ps, device=self.device)
                try:
                    out = self.network(dummy)
                    # Deep supervision returns a list — index 0 is the finest scale output
                    # (highest res, used for final prediction)
                    if isinstance(out, (list, tuple)):
                        print(f"DEBUG FORWARD SANITY: deep_supervision outputs={len(out)}, shapes={[list(o.shape) for o in out]}")
                        out0 = out[0]
                    else:
                        out0 = out
                    std = out0.std().item()
                    has_nan = torch.isnan(out0).any().item()
                    # Check predicted class distribution
                    preds = out0.argmax(dim=1)
                    unique_classes = preds.unique().numel()
                    dominant_class = preds.flatten().mode().values.item()
                    dominant_frac = (preds == dominant_class).float().mean().item()
                    print(f"DEBUG FORWARD SANITY: std={std:.4e}  nan={has_nan}  shape={list(out0.shape)}")
                    print(f"DEBUG FORWARD SANITY: predicted {unique_classes} unique classes, dominant={dominant_class} ({dominant_frac*100:.1f}% of voxels)")
                    if std < 1e-6 or has_nan:
                        print("WARNING: output is degenerate!")
                    elif dominant_frac > 0.99:
                        print("WARNING: network predicts almost entirely one class — weights may not be loaded correctly")
                    else:
                        print("DEBUG FORWARD SANITY: OK")
                except Exception as e:
                    print(f"WARNING: Forward pass sanity check failed: {e}")
            self.network.train()
            # ---- end forward sanity ----

            # ---- Check lora_B is actually zero and norm params match base ----
            lora_b_max = max(p.abs().max().item() for n, p in self.network.named_parameters() if 'lora_B' in n)
            lora_a_max = max(p.abs().max().item() for n, p in self.network.named_parameters() if 'lora_A' in n)
            print(f"DEBUG LORA INIT: lora_B max={lora_b_max:.4e}  lora_A max={lora_a_max:.4e}")
            if lora_b_max > 1e-8:
                print("WARNING: lora_B is NOT zero — weight loading is corrupting it!")

            # Compare every matching key between LoRA net and base checkpoint
            # to find exactly which weights differ
            base_path_check = self.weight_paths.get(self.configuration_name)
            if base_path_check and os.path.exists(base_path_check):
                base_ckpt_check = torch.load(base_path_check, map_location='cpu')
                base_w = base_ckpt_check.get('network_weights', base_ckpt_check.get('state_dict', base_ckpt_check))
                lora_sd = {n: p.detach().cpu() for n, p in self.network.named_parameters()}
                lora_sd.update({n: b.detach().cpu() for n, b in self.network.named_buffers()})
                mismatches = []
                for key, base_p in base_w.items():
                    if key in lora_sd and base_p.shape == lora_sd[key].shape:
                        diff = (base_p - lora_sd[key]).abs().max().item()
                        if diff > 1e-5:
                            mismatches.append((key, diff))
                mismatches.sort(key=lambda x: -x[1])
                print(f"DEBUG WEIGHT DIFF: {len(mismatches)} keys differ between base and LoRA net")
                for k, d in mismatches[:10]:
                    print(f"  {k}: max_diff={d:.4e}")
                del base_ckpt_check, base_w
            # ---- end checks ----
            # Load the base checkpoint, build a plain network, run same input through both.
            # If outputs differ, the LoRA injection is changing the forward pass.
            try:
                base_path = self.weight_paths.get(self.configuration_name)
                if base_path and os.path.exists(base_path):
                    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
                    base_ckpt = torch.load(base_path, map_location='cpu')
                    base_weights = base_ckpt.get('network_weights', base_ckpt.get('state_dict', base_ckpt))

                    base_net = get_network_from_plans(
                        self.configuration_manager.network_arch_class_name,
                        self.configuration_manager.network_arch_init_kwargs,
                        self.configuration_manager.network_arch_init_kwargs_req_import,
                        self.num_input_channels,
                        self.label_manager.num_segmentation_heads,
                        allow_init=True,
                        deep_supervision=self.enable_deep_supervision
                    ).to(self.device)
                    base_net.load_state_dict(base_weights, strict=True)
                    base_net.eval()

                    self.network.eval()
                    torch.manual_seed(42)
                    dummy = torch.randn(1, self.num_input_channels,
                                       *self.configuration_manager.patch_size,
                                       device=self.device)
                    with torch.no_grad():
                        lora_out = self.network(dummy)
                        base_out = base_net(dummy)

                    lora0 = lora_out[0] if isinstance(lora_out, (list, tuple)) else lora_out
                    base0 = base_out[0] if isinstance(base_out, (list, tuple)) else base_out
                    diff = (lora0 - base0).abs()
                    print(f"DEBUG EQUIV: max_diff={diff.max().item():.4e}  mean_diff={diff.mean().item():.4e}")
                    if diff.max().item() < 1e-4:
                        print("DEBUG EQUIV: PASS — LoRA net is identical to base at init")
                    else:
                        print("DEBUG EQUIV: FAIL — LoRA net differs from base at init!")
                        # Find which output scale differs most
                        if isinstance(lora_out, (list, tuple)):
                            for i, (lo, bo) in enumerate(zip(lora_out, base_out)):
                                d = (lo - bo).abs()
                                print(f"  scale {i}: shape={list(lo.shape)} max_diff={d.max().item():.4e}")
                    self.network.train()
                    del base_net, base_weights, base_ckpt
            except Exception as e:
                print(f"DEBUG EQUIV: failed with {e}")
            # ---- end equivalence check ----

            self._init_sanity_done = True
    
    

    # this is the heart, it gets called in training an inference, do not change fundamentally  
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                dataset_json: dict,
                                configuration_manager: ConfigurationManager,
                                num_input_channels: int,
                                num_output_channels: int,
                                enable_deep_supervision: bool = True) -> nn.Module:
        """
        Refactored to match nnU-Net v2.5+ signature while injecting LoRA.
        """
        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

        # 1. BUILD BASE NETWORK
        # We use the managers directly as intended by the base class
        network = get_network_from_plans(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision
        )

        # 2. IDENTIFY CONTEXT (Cascade vs Standard)
        # The configuration_manager now explicitly tracks this
        is_cascade = configuration_manager.previous_stage_name is not None
        
        # 3. APPLY LORA & FREEZE BACKBONE
        # Lock all base weights first
        for p in network.parameters():
            p.requires_grad = False

        # Logic: skip 0 for cascade, 1 for standard (matches your pediatric adaptation strategy)
        skip = 0 if is_cascade else 1
        
        # Static injection ensures it works during inference/predictor initialization
        nnUNetTrainerUniversalAdapter._inject_lora_static(
            network, 
            "cascade" if is_cascade else "standard",
            stages_to_skip=skip,
            adapter_rank=32,
            adapter_alpha=64,
            adapter_dropout=0.3
        )

        # 4. SELECTIVE UNFREEZING
        # Unfreeze Norms: crucial for domain shifts/pediatric data
        for m in network.modules():
            if isinstance(m, (nn.InstanceNorm3d, nn.GroupNorm, nn.BatchNorm3d)):
                for p in m.parameters():
                    p.requires_grad = True

        # Unfreeze Segmentation Heads (always needed for fine-tuning)
        if hasattr(network.decoder, 'seg_layers'):
            for m in network.decoder.seg_layers:
                for p in m.parameters():
                    p.requires_grad = True

        return network
    
    @staticmethod
    def _inject_lora_static(network, configuration_name, stages_to_skip=2, adapter_rank=32, adapter_alpha=64, adapter_dropout=0.3):
        """Static version of _inject_lora for use in build_network_architecture"""
        print(f"DEBUG: Injecting LoRA with configuration: {configuration_name}")
        is_cascade = "cascade" == configuration_name

        def replace(m):
            nnUNetTrainerUniversalAdapter._replace_convs_static(m, adapter_rank, adapter_alpha, adapter_dropout)

        # Encoder stages — respect skip
        for i, stage in enumerate(network.encoder.stages):
            if not is_cascade and i < stages_to_skip:
                continue
            replace(stage)

        # Decoder — must be handled surgically because network.decoder contains
        # a full decoder.encoder (mirror of the main encoder) whose early stages
        # must also be skipped, plus the actual decoder.stages and transpconvs.
        decoder = network.decoder

        # decoder.encoder mirrors the main encoder — apply same skip rule
        if hasattr(decoder, 'encoder'):
            for i, stage in enumerate(decoder.encoder.stages):
                if not is_cascade and i < stages_to_skip:
                    continue
                replace(stage)

        # decoder.stages are the actual upsampling conv blocks — always adapt
        if hasattr(decoder, 'stages'):
            for stage in decoder.stages:
                replace(stage)

        # transpconvs are 1x1x1 / strided convs for upsampling — kernel_size check
        # inside _replace_convs_static will skip them if kernel < 2, so safe to pass
        if hasattr(decoder, 'transpconvs'):
            for tc in decoder.transpconvs:
                replace(tc)

    @staticmethod
    def _replace_convs_static(module, adapter_rank, adapter_alpha, adapter_dropout):
        """
        Replace Conv3d children with LoRAConv3d.

        nnUNet's ConvDropoutNormReLU stores the same Conv3d under BOTH
        `.conv` (a direct attribute alias) and `.all_modules[0]` (inside the
        Sequential that the forward() actually iterates).  We must only wrap
        it once, at the all_modules[0] location.

        Rule: skip any child whose attribute name is exactly "conv" — that is
        always the alias in nnUNet and is never the path used by forward().
        The real replacement happens when we recurse into `all_modules`.
        """
        for name, child in list(module.named_children()):
            # Skip the .conv alias — it duplicates all_modules[0]
            if name == 'conv' and isinstance(child, nn.Conv3d):
                continue
            # Already wrapped — skip
            if isinstance(child, LoRAConv3d):
                continue
            if isinstance(child, nn.Conv3d) and child.kernel_size[0] > 1:
                setattr(module, name, LoRAConv3d(
                    child,
                    r=adapter_rank,
                    alpha=adapter_alpha,
                    dropout=adapter_dropout
                ))
            else:
                nnUNetTrainerUniversalAdapter._replace_convs_static(child, adapter_rank, adapter_alpha, adapter_dropout)

    def _inject_lora(self, network):
        """Instance method version for reference"""
        is_cascade = "cascade" in self.configuration_name
    
        # Encoder
        for i, stage in enumerate(network.encoder.stages):
            if not is_cascade and i < self.stages_to_skip:
                continue
            self._replace_convs(stage)

        # Decoder (always adapt)
        self._replace_convs(network.decoder)


    def _replace_convs(self, module):
        for name, child in list(module.named_children()):
            if name == 'conv' and isinstance(child, nn.Conv3d):
                continue
            if isinstance(child, LoRAConv3d):
                continue
            if isinstance(child, nn.Conv3d) and child.kernel_size[0] > 1:
                setattr(module, name, LoRAConv3d(
                    child,
                    r=self.adapter_rank,
                    alpha=self.adapter_alpha, dropout=self.adapter_dropout
                ))
            else:
                self._replace_convs(child)


    def _smart_load_weights(self):
        path = self.weight_paths.get(self.configuration_name)
        print(path)
        if not path or not os.path.exists(path): 
            self.print_to_log_file(f"WARNING: No pretrained weights found at {path}")
            return

        checkpoint = torch.load(path, map_location='cpu')
        old_state = checkpoint.get('network_weights', checkpoint.get('state_dict', checkpoint))

        print("Old network:")
        print(old_state.keys())

        # CRITICAL: use named_parameters() so we write directly into the live model tensors.
        # state_dict() returns copies — copying into those copies is a no-op on the actual model.
        # We also need named_buffers() to cover running_mean/var in BN layers.
        param_dict: dict[str, torch.Tensor] = {
            **{name: p for name, p in self.network.named_parameters()},
            **{name: b for name, b in self.network.named_buffers()},
        }

        def old_key(new_key: str) -> str:
            """
            Try to map a new-model key back to a pretrained-checkpoint key.

            nnUNet wraps Conv3d inside ConvDropoutNormReLU whose named_children are
            stored as  all_modules  (a nn.Sequential).  The key in a plain
            nnUNetTrainer checkpoint therefore looks like:
                encoder.stages.0.0.convs.0.all_modules.0.weight
            After LoRA injection the Conv3d at  all_modules.0  is replaced by a
            LoRAConv3d whose weight/bias attributes keep the same name, so the key
            stays identical.  No stripping should be needed — but we keep several
            fallback patterns just in case the source checkpoint was saved from a
            trainer that used a different wrapper depth.
            """
            candidates = [
                new_key,                                            # identity — always try first
                new_key.replace('.conv.weight', '.weight')
                       .replace('.conv.bias',   '.bias'),
                new_key.replace('.all_modules.0.', '.'),           # flatten one wrapper level
                new_key.replace('.all_modules.0.', '.')
                       .replace('.conv.weight', '.weight')
                       .replace('.conv.bias',   '.bias'),
            ]
            for c in candidates:
                if c in old_state:
                    return c
            return new_key   # not found — return as-is so [MISSING] is logged


        # ---- KEY FORMAT DIAGNOSTIC ----
        old_keys_sample = list(old_state.keys())[:15]
        new_keys_sample = list(param_dict.keys())[:15]
        self.print_to_log_file(f"[KEY DIAG] First 15 OLD keys: {old_keys_sample}")
        self.print_to_log_file(f"[KEY DIAG] First 15 NEW keys: {new_keys_sample}")
        # Count how many new keys (non-lora, non-head) actually exist in old_state
        n_found = sum(
            1 for k in param_dict
            if not any(x in k for x in ["lora_A","lora_B","lora_dropout",".all_modules.0.norm"])
            and 'seg_layers' not in k and 'output' not in k
            and old_key(k) in old_state
        )
        n_total = sum(
            1 for k in param_dict
            if not any(x in k for x in ["lora_A","lora_B","lora_dropout",".all_modules.0.norm"])
            and 'seg_layers' not in k and 'output' not in k
        )
        self.print_to_log_file(f"[KEY DIAG] {n_found}/{n_total} new base keys found directly in old_state")
        # ---- END DIAGNOSTIC ----

        self.print_to_log_file("--- WEIGHT SURGERY REPORT ---")

        matched = 0
        skipped_lora = 0
        skipped_head = 0
        missing = 0
        surgery = 0

        for key, live_p in param_dict.items():
            # --- Skip LoRA-only params: lora_A, lora_B, and the LoRAConv3d's
            #     internal bottleneck norm (.all_modules.0.norm.*).
            #     Do NOT skip the real ConvDropoutNormReLU norms (.norm.weight/bias)
            #     — those need to be loaded from the checkpoint. ---
            if any(x in key for x in ["lora_A", "lora_B", "lora_dropout", ".all_modules.0.norm"]):
                skipped_lora += 1
                continue

            # --- Seg head: load if shapes match, skip only on actual class-count mismatch ---
            if 'seg_layers' in key or 'output' in key:
                src_key_head = old_key(key)
                if src_key_head in old_state:
                    old_p = old_state[src_key_head].to(live_p.device)
                    if old_p.shape == live_p.shape:
                        with torch.no_grad():
                            live_p.copy_(old_p)
                        matched += 1
                    else:
                        # Splicing: Copy existing classes, leave new classes as-is (or zero)
                        with torch.no_grad():
                            # For Conv3d weights [OutC, InC, K, K, K] or Bias [OutC]
                            min_classes = min(old_p.shape[0], live_p.shape[0])
                            live_p[:min_classes].copy_(old_p[:min_classes])
                            
                        self.print_to_log_file(f"[HEAD SURGERY] Spliced {min_classes} classes into {key}")
                        surgery += 1
                continue

            src_key = old_key(key)
            if src_key not in old_state:
                self.print_to_log_file(f"[MISSING] {key} (looked up as '{src_key}') not in pretrained weights")
                missing += 1
                continue

            old_p = old_state[src_key].to(live_p.device)

            with torch.no_grad():
                if old_p.shape == live_p.shape:
                    live_p.copy_(old_p)
                    matched += 1
                else:
                    # Shape-mismatch surgery: copy the overlapping slice, zero the rest
                    slices = tuple(slice(0, min(d_o, d_n)) for d_o, d_n in zip(old_p.shape, live_p.shape))
                    live_p.zero_()
                    live_p[slices].copy_(old_p[slices])
                    self.print_to_log_file(
                        f"[SURGERY] {key}: Old {list(old_p.shape)} -> New {list(live_p.shape)}"
                    )
                    surgery += 1

        self.print_to_log_file(
            f"--- SURGERY SUMMARY: {matched} matched | {surgery} surgery | "
            f"{missing} missing | {skipped_head} head-skipped | {skipped_lora} lora-skipped ---"
        )
        self.print_to_log_file("--- END OF SURGERY REPORT ---")

        # ---- Hard sanity: spot-check 3 base weight tensors directly on the live model ----
        self.print_to_log_file("--- POST-SURGERY SPOT CHECK ---")
        checked = 0
        for key, live_p in param_dict.items():
            if any(x in key for x in ["lora_A","lora_B","lora_dropout",".all_modules.0.norm"]):
                continue
            if 'seg_layers' in key or 'output' in key:
                continue
            src_key = old_key(key)
            if src_key not in old_state:
                continue
            old_p = old_state[src_key]
            if old_p.shape != live_p.shape:
                continue
            diff = (old_p.cpu() - live_p.detach().cpu()).abs().mean().item()
            self.print_to_log_file(
                f"[SPOT] {key}: live_mean={live_p.detach().cpu().abs().mean():.5e}  "
                f"old_mean={old_p.abs().mean():.5e}  diff={diff:.5e}  "
                f"{'OK' if diff < 1e-6 else '*** MISMATCH ***'}"
            )
            checked += 1
            if checked >= 5:
                break
        self.print_to_log_file("--- END SPOT CHECK ---")


        # ---- Integrity verification (reads back from the live model) ----
        self.print_to_log_file("--- Weight LOADING CHECK ---")
        verify_matches = 0
        verify_failures = 0
        for key, live_p in param_dict.items():
            if any(x in key for x in ["lora_A", "lora_B", "lora_dropout", ".all_modules.0.norm"]):
                continue
            if 'seg_layers' in key or 'output' in key:
                continue

            src_key = old_key(key)
            if src_key not in old_state:
                continue

            old_p = old_state[src_key]
            if old_p.shape == live_p.shape:
                diff = (old_p.cpu() - live_p.cpu()).abs().mean().item()
                if diff < 1e-7:
                    verify_matches += 1
                else:
                    self.print_to_log_file(f"[DIFF FOUND] {key}: Mean Diff = {diff:.6e}")
                    verify_failures += 1

        self.print_to_log_file(
            f"Verification: {verify_matches} identical layers, {verify_failures} different layers."
        )





            


    def _debug_weight_integrity(self):
        print("DEBUG weight integrity check:")

        for name, p in self.network.named_parameters():
            print(
                f"{name}: "
                f"mean={p.data.mean():.4e}, "
                f"std={p.data.std():.4e}, "
                f"max={p.data.abs().max():.4e}"
            )
                

    def on_train_start(self):
        """Starts a new MLflow run for this specific fold execution"""
        super().on_train_start()
        if mlflow.active_run(): 
            mlflow.end_run()
        # Only log from the main process (rank 0) to avoid duplicate runs in DDP
        if self.local_rank == 0:
            run_name = f"Fold_{self.fold}_{self.configuration_name}"
            mlflow.start_run(run_name=run_name, log_system_metrics= True)  #track system metrics, can be helpful to see if we bottleneck, low gpu usage in cascade is normal, as we cpu bottleneck 
            
            # Log relevant setup parameters
            mlflow.log_params({
                "fold": self.fold,
                "config": self.configuration_name,
                "lora_r": self.adapter_rank,
                "lora_alpha": self.adapter_alpha,
                "lr": self.initial_lr,
                "epochs": self.num_epochs,
                "warmup" : self.warmup,
                "head_weight" : self.head_weight
            })

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
    


    # does not work for this style of data
    def log_dataset_metadata(self, fold):
        import mlflow.data  
        from mlflow.data.dataset_source import FileSystemDatasetSource

        if self.local_rank == 0:
            # Define the source of your preprocessed data
            dataset_path = r"D:\nnUNet_preprocessed\Dataset118_Scolsmall"
            source = FileSystemDatasetSource(dataset_path)
            
            # Create a generic dataset object
            # You can name it by fold to distinguish versions in MLflow
            dataset = mlflow.data.from_pandas( # Mocking with from_pandas for metadata structure
                pd.DataFrame({"path": [dataset_path]}), 
                source=source, 
                name=f"Scoliosis_Fold_{self.fold}",
                targets="segmentation_labels"
            )
            
            # Register the dataset to the current run
            # Using context="training" helps filter in the UI
            mlflow.log_input(dataset, context="training")

    def on_train_epoch_start(self):
        self.network.train()
        self.lr_scheduler.step()
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

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


class nnUNetSequentialLR(SequentialLR):
    """
    Custom wrapper to make SequentialLR compatible with nnU-Net.
    nnU-Net calls .step(epoch), but SequentialLR only wants .step().
    """
    def step(self, epoch=None):
        # We ignore the 'epoch' argument passed by nnU-Net
        super().step()


# -------------------------------------------------------------------------
# Custom Predictor for LoRA Trainer
# -------------------------------------------------------------------------
def create_lora_predictor(base_predictor):
    from batchgenerators.utilities.file_and_folder_operations import load_json, join
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
    import nnunetv2
    import torch
    import types

    def initialize_from_trained_model_folder_lora(self, model_training_output_dir: str,
                                                  use_folds: Union[Tuple[Union[int, str]], None],
                                                  checkpoint_name: str = 'checkpoint_final.pth'):
        # 1. AUTO-DETECT METADATA
        if use_folds is None:
            use_folds = self.auto_detect_available_folds(model_training_output_dir, checkpoint_name)
        
        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        # 2. LOAD PARAMETERS & IDENTIFY TRAINER
        parameters = []
        for i, f in enumerate(use_folds):
            f = int(f) if f != 'all' else f
            checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                    map_location=torch.device('cpu'), weights_only=False)
            if i == 0:
                trainer_name = checkpoint['trainer_name']
                configuration_name = checkpoint['init_args']['configuration']
                inference_allowed_mirroring_axes = checkpoint.get('inference_allowed_mirroring_axes')
            parameters.append(checkpoint['network_weights'])
        
        # 3. UNIFIED ARCHITECTURE CALL
        config_manager = plans_manager.get_configuration(configuration_name)
        trainer_class = recursive_find_python_class(
            join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
            trainer_name, 'nnunetv2.training.nnUNetTrainer'
        )

        num_input_channels = determine_num_input_channels(plans_manager, config_manager, dataset_json)
        num_output_channels = plans_manager.get_label_manager(dataset_json).num_segmentation_heads

        # This is the unified call
        network = trainer_class.build_network_architecture(
            plans_manager,
            dataset_json,
            config_manager,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision=False  # Always False for inference
        )
        
        # 4. FINALIZE PREDICTOR STATE
        self.plans_manager = plans_manager
        self.configuration_manager = config_manager
        self.list_of_parameters = parameters
        self.network = network
        self.network.load_state_dict(parameters[0]) # Initial fold

        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        
        # Respect global compile settings
        if ('nnUNet_compile' in os.environ.keys()) and (os.environ['nnUNet_compile'].lower() in ('true', '1', 't')):
            self.network = torch.compile(self.network)

    # Override the method
    base_predictor.initialize_from_trained_model_folder = types.MethodType(
        initialize_from_trained_model_folder_lora, base_predictor
    )
    return base_predictor


#allows misatch between old and new size 

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler

class nnUNetTrainerNewHead(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("Scoliosis_FFT_V3_lowres")
        self.num_epochs = 200
        self.weight_factor = 10
        self.initial_lr = 1e-4
        self.warmup = 0.1


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

            print(self.network.state_dict().keys())
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)
            
            fine_tune = True 

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
                    {'params': head_params, 'lr': self.initial_lr * self.weight_factor}
                ]


                
                self.optimizer = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)

                # 3. Setup Sequential Scheduler: 10%-epoch Warmup + Cosine Decay 
                total_epochs = self.num_epochs
                warmup_epochs = self.warmup * total_epochs
                warmup_epochs = int(warmup_epochs)
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
                
                self.lr_scheduler = nnUNetSequentialLR(
                    self.optimizer, 
                    schedulers=[warmup_scheduler, cosine_scheduler], 
                    milestones=[warmup_epochs]
                )
            else: self.optimizer, self.lr_scheduler = self.configure_optimizers()


            self.loss = self._build_loss()

            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    
    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()

        checkpoint = torch.load(filename_or_checkpoint, map_location=self.device) if isinstance(filename_or_checkpoint, str) else filename_or_checkpoint
        old_state_dict = checkpoint['network_weights']
        current_model_dict = self.network.state_dict()
        final_state_dict = {}

        for k, curr_v in current_model_dict.items():
            # Clean DDP prefix if necessary
            old_key = k if k in old_state_dict else f'module.{k}'
            
            if old_key in old_state_dict:
                old_v = old_state_dict[old_key]
                
                if old_v.shape == curr_v.shape:
                    final_state_dict[k] = old_v
                else:
                    # SURGERY: Copy overlapping weights (Fixes transpconvs and seg_layers)
                    print(f"[SURGERY] Splicing {k}: {list(old_v.shape)} -> {list(curr_v.shape)}")
                    new_param = curr_v.clone()
                    # Create slices for all dimensions (C, Cin, H, W, D)
                    slices = tuple(slice(0, min(o, c)) for o, c in zip(old_v.shape, curr_v.shape))
                    new_param.zero_()
                    new_param[slices].copy_(old_v[slices])
                    final_state_dict[k] = new_param
            else:
                print(f"[MISSING] {k} not found in checkpoint.")
        print(f"Final network structure {final_state_dict.keys()}")

        for k, v in final_state_dict.items():
            # We focus on the layers most likely changed by Cascade or your custom head
            if any(x in k for x in ['encoder.stages.0.0', 'seg_layers', 'output']):
                status = "SPLICED" if k in locals().get('spliced_keys', []) else "MATCHED"
                print(f"[{status}] {k: <40} | Shape: {list(v.shape)}")
        if hasattr(self.network, 'decoder'):
            print(f"SUCCESS: Decoder found at {type(self.network.decoder)}")
        else:
            print("CRITICAL: Decoder NOT FOUND on self.network after initialization!")

        self.network.load_state_dict(final_state_dict, strict=False)
        # -------------------------------

        # Optional: Reset optimizer if you don't want to carry over the learning state 
        # from the previous class count/momentum
       

       # --- transfer learning reset ---
        self.current_epoch = 0
        self._best_ema = None

        from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
        self.logger = nnUNetLogger()





    def configure_optimizers(self):
   
        seg_layer_params = []
        base_params = []
        
        # Identify which parameters belong to the head
        for name, param in self.network.named_parameters():
            if 'seg_layers' in name:
                seg_layer_params.append(param)
            else:
                base_params.append(param)

        # 2. Create parameter groups with different LRs
        # Example: Head gets 1e-2, Body gets 1e-3
        
        params = [
            {'params': base_params, 'lr': self.initial_lr * self.weight_factor }, 
            {'params': seg_layer_params, 'lr': self.initial_lr}
        ]

        # 3. Initialize Optimizer with the group list
        optimizer = torch.optim.SGD(params, 
                                    weight_decay=self.weight_decay,
                                    momentum=0.99, 
                                    nesterov=True)
        
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler  
    
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
                "lr_weight_factor" : self.weight_factor,
                "epochs": self.num_epochs, 
                "warmup_rate" : self.warmup
            })

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

    def maybe_load_checkpoint(nnunet_trainer: nnUNetTrainer, continue_training: bool, validation_only: bool,
                          pretrained_weights_file: str = None):
        if continue_training and pretrained_weights_file is not None:
            raise RuntimeError('Cannot both continue a training AND load pretrained weights. Pretrained weights can only '
                            'be used at the beginning of the training.')
        if continue_training:
            expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_final.pth')
            if not isfile(expected_checkpoint_file):
                expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_latest.pth')
            # special case where --c is used to run a previously aborted validation
            if not isfile(expected_checkpoint_file):
                expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_best.pth')
            if not isfile(expected_checkpoint_file):
                print(f"WARNING: Cannot continue training because there seems to be no checkpoint available to "
                                f"continue from. Starting a new training...")
                expected_checkpoint_file = None
        elif validation_only:
            expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_final.pth')
            if not isfile(expected_checkpoint_file):
                raise RuntimeError(f"Cannot run validation because the training is not finished yet!")
        else:
            if pretrained_weights_file is not None:
                if not nnunet_trainer.was_initialized:
                    nnunet_trainer.initialize()

                print(f"[INFO] Initializing network from checkpoint (partial load): "
                    f"{pretrained_weights_file}")

                # Route through trainer logic instead of strict loader
                nnunet_trainer.load_checkpoint(pretrained_weights_file)

            expected_checkpoint_file = None

        if expected_checkpoint_file is not None:
            nnunet_trainer.load_checkpoint(expected_checkpoint_file)


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
                noise_variance=(0, 0.05),  # lowered here so it handles the stronger blur and gamma
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.1
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
                gamma=BGContrast((0.5, 2.0)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.5, 2.0)),
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




if __name__ == '__main__':
    ########################## predict a bunch of files
    from nnunetv2.paths import nnUNet_results, nnUNet_raw
    from batchgenerators.utilities.file_and_folder_operations import load_json, join
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
    import nnunetv2
    import torch
    import types

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    predictor = create_lora_predictor(predictor)

    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, 'Dataset118_Scolsmall/nnUNetTrainerUniversalAdapter__nnUNetPlans__3d_lowres'),
        use_folds=(0,),
        checkpoint_name='checkpoint_final.pth',
    )

    print("\n===== MANUAL FORWARD TEST =====")

    net = predictor.network
    net.eval()

    ps = predictor.configuration_manager.patch_size
    print("patch size:", ps)

    def first_conv_in_channels(module):
        for m in module.modules():
            if isinstance(m, torch.nn.Conv3d) or m.__class__.__name__ == "LoRAConv3d":
                return m.in_channels
        raise RuntimeError("No Conv3d found in module")

    c = first_conv_in_channels(net.encoder.stages[0])
    print("channels:", c)

    x = torch.randn(1, c, *ps, device=predictor.device)

    with torch.no_grad():
        y = net(x)

    if isinstance(y, (list, tuple)):
        print("output is list:")
        for i, t in enumerate(y):
            print(i, t.shape)
    else:
        print("output:", y.shape)

    print("===== shallow MANUAL FORWARD OK =====\n")

    print("\n===== FULL DEEP PROPAGATION TEST =====")

    net = predictor.network
    net.eval()

    ps = predictor.configuration_manager.patch_size

    def first_conv_in_channels(module):
        for m in module.modules():
            if isinstance(m, torch.nn.Conv3d) or m.__class__.__name__ == "LoRAConv3d":
                return m.in_channels
        raise RuntimeError("No Conv3d found")

    # build dummy input
    c = first_conv_in_channels(net.encoder.stages[0])
    x = torch.randn(1, c, *ps, device=predictor.device)

    print("Input:", x.shape)

    # -------- Encoder --------
    skips = []
    for i, stage in enumerate(net.encoder.stages):
        print(f"\n[ENCODER stage {i}] input :", x.shape)

        x = stage(x)
        skips.append(x)

        print(f"[ENCODER stage {i}] output:", x.shape)

    # x is now bottleneck output
    print("\n[BOTTLENECK OUTPUT]:", x.shape)

        # -------- Decoder --------
        # remove deepest feature (bottleneck has no skip)
    skips = skips[:-1]

    for i in range(len(net.decoder.stages)):
        up = net.decoder.transpconvs[i]
        stage = net.decoder.stages[i]
        skip = skips[-(i+1)]

        print(f"\n[DECODER stage {i}] before up:", x.shape)
        x = up(x)
        print(f"[DECODER stage {i}] after up :", x.shape, "skip:", skip.shape)

        x = torch.cat((x, skip), dim=1)
        print(f"[DECODER stage {i}] after cat:", x.shape)

        x = stage(x)
        print(f"[DECODER stage {i}] after conv:", x.shape)

        # -------- Segmentation head --------
    print("\n[SEG HEAD]")
    out = net.decoder.seg_layers[-1](x)
    print("Segmentation output:", out.shape)

    print("\n===== FULL DEEP PROPAGATION OK =====\n")