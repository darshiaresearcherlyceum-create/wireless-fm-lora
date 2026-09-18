import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class WirelessTaskHeads(nn.Module):
    """
    Dedicated output prediction heads for 5 physical layer tasks:
      Head 0: Channel Estimation [B, N_sub, 1024] (Delay-domain Denoising + Neural Residual)
      Head 1: MIMO Detection [B, N_sub, 16]
      Head 2: Multi-User Precoding [B, N_sub, 1024] (Structured Differentiable RZF + Neural Beamforming)
      Head 3: Channel Decoding [B, N_sub, 16]
      Head 4: User Localization [B, 24] (User-Decomposed Spatial Features + Scene Fusion)
    """
    def __init__(self, hidden_dim: int = 768, num_subcarriers: int = 64, semantic_coupling: bool = False, scene_dim: int = 256):
        super().__init__()
        self.semantic_coupling = semantic_coupling
        self.scene_dim = scene_dim
        self.hidden_dim = hidden_dim
        self.num_subcarriers = num_subcarriers
        
        # Task 0: Channel Estimation (Delay-domain multipath filtering + transformer-conditioned refinement)
        # Soft learnable delay gate: Taps 0..15 strongly preferred (+3.0), taps 16..63 softly suppressed (-2.5) but trainable
        init_gate = torch.full((num_subcarriers,), -2.5)
        init_gate[:16] = 3.0 # In-band delay taps pass through
        self.delay_gate = nn.Parameter(init_gate)
        
        # Environment-adaptive delay gating network: maps scene embedding z_s [B, scene_dim] to delay tap offsets [B, num_subcarriers]
        self.scene_delay_adapter = nn.Sequential(
            nn.Linear(scene_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_subcarriers)
        )
        nn.init.zeros_(self.scene_delay_adapter[-1].weight)
        nn.init.zeros_(self.scene_delay_adapter[-1].bias)
        
        # SNR-aware blending mechanism alpha(SNR) in [0, 1]
        # Prevents high-SNR error floor by adaptively bypassing lossy delay filtering when noise is small
        self.snr_gate_ce = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )
        # Initialize: negative slope so low SNR has high alpha (denoising active), high SNR has low alpha (raw LS bypass)
        nn.init.constant_(self.snr_gate_ce[0].weight, -1.0)
        nn.init.constant_(self.snr_gate_ce[0].bias, 0.0)
        nn.init.constant_(self.snr_gate_ce[2].weight, 1.0)
        nn.init.constant_(self.snr_gate_ce[2].bias, 0.5)
        
        # Neural CE residual projection head
        self.head_ce = nn.Sequential(
            nn.Linear(hidden_dim + 1024, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 1024)
        )
        # Trainable scale for neural CE residual correction over LS (initialized at 1.0)
        self.ce_residual_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        
        # Task 1: MIMO Detection (Channel-aware Physical Features + Dedicated Detection Head)
        # Features: Received signal Y_d (16), Matched Filter z_mf (16), Linear MMSE z_mmse (16), Effective Diag (16), Gram matrix G (128) -> Total 192
        self.head_det = nn.Sequential(
            nn.Linear(hidden_dim + 192, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 16)
        )
        
        # Task 2: MU-MIMO Precoding (Pure Neural Precoding Head predicting complex W_NN in C^{64 x 8})
        self.head_prec = nn.Sequential(
            nn.Linear(hidden_dim + 1024, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 1024)
        )
        
        # Task 3: Channel Decoding
        self.head_dec = nn.Linear(hidden_dim, 16)
        
        # Task 4: User 3D Localization (2D Spatial FFT Beamspace + Delay Profile + Relative Power)
        self.head_loc = nn.Sequential(
            nn.Linear(81, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3)
        )
            
    def forward(
        self,
        h_shared: torch.Tensor,
        inputs: Optional[torch.Tensor] = None,
        z_s: Optional[torch.Tensor] = None,
        snr: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        h_shared: [B, N_sub, hidden_dim]
        inputs: Optional [B, N_sub, 1040] raw multi-task input tensor
        z_s: Optional [B, scene_dim] scene representation vector from scene encoder
        snr: Optional [B] or [B, 1] SNR in dB
        """
        B_size, N_sub, _ = h_shared.shape
        device = h_shared.device
        dtype = h_shared.dtype
        
        # ==================== Task 0: Channel Estimation ====================
        if inputs is not None:
            raw_pilot = inputs[:, :, :1024] # [B, N_sub, 1024] - pristine LS observation Y_p
            # Transform to complex domain [B, N_sub, 8, 64]
            ch_real = raw_pilot[:, :, :512].reshape(B_size, N_sub, 8, 64)
            ch_imag = raw_pilot[:, :, 512:].reshape(B_size, N_sub, 8, 64)
            H_c = torch.complex(ch_real, ch_imag)
            
            # Delay-domain transform along subcarrier dimension (dim=1)
            H_delay = torch.fft.ifft(H_c, dim=1) # [B, N_sub, 8, 64]
            
            # Differentiable soft delay gate with environment-adaptive modulation
            if z_s is not None:
                g_env = self.scene_delay_adapter(z_s) # [B, N_sub]
                gate_logits = self.delay_gate.unsqueeze(0) + g_env # [B, N_sub]
                delay_mask = torch.sigmoid(gate_logits).view(B_size, N_sub, 1, 1)
            else:
                delay_mask = torch.sigmoid(self.delay_gate).view(1, N_sub, 1, 1)
                
            H_filtered_delay = H_delay * delay_mask
            H_denoised = torch.fft.fft(H_filtered_delay, dim=1)
            
            denoised_flat = torch.cat([
                H_denoised.real.reshape(B_size, N_sub, 512),
                H_denoised.imag.reshape(B_size, N_sub, 512)
            ], dim=-1) # [B, N_sub, 1024]
            
            # SNR-aware blending mechanism alpha(SNR) in [0, 1]
            if snr is not None:
                snr_val = snr.view(B_size, 1).float()
            else:
                # Unsupervised noise power estimate from tail delay taps (taps 32..63)
                tail_pwr = torch.mean(torch.abs(H_delay[:, 32:, :, :]) ** 2, dim=(1, 2, 3), keepdim=True).view(B_size, 1)
                snr_val = -10.0 * torch.log10(torch.clamp(tail_pwr, min=1e-5, max=10.0))
                
            snr_norm = (snr_val - 10.0) / 10.0 # Standardize around 10 dB
            alpha_ce = torch.sigmoid(self.snr_gate_ce(snr_norm)).unsqueeze(1) # [B, 1, 1]
            
            # Reference Least Squares baseline (pristine LS observation Y_p)
            ls_ce = raw_pilot
            
            # Physics-guided delay-domain residual prior
            delay_residual = alpha_ce * (denoised_flat - ls_ce)
            
            # Differentiable neural residual refinement from foundation-model representation
            ce_in = torch.cat([h_shared, denoised_flat], dim=-1)
            backbone_residual = self.head_ce(ce_in)
            
            # Unified differentiable neural residual correction over LS
            neural_residual = delay_residual + backbone_residual
            
            # final_CE = LS_CE + learnable_scale * neural_residual
            out_ce = ls_ce + self.ce_residual_scale * neural_residual
        else:
            dummy_pilot = torch.zeros(B_size, N_sub, 1024, device=device, dtype=dtype)
            ce_in = torch.cat([h_shared, dummy_pilot], dim=-1)
            neural_residual = self.head_ce(ce_in)
            out_ce = dummy_pilot + self.ce_residual_scale * neural_residual
            alpha_ce = torch.full((B_size, 1, 1), 0.5, device=device, dtype=dtype)
            
        # ==================== Task 1: MIMO Detection ====================
        if inputs is not None:
            # Extract received signals Y_d and channel matrix H
            yd_real = inputs[:, :, :8]
            yd_imag = inputs[:, :, 8:16]
            Y_c = torch.complex(yd_real, yd_imag).unsqueeze(-1) # [B, N_sub, 8, 1]
            
            ch_r = inputs[:, :, 16:528].reshape(B_size, N_sub, 8, 64)
            ch_i = inputs[:, :, 528:1040].reshape(B_size, N_sub, 8, 64)
            H_c = torch.complex(ch_r, ch_i) # [B, N_sub, 8, 64]
            
            # Gram correlation matrix G = H H^H [B, N_sub, 8, 8]
            G = torch.matmul(H_c, H_c.mH)
            G_real = G.real.reshape(B_size, N_sub, 64)
            G_imag = G.imag.reshape(B_size, N_sub, 64)
            
            # Compute Zero-Forcing beamforming weights and effective channel H_eff = H W_zf
            eye_K = torch.eye(8, dtype=H_c.dtype, device=device).unsqueeze(0).unsqueeze(0)
            X_pinv = torch.linalg.solve(G + 0.01 * eye_K, H_c) # [B, N_sub, 8, 64]
            V = X_pinv.mH # [B, N_sub, 64, 8]
            v_norm = torch.linalg.norm(V, dim=2, keepdim=True) + 1e-8
            W_zf = V / v_norm # [B, N_sub, 64, 8]
            
            H_eff = torch.matmul(H_c, W_zf) # [B, N_sub, 8, 8]
            G_eff = torch.matmul(H_eff.mH, H_eff) # [B, N_sub, 8, 8]
            
            # 1. Matched Filter Feature: z_mf = H_eff^H Y
            z_mf = torch.matmul(H_eff.mH, Y_c).squeeze(-1) # [B, N_sub, 8]
            z_mf_real = z_mf.real
            z_mf_imag = z_mf.imag
            
            # 2. Linear MMSE Equalization Feature: z_mmse = (G_eff + 0.05*I)^(-1) H_eff^H Y
            z_mmse = torch.linalg.solve(G_eff + 0.05 * eye_K, torch.matmul(H_eff.mH, Y_c)).squeeze(-1) # [B, N_sub, 8]
            z_mmse_real = z_mmse.real
            z_mmse_imag = z_mmse.imag
            
            # 3. Effective channel diagonal gains
            h_diag_real = torch.diagonal(H_eff.real, dim1=2, dim2=3) # [B, N_sub, 8]
            h_diag_imag = torch.diagonal(H_eff.imag, dim1=2, dim2=3) # [B, N_sub, 8]
            
            det_phys = torch.cat([
                yd_real, yd_imag,
                z_mf_real, z_mf_imag,
                z_mmse_real, z_mmse_imag,
                h_diag_real, h_diag_imag,
                G_real, G_imag
            ], dim=-1) # [B, N_sub, 192]
            
            det_in = torch.cat([h_shared, det_phys], dim=-1)
            delta_det = self.head_det(det_in) # [B, N_sub, 16]
            
            # Physics-guided residual prediction
            base_syms = torch.cat([z_mmse_real, z_mmse_imag], dim=-1) # [B, N_sub, 16]
            out_det = base_syms + delta_det # [B, N_sub, 16]
        else:
            dummy_phys = torch.zeros(B_size, N_sub, 192, device=device, dtype=dtype)
            det_in = torch.cat([h_shared, dummy_phys], dim=-1)
            out_det = self.head_det(det_in)
            
        # ==================== Task 2: MU-MIMO Precoding ====================
        if inputs is not None:
            h_in = inputs[:, :, :1024]
            prec_in = torch.cat([h_shared, h_in], dim=-1)
        else:
            dummy_pilot = torch.zeros(B_size, N_sub, 1024, device=device, dtype=dtype)
            prec_in = torch.cat([h_shared, dummy_pilot], dim=-1)
            
        # Pure neural precoder prediction from shared backbone & channel features
        W_pred_raw = self.head_prec(prec_in) # [B, N_sub, 1024]
        w_real = W_pred_raw[:, :, :512].reshape(B_size, N_sub, 64, 8)
        w_imag = W_pred_raw[:, :, 512:].reshape(B_size, N_sub, 64, 8)
        W_neural = torch.complex(w_real, w_imag) # [B, N_sub, 64, 8]
        
        # Per-column power normalization along BS antennas (dim=2): ||w_k||_2 = 1 => total power = K = 8
        w_col_norm = torch.linalg.norm(W_neural, dim=2, keepdim=True) + 1e-8
        W_norm = W_neural / w_col_norm
        
        out_prec = torch.cat([
            W_norm.real.reshape(B_size, N_sub, 512),
            W_norm.imag.reshape(B_size, N_sub, 512)
        ], dim=-1)
            
        # ==================== Task 3: Channel Decoding ====================
        out_dec = self.head_dec(h_shared) # [B, N_sub, 16]
        
        # ==================== Task 4: User 3D Localization ====================
        h_mean = h_shared.mean(dim=1) # [B, hidden_dim]
        h_std = h_shared.std(dim=1)   # [B, hidden_dim]
        if inputs is not None:
            # Extract user-specific physical channel AoA & Delay features
            ch_r = inputs[:, :, 16:528].reshape(B_size, N_sub, 8, 64)
            ch_i = inputs[:, :, 528:1040].reshape(B_size, N_sub, 8, 64)
            H_c = torch.complex(ch_r, ch_i) # [B, N_sub, 8, 64]
            
            # 1. 2D Angle-of-Arrival (AoA) Beamspace across 8x8 BS UPA antennas (dims 3, 4)
            H_2d = H_c.reshape(B_size, N_sub, 8, 8, 8)
            beamspace_2d = torch.abs(torch.fft.fft2(H_2d, dim=(3, 4))).mean(dim=1) # [B, 8, 8, 8]
            beam_flat = beamspace_2d.reshape(B_size, 8, 64)
            beam_norm = beam_flat / (beam_flat.norm(dim=-1, keepdim=True) + 1e-6)
            
            # 2. Delay (ToA) multipath profile across 64 subcarriers (dim=1)
            delay_spec = torch.abs(torch.fft.ifft(H_c, dim=1)).mean(dim=3)[:, :16, :] # [B, 16, 8]
            delay_norm = delay_spec.permute(0, 2, 1) / (delay_spec.permute(0, 2, 1).norm(dim=-1, keepdim=True) + 1e-6) # [B, 8, 16]
            
            # 3. Channel power per user
            p_k = (ch_r**2 + ch_i**2).sum(dim=-1).mean(dim=1, keepdim=True).permute(0, 2, 1) # [B, 8, 1]
            p_norm = torch.clamp((p_k + 1e-6)**(-0.25) * 0.2, 0.0, 1.0)
            
            u_feat = torch.cat([beam_norm, delay_norm, p_norm], dim=-1) # [B, 8, 81]
            out_users = torch.sigmoid(self.head_loc(u_feat)) # [B, 8, 3]
            out_loc = out_users.reshape(B_size, 24)
        else:
            out_loc = torch.full((B_size, 24), 0.5, device=device, dtype=dtype)
            
        return {
            'out_0': out_ce,
            'ce_alpha': alpha_ce.squeeze(-1),
            'out_1': out_det,
            'out_2': out_prec,
            'out_3': out_dec,
            'out_4': out_loc
        }

