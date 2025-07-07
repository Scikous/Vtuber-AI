import logging

import librosa
import numpy as np
import torch
from coqpit import Coqpit
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations
from trainer.io import load_fsspec

import TTS.vc.layers.freevc.modules as modules
from TTS.tts.layers.vits.discriminator import DiscriminatorS
from TTS.tts.utils.helpers import sequence_mask
from TTS.tts.utils.speakers import SpeakerManager
from TTS.vc.configs.freevc_config import FreeVCConfig
from TTS.vc.layers.freevc.commons import init_weights, rand_slice_segments
from TTS.vc.layers.freevc.mel_processing import mel_spectrogram_torch
from TTS.vc.layers.freevc.speaker_encoder.speaker_encoder import SpeakerEncoder as SpeakerEncoderEx
from TTS.vc.layers.freevc.wavlm import get_wavlm
from TTS.vc.models.base_vc import BaseVC
from TTS.vocoder.models.hifigan_discriminator import DiscriminatorP

logger = logging.getLogger(__name__)


class ResidualCouplingBlock(nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, n_flows=4, gin_channels=0):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class Encoder(nn.Module):
    def __init__(
        self, in_channels, out_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class Generator(torch.nn.Module):
    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        logger.info("Removing weight norm...")
        for l in self.ups:
            remove_parametrizations(l, "weight")
        for l in self.resblocks:
            remove_parametrizations(l, "weight")


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class SpeakerEncoder(torch.nn.Module):
    def __init__(self, mel_n_channels=80, model_num_layers=3, model_hidden_size=256, model_embedding_size=256):
        super().__init__()
        self.lstm = nn.LSTM(mel_n_channels, model_hidden_size, model_num_layers, batch_first=True)
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        self.relu = nn.ReLU()

    def forward(self, mels):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(mels)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        return embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)

    def compute_partial_slices(self, total_frames, partial_frames, partial_hop):
        mel_slices = []
        for i in range(0, total_frames - partial_frames, partial_hop):
            mel_range = torch.arange(i, i + partial_frames)
            mel_slices.append(mel_range)

        return mel_slices

    def embed_utterance(self, mel, partial_frames=128, partial_hop=64):
        mel_len = mel.size(1)
        last_mel = mel[:, -partial_frames:]

        if mel_len > partial_frames:
            mel_slices = self.compute_partial_slices(mel_len, partial_frames, partial_hop)
            mels = list(mel[:, s] for s in mel_slices)
            mels.append(last_mel)
            mels = torch.stack(tuple(mels), 0).squeeze(1)

            with torch.no_grad():
                partial_embeds = self(mels)
            embed = torch.mean(partial_embeds, axis=0).unsqueeze(0)
            # embed = embed / torch.linalg.norm(embed, 2)
        else:
            with torch.no_grad():
                embed = self(last_mel)

        return embed


class FreeVC(BaseVC):
    """

    Paper::
        https://arxiv.org/abs/2210.15418#

    Paper Abstract::
        Voice conversion (VC) can be achieved by first extracting source content information and target speaker
        information, and then reconstructing waveform with these information. However, current approaches normally
        either extract dirty content information with speaker information leaked in, or demand a large amount of
        annotated data for training. Besides, the quality of reconstructed waveform can be degraded by the
        mismatch between conversion model and vocoder. In this paper, we adopt the end-to-end framework of VITS for
        high-quality waveform reconstruction, and propose strategies for clean content information extraction without
        text annotation. We disentangle content information by imposing an information bottleneck to WavLM features,
        and propose the spectrogram-resize based data augmentation to improve the purity of extracted content
        information. Experimental results show that the proposed method outperforms the latest VC models trained with
        annotated data and has greater robustness.

    Original Code::
        https://github.com/OlaWod/FreeVC

    Examples:
        >>> from TTS.vc.configs.freevc_config import FreeVCConfig
        >>> from TTS.vc.models.freevc import FreeVC
        >>> config = FreeVCConfig()
        >>> model = FreeVC(config)
    """

    def __init__(self, config: Coqpit, speaker_manager: SpeakerManager = None):
        super().__init__(config, None, speaker_manager, None)

        self.init_multispeaker(config)

        self.spec_channels = self.args.spec_channels
        self.inter_channels = self.args.inter_channels
        self.hidden_channels = self.args.hidden_channels
        self.filter_channels = self.args.filter_channels
        self.n_heads = self.args.n_heads
        self.n_layers = self.args.n_layers
        self.kernel_size = self.args.kernel_size
        self.p_dropout = self.args.p_dropout
        self.resblock = self.args.resblock
        self.resblock_kernel_sizes = self.args.resblock_kernel_sizes
        self.resblock_dilation_sizes = self.args.resblock_dilation_sizes
        self.upsample_rates = self.args.upsample_rates
        self.upsample_initial_channel = self.args.upsample_initial_channel
        self.upsample_kernel_sizes = self.args.upsample_kernel_sizes
        self.segment_size = self.args.segment_size
        self.gin_channels = self.args.gin_channels
        self.ssl_dim = self.args.ssl_dim
        self.use_spk = self.args.use_spk

        self.enc_p = Encoder(self.args.ssl_dim, self.inter_channels, self.hidden_channels, 5, 1, 16)
        self.dec = Generator(
            self.inter_channels,
            self.resblock,
            self.resblock_kernel_sizes,
            self.resblock_dilation_sizes,
            self.upsample_rates,
            self.upsample_initial_channel,
            self.upsample_kernel_sizes,
            gin_channels=self.gin_channels,
        )
        self.enc_q = Encoder(
            self.spec_channels, self.inter_channels, self.hidden_channels, 5, 1, 16, gin_channels=self.gin_channels
        )
        self.flow = ResidualCouplingBlock(
            self.inter_channels, self.hidden_channels, 5, 1, 4, gin_channels=self.gin_channels
        )
        if not self.use_spk:
            self.enc_spk = SpeakerEncoder(model_hidden_size=self.gin_channels, model_embedding_size=self.gin_channels)
        else:
            self.load_pretrained_speaker_encoder()

        self.wavlm = get_wavlm()

    def load_pretrained_speaker_encoder(self):
        """Load pretrained speaker encoder model as mentioned in the paper."""
        logger.info("Loading pretrained speaker encoder model ...")
        self.enc_spk_ex = SpeakerEncoderEx(
            "https://github.com/coqui-ai/TTS/releases/download/v0.13.0_models/speaker_encoder.pt"
        )

    def init_multispeaker(self, config: Coqpit):
        """Initialize multi-speaker modules of a model. A model can be trained either with a speaker embedding layer
        or with external `d_vectors` computed from a speaker encoder model.

        You must provide a `speaker_manager` at initialization to set up the multi-speaker modules.

        Args:
            config (Coqpit): Model configuration.
            data (List, optional): Dataset items to infer number of speakers. Defaults to None.
        """
        self.num_spks = self.args.num_spks
        if self.speaker_manager:
            self.num_spks = self.speaker_manager.num_spks

    def forward(
        self,
        c: torch.Tensor,
        spec: torch.Tensor,
        g: torch.Tensor | None = None,
        mel: torch.Tensor | None = None,
        c_lengths: torch.Tensor | None = None,
        spec_lengths: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Forward pass of the model.

        Args:
            c: WavLM features. Shape: (batch_size, c_seq_len).
            spec: The input spectrogram. Shape: (batch_size, spec_seq_len, spec_dim).
            g: The speaker embedding. Shape: (batch_size, spk_emb_dim).
            mel: The input mel-spectrogram for the speaker encoder. Shape: (batch_size, mel_seq_len, mel_dim).
            c_lengths: The lengths of the WavLM features. Shape: (batch_size,).
            spec_lengths: The lengths of the spectrogram. Shape: (batch_size,).

        Returns:
            o: The output spectrogram. Shape: (batch_size, spec_seq_len, spec_dim).
            ids_slice: The slice indices. Shape: (batch_size, num_slices).
            spec_mask: The spectrogram mask. Shape: (batch_size, spec_seq_len).
            (z, z_p, m_p, logs_p, m_q, logs_q): A tuple of latent variables.
        """

        # If c_lengths is None, set it to the length of the last dimension of c
        if c_lengths is None:
            c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)

        # If spec_lengths is None, set it to the length of the last dimension of spec
        if spec_lengths is None:
            spec_lengths = (torch.ones(spec.size(0)) * spec.size(-1)).to(spec.device)

        # If use_spk is False, compute g from mel using enc_spk
        g = None
        if not self.use_spk:
            g = self.enc_spk(mel).unsqueeze(-1)

        # Compute m_p, logs_p, z, m_q, logs_q, and spec_mask using enc_p and enc_q
        _, m_p, logs_p, _ = self.enc_p(c, c_lengths)
        z, m_q, logs_q, spec_mask = self.enc_q(spec.transpose(1, 2), spec_lengths, g=g)

        # Compute z_p using flow
        z_p = self.flow(z, spec_mask, g=g)

        # Randomly slice z and compute o using dec
        z_slice, ids_slice = rand_slice_segments(z, spec_lengths, self.segment_size)
        o = self.dec(z_slice, g=g)

        return o, ids_slice, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    @torch.inference_mode()
    def inference(self, c, g=None, c_lengths=None):
        """
        Inference pass of the model

        Args:
            c (torch.Tensor): Input tensor. Shape: (batch_size, c_seq_len).
            g (torch.Tensor): Speaker embedding tensor. Shape: (batch_size, spk_emb_dim).
            mel (torch.Tensor): Mel-spectrogram tensor. Shape: (batch_size, mel_seq_len, mel_dim).
            c_lengths (torch.Tensor): Lengths of the input tensor. Shape: (batch_size,).

        Returns:
            torch.Tensor: Output tensor.
        """
        if c_lengths is None:
            c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)
        z_p, m_p, logs_p, c_mask = self.enc_p(c, c_lengths)
        z = self.flow(z_p, c_mask, g=g, reverse=True)
        o = self.dec(z * c_mask, g=g)
        return o

    def extract_wavlm_features(self, y):
        """Extract WavLM features from an audio tensor.

        Args:
            y (torch.Tensor): Audio tensor. Shape: (batch_size, audio_seq_len).
        """

        with torch.no_grad():
            c = self.wavlm.extract_features(y)[0]
        c = c.transpose(1, 2)
        return c

    def load_audio(self, wav):
        """Read and format the input audio."""
        if isinstance(wav, str):
            wav, _ = librosa.load(wav, sr=self.config.audio.input_sample_rate)
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav).to(self.device)
        if isinstance(wav, torch.Tensor):
            wav = wav.to(self.device)
        if isinstance(wav, list):
            wav = torch.from_numpy(np.array(wav)).to(self.device)
        return wav.float()

    @torch.inference_mode()
    def voice_conversion(self, src: str | torch.Tensor, tgt: list[str | torch.Tensor]):
        """
        Voice conversion pass of the model.

        Args:
            src (str or torch.Tensor): Source utterance.
            tgt (list of str or torch.Tensor): Target utterances.

        Returns:
            torch.Tensor: Output tensor.
        """

        # src
        wav_src = self.load_audio(src)
        c = self.extract_wavlm_features(wav_src[None, :])

        # tgt
        g_tgts = []
        for tg in tgt:
            wav_tgt = self.load_audio(tg).cpu().numpy()
            wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)

            if self.config.model_args.use_spk:
                g_tgts.append(self.enc_spk_ex.embed_utterance(wav_tgt)[None, :, None])
            else:
                wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0).to(self.device)
                mel_tgt = mel_spectrogram_torch(
                    wav_tgt,
                    self.config.audio.filter_length,
                    self.config.audio.n_mel_channels,
                    self.config.audio.input_sample_rate,
                    self.config.audio.hop_length,
                    self.config.audio.win_length,
                    self.config.audio.mel_fmin,
                    self.config.audio.mel_fmax,
                )
                g_tgts.append(self.enc_spk.embed_utterance(mel_tgt.transpose(1, 2)).unsqueeze(-1))

        g_tgt = torch.stack(g_tgts).mean(dim=0)
        audio = self.inference(c, g=g_tgt)
        return audio[0][0].data.cpu().float().numpy()

    def eval_step(): ...

    @staticmethod
    def init_from_config(config: FreeVCConfig) -> "FreeVC":
        model = FreeVC(config)
        return model

    def load_checkpoint(self, config, checkpoint_path, eval=False, strict=True, cache=False):
        state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"), cache=cache)
        self.load_state_dict(state["model"], strict=strict)
        if eval:
            self.eval()

    def train_step(): ...
