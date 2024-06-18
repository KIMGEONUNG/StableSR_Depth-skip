from ldm.modules.diffusionmodules.openaimodel import UNetModelDualcondV2
from ldm.modules.diffusionmodules.util import (
    timestep_embedding, )

from ldm.models.diffusion.ddpm import LatentDiffusionSRTextWT
import torch


class LatentDiffusionSRTextWT_Depthskip(LatentDiffusionSRTextWT):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_depth(self, depth):
        unet = self.model.diffusion_model
        unet.depth = depth
        unet.encoder_skip = list(range(unet.depth + 1, 12))
        unet.decoder_skip = list(range(0, 11 - unet.depth))


class UNetModelDualcondV2_Depthskip(UNetModelDualcondV2):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth = 12
        self.encoder_skip = []
        self.decoder_skip = []
        self.dimension_spec = {
            "D0": {
                "skip": 320,
                "input": 320,
                "spatial": 1
            },
            "D1": {
                "skip": 320,
                "input": 320,
                "spatial": 1
            },
            "D2": {
                "skip": 320,
                "input": 640,
                "spatial": 1
            },
            "D3": {
                "skip": 320,
                "input": 640,
                "spatial": 2
            },
            "D4": {
                "skip": 640,
                "input": 640,
                "spatial": 2
            },
            "D5": {
                "skip": 640,
                "input": 1280,
                "spatial": 2
            },
            "D6": {
                "skip": 640,
                "input": 1280,
                "spatial": 4
            },
            "D7": {
                "skip": 1280,
                "input": 1280,
                "spatial": 4
            },
            "D8": {
                "skip": 1280,
                "input": 1280,
                "spatial": 4
            },
            "D9": {
                "skip": 1280,
                "input": 1280,
                "spatial": 8
            },
            "D10": {
                "skip": 1280,
                "input": 1280,
                "spatial": 8
            },
            "D11": {
                "skip": 1280,
                "input": 1280,
                "spatial": 8
            }
        }

    def is_skip(self):
        return self.depth < 12

    def forward(self,
                x,
                timesteps=None,
                context=None,
                struct_cond=None,
                y=None,
                **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps,
                                   self.model_channels,
                                   repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0], )
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for i, module in enumerate(self.input_blocks):
            if self.is_skip() and i in self.encoder_skip:
                continue
            h = module(h, emb, context, struct_cond)
            hs.append(h)

        if self.is_skip():
            h = torch.zeros(h.shape[0],
                            self.dimension_spec[f"D{self.depth}"]["input"],
                            h.shape[2], h.shape[3]).to(h.device)
        else:
            h = self.middle_block(h, emb, context, struct_cond)

        for i, module in enumerate(self.output_blocks):
            if self.is_skip() and i in self.decoder_skip:
                continue
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context, struct_cond)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)
