"""Native sd-scripts loading; no inference-time merging of trainable LoRA."""
import gc
from types import SimpleNamespace
import torch


class AnimaBackend:
    def __init__(self, config, device):
        import anima_minimal_inference as infer
        from library import strategy_anima, strategy_base, anima_train_utils
        from networks import lora_anima
        self.infer, self.device = infer, device
        self.args = SimpleNamespace(
            dit=config['dit'], text_encoder=config['text_encoder'], vae=config['vae'],
            attn_mode='torch', lycoris=False, lora_weight=None, lora_multiplier=None,
            fp8=False, fp8_scaled=False, text_encoder_cpu=False,
            vae_chunk_size=None, vae_disable_cache=False, qwen_image_vae_2d=True,
        )
        self.model = infer.load_dit_model(self.args, device, torch.bfloat16)
        self.text_encoder = infer.load_text_encoder(self.args, device=device)
        self.text_encoder.requires_grad_(False)
        self.network, state = lora_anima.create_network_from_weights(
            1.0, config['lora'], None, [self.text_encoder], self.model, for_inference=False)
        self.network.apply_to([self.text_encoder], self.model)
        self.network.load_state_dict(state, strict=True)
        self.network.to(device=device, dtype=torch.float32)
        # Keep the complete native checkpoint, including frozen TE/adapter LoRAs.
        self.network.requires_grad_(False)
        for module in self.network.unet_loras:
            if 'llm_adapter' not in module.lora_name:
                module.requires_grad_(True)
        self.network.eval()  # deterministic policy: no LoRA dropout during update
        self.params = [p for p in self.network.parameters() if p.requires_grad]
        if not self.params:
            raise ValueError('No trainable DiT LoRA parameters')
        strategy_base.TokenizeStrategy.set_strategy(strategy_anima.AnimaTokenizeStrategy(
            qwen3_path=config['text_encoder'], t5_tokenizer_path=None,
            qwen3_max_length=512, t5_max_length=512))
        strategy_base.TextEncodingStrategy.set_strategy(strategy_anima.AnimaTextEncodingStrategy())
        self.vae = anima_train_utils.load_qwen_image_vae(self.args, device='cpu')
        self.vae.to(dtype=torch.bfloat16).eval().requires_grad_(False)
        self.cache = {}

    def cache_prompts(self, prompts, negative):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for prompt in prompts:
                self.args.prompt, self.args.negative_prompt = prompt, negative
                c, n = self.infer.prepare_text_inputs(self.args, self.device, self.model,
                    {'text_encoder': self.text_encoder, 'conds_cache': self.cache})
                self.cache[prompt] = c['embed']
                self.cache[negative] = n['embed']
        # Every cached embedding remains valid because TE and adapter are frozen.
        self.text_encoder.to('cpu')
        for module in self.network.text_encoder_loras:
            module.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()

    def velocity(self, x, sigma, prompt, negative, cfg):
        x = x.to(self.device, dtype=torch.bfloat16)
        t = torch.full((x.shape[0],), float(sigma), device=self.device, dtype=torch.bfloat16)
        mask = torch.zeros(x.shape[0], 1, *x.shape[-2:], device=self.device, dtype=x.dtype)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            cond = self.model(x, t, self.cache[prompt][0].to(self.device), padding_mask=mask)
            if cfg == 1:
                return cond.float()
            uncond = self.model(x, t, self.cache[negative][0].to(self.device), padding_mask=mask)
            return (uncond + cfg * (cond - uncond)).float()

    def velocity_batch(self, x, sigmas, prompts):
        x = x.to(self.device, dtype=torch.bfloat16)
        t = sigmas.to(device=self.device, dtype=torch.bfloat16).view(-1)
        mask = torch.zeros(x.shape[0], 1, *x.shape[-2:], device=self.device, dtype=x.dtype)
        conds = [self.cache[p][0] for p in prompts]
        max_len = max(c.shape[1] for c in conds)
        if any(c.shape[1] != max_len for c in conds):
            padded = []
            for c in conds:
                if c.shape[1] < max_len:
                    pad = torch.zeros(c.shape[0], max_len - c.shape[1], *c.shape[2:], device=c.device, dtype=c.dtype)
                    c = torch.cat([c, pad], dim=1)
                padded.append(c)
            cond = torch.cat(padded, dim=0).to(self.device)
        else:
            cond = torch.cat(conds, dim=0).to(self.device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return self.model(x, t, cond, padding_mask=mask).float()

    @torch.no_grad()
    def save_image(self, latent, path):
        from PIL import Image
        pixels = self.infer.decode_latent(self.vae, latent, self.device)
        pixels = ((pixels.clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).numpy()
        Image.fromarray(pixels).save(path)

    def save(self, path, metadata):
        self.network.save_weights(str(path), torch.float32, metadata)
