# Copyright (c) IRIS Contributors. All rights reserved.
from types import MethodType

import torch

from .audio_layer_aggregation import (
    DEFAULT_AUDIO_WEIGHT_DOCUMENT_TEXT,
    DEFAULT_AUDIO_WEIGHT_TEXT_MODEL,
    AudioLayerAggregationConfig,
    GlobalAudioLayerWeightModule,
    MiniLMAudioLayerWeightModule,
    OnlineAudioLayerAggregator,
)


def _aggregation_config(config, **overrides) -> AudioLayerAggregationConfig:
    def value(name, default):
        override = overrides.get(name)
        value = getattr(config, name, default) if override is None else override
        return default if value is None else value

    mode = value("audio_layer_aggregation", "final")
    weight_backend = value("audio_weight_backend", "minilm")
    text_model = value("audio_weight_text_model", DEFAULT_AUDIO_WEIGHT_TEXT_MODEL)
    text_revision = value("audio_weight_text_revision", None)
    text_config = value("audio_weight_text_config", None)
    if mode == "learned" and weight_backend == "minilm" and text_config is None:
        from transformers import AutoConfig

        text_config = AutoConfig.from_pretrained(
            text_model, revision=text_revision
        ).to_dict()

    return AudioLayerAggregationConfig(
        mode=mode,
        layer_index=value("audio_layer_index", None),
        train_projection=value("train_audio_projection", False),
        weight_backend=weight_backend,
        init_final_prob=value("audio_weight_init_final_prob", None),
        text_model=text_model,
        text_revision=text_revision,
        text_config=text_config,
        train_text_encoder=value("train_audio_weight_text_encoder", False),
        document_text=value(
            "audio_weight_document_text", DEFAULT_AUDIO_WEIGHT_DOCUMENT_TEXT
        ),
    )


def _store_aggregation_config(
    config, aggregation_config: AudioLayerAggregationConfig
) -> None:
    config.audio_layer_aggregation = aggregation_config.mode
    config.audio_layer_index = aggregation_config.layer_index
    config.train_audio_projection = aggregation_config.train_projection
    if (
        aggregation_config.mode == "learned"
        and aggregation_config.weight_backend == "minilm"
    ):
        config.audio_weight_hidden_size = aggregation_config.text_config["hidden_size"]
    config.audio_weight_backend = aggregation_config.weight_backend
    config.audio_weight_init_final_prob = aggregation_config.init_final_prob
    config.audio_weight_text_model = aggregation_config.text_model
    config.audio_weight_text_revision = aggregation_config.text_revision
    config.audio_weight_text_config = aggregation_config.text_config
    config.train_audio_weight_text_encoder = aggregation_config.train_text_encoder
    config.audio_weight_document_text = aggregation_config.document_text
    config.iris_embedding_model = True


def _save_aggregation_config(
    config, aggregation_config: AudioLayerAggregationConfig, num_audio_layers: int
) -> None:
    _store_aggregation_config(config, aggregation_config)
    config.audio_weight_num_output_layers = num_audio_layers
    config.audio_weight_includes_pre_encoder = True


def apply_audio_aggregation_config(config, **overrides) -> None:
    """Store loader arguments before the Transformers model is constructed."""
    aggregation_config = _aggregation_config(config, **overrides)
    _store_aggregation_config(config, aggregation_config)
    config.audio_weight_includes_pre_encoder = True


def _set_trainable(modules, trainable: bool) -> None:
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = trainable


def initialize_audio_layer_weight_module(model) -> None:
    """Initialize a new controller after base checkpoint loading has completed."""
    module = getattr(model, "audio_layer_weight_module", None)
    if module is None:
        return
    if isinstance(module, MiniLMAudioLayerWeightModule):
        module.load_pretrained_encoder(
            model.config.audio_weight_text_model,
            getattr(model.config, "audio_weight_text_revision", None),
        )
        return
    module.reset_parameters()


def is_iris_checkpoint(config) -> bool:
    """Return whether config describes a saved IRIS checkpoint with trained weights."""
    return any(
        architecture.startswith("Iris")
        for architecture in getattr(config, "architectures", [])
    )


def configure_qwen_audio_aggregation(model, **overrides):
    """Attach IRIS aggregation to a Qwen2.5-Omni Thinker model."""
    aggregation_config = _aggregation_config(model.config, **overrides)
    tower = model.audio_tower
    num_states = len(tower.layers) + 1
    aggregation_config.validate(num_states)
    _save_aggregation_config(model.config, aggregation_config, num_states)

    _set_trainable((tower,), False)
    _set_trainable((tower.ln_post, tower.proj), aggregation_config.train_projection)
    if aggregation_config.mode == "final":
        model.audio_layer_weight_module = None
        return model

    if aggregation_config.mode == "learned":
        if aggregation_config.weight_backend == "minilm":
            if aggregation_config.text_config is None:
                raise ValueError(
                    "MiniLM audio routing requires audio_weight_text_config"
                )
            model.audio_layer_weight_module = MiniLMAudioLayerWeightModule(
                aggregation_config.text_config,
                num_states,
                aggregation_config.train_text_encoder,
                aggregation_config.init_final_prob,
            )
        elif aggregation_config.weight_backend == "global":
            model.audio_layer_weight_module = GlobalAudioLayerWeightModule(num_states)
    else:
        model.audio_layer_weight_module = None

    tower._iris_aggregation_mode = aggregation_config.mode
    tower._iris_aggregator = None
    tower._iris_layer_weights = None

    def prepare_audio_layer_weights(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        audio_weight_input_ids=None,
        audio_weight_attention_mask=None,
    ):
        if aggregation_config.mode != "learned":
            self.audio_tower._iris_layer_weights = None
            return
        if aggregation_config.weight_backend == "global":
            if input_ids is None:
                raise ValueError("Global audio routing requires a batch size")
            self.audio_tower._iris_layer_weights = self.audio_layer_weight_module(
                input_ids.size(0)
            )
        elif aggregation_config.weight_backend == "minilm":
            if audio_weight_input_ids is None or audio_weight_attention_mask is None:
                raise ValueError("MiniLM audio routing requires tokenized instructions")
            self.audio_tower._iris_layer_weights = self.audio_layer_weight_module(
                audio_weight_input_ids, audio_weight_attention_mask
            )

    model.prepare_audio_layer_weights = MethodType(prepare_audio_layer_weights, model)

    def model_pre_hook(module, args, kwargs):
        audio_weight_input_ids = kwargs.pop("audio_weight_input_ids", None)
        audio_weight_attention_mask = kwargs.pop("audio_weight_attention_mask", None)
        if (
            kwargs.get("input_features") is not None
            and kwargs.get("input_ids") is not None
            and aggregation_config.weight_backend != "global"
        ):
            module.prepare_audio_layer_weights(
                kwargs["input_ids"],
                kwargs.get("attention_mask"),
                kwargs.get("position_ids"),
                audio_weight_input_ids,
                audio_weight_attention_mask,
            )
        return args, kwargs

    def tower_pre_hook(module, args, kwargs):
        aftercnn_lens = kwargs.get("aftercnn_lens")
        if aftercnn_lens is None:
            raise ValueError("Qwen audio aggregation requires aftercnn_lens")
        if (
            aggregation_config.mode == "learned"
            and aggregation_config.weight_backend == "global"
        ):
            weights = model.audio_layer_weight_module(aftercnn_lens.numel())
            module._iris_layer_weights = weights
        else:
            weights = module._iris_layer_weights
        if aggregation_config.mode == "learned" and (
            weights is None or weights.size(0) != aftercnn_lens.numel()
        ):
            raise ValueError(
                "Audio layer weights are not aligned with the Qwen audio batch"
            )
        module._iris_owner_indices = torch.repeat_interleave(
            torch.arange(aftercnn_lens.numel(), device=aftercnn_lens.device),
            aftercnn_lens,
        )
        module._iris_aggregator = OnlineAudioLayerAggregator(
            aggregation_config.mode,
            weights,
            aggregation_config.selected_state_index,
        )

    def layer_hook(layer_index):
        def hook(module, args, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            tower._iris_aggregator.add(
                hidden_states, layer_index, tower._iris_owner_indices
            )

        return hook

    def first_layer_pre_hook(module, args, kwargs):
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and args:
            hidden_states = args[0]
        if hidden_states is None:
            raise ValueError("Qwen pre-encoder audio state is missing")
        tower._iris_aggregator.add(hidden_states, 0, tower._iris_owner_indices)

    def tower_hook(module, args, kwargs, output):
        mixed = module._iris_aggregator.result(num_states)
        aftercnn_lens = kwargs["aftercnn_lens"]
        token_audio = []
        for states in mixed.split(aftercnn_lens.tolist(), dim=0):
            states = module.avg_pooler(states.transpose(0, 1)).transpose_(0, 1)
            token_audio.append(module.proj(module.ln_post(states)))
        token_audio = torch.cat(token_audio, dim=0)
        output.last_hidden_state = token_audio
        module._iris_aggregator = None
        module._iris_layer_weights = None
        return output

    model.register_forward_pre_hook(model_pre_hook, with_kwargs=True)
    tower.register_forward_pre_hook(tower_pre_hook, with_kwargs=True)
    tower.layers[0].register_forward_pre_hook(first_layer_pre_hook, with_kwargs=True)
    for index, layer in enumerate(tower.layers, start=1):
        layer.register_forward_hook(layer_hook(index))
    tower.register_forward_hook(tower_hook, with_kwargs=True)
    return model


def configure_gemma_audio_aggregation(model, **overrides):
    """Attach IRIS aggregation to a Gemma 4 conditional-generation model."""
    core = model.model
    aggregation_config = _aggregation_config(model.config, **overrides)
    tower = core.audio_tower
    num_states = len(tower.layers) + 1
    aggregation_config.validate(num_states)
    _save_aggregation_config(model.config, aggregation_config, num_states)

    _set_trainable((tower, core.embed_audio), False)
    projection_modules = (
        tower.output_proj,
        core.embed_audio.embedding_pre_projection_norm,
        core.embed_audio.embedding_projection,
    )
    _set_trainable(projection_modules, aggregation_config.train_projection)
    if aggregation_config.mode == "final":
        model.audio_layer_weight_module = None
        return model

    if aggregation_config.mode == "learned":
        if aggregation_config.weight_backend == "minilm":
            if aggregation_config.text_config is None:
                raise ValueError(
                    "MiniLM audio routing requires audio_weight_text_config"
                )
            model.audio_layer_weight_module = MiniLMAudioLayerWeightModule(
                aggregation_config.text_config,
                num_states,
                aggregation_config.train_text_encoder,
                aggregation_config.init_final_prob,
            )
        elif aggregation_config.weight_backend == "global":
            model.audio_layer_weight_module = GlobalAudioLayerWeightModule(num_states)
    else:
        model.audio_layer_weight_module = None

    tower._iris_aggregation_mode = aggregation_config.mode
    tower._iris_aggregator = None
    tower._iris_layer_weights = None

    def prepare_audio_layer_weights(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        audio_weight_input_ids=None,
        audio_weight_attention_mask=None,
        audio_batch_size=None,
    ):
        if aggregation_config.mode != "learned":
            tower._iris_layer_weights = None
            return
        if aggregation_config.weight_backend == "global":
            if audio_batch_size is None:
                if input_ids is None:
                    raise ValueError("Global audio routing requires a batch size")
                audio_batch_size = input_ids.size(0)
            weights = self.audio_layer_weight_module(audio_batch_size)
        elif aggregation_config.weight_backend == "minilm":
            if audio_weight_input_ids is None or audio_weight_attention_mask is None:
                raise ValueError("MiniLM audio routing requires tokenized instructions")
            weights = self.audio_layer_weight_module(
                audio_weight_input_ids, audio_weight_attention_mask
            )
        tower._iris_layer_weights = weights

    model.prepare_audio_layer_weights = MethodType(prepare_audio_layer_weights, model)

    def model_pre_hook(module, args, kwargs):
        audio_weight_input_ids = kwargs.pop("audio_weight_input_ids", None)
        audio_weight_attention_mask = kwargs.pop("audio_weight_attention_mask", None)
        if kwargs.get("input_features") is None:
            return args, kwargs
        module.prepare_audio_layer_weights(
            kwargs.get("input_ids"),
            kwargs.get("attention_mask"),
            kwargs.get("position_ids"),
            audio_weight_input_ids,
            audio_weight_attention_mask,
            kwargs["input_features"].size(0),
        )
        weights = tower._iris_layer_weights
        if aggregation_config.mode == "learned" and (
            weights is None or weights.size(0) != kwargs["input_features"].size(0)
        ):
            raise ValueError(
                "Audio layer weights are not aligned with the Gemma audio batch"
            )
        return args, kwargs

    def tower_pre_hook(module, args, kwargs):
        module._iris_aggregator = OnlineAudioLayerAggregator(
            aggregation_config.mode,
            module._iris_layer_weights,
            aggregation_config.selected_state_index,
        )

    def layer_hook(layer_index):
        def hook(module, args, output):
            tower._iris_aggregator.add(output, layer_index)

        return hook

    def first_layer_pre_hook(module, args, kwargs):
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and args:
            hidden_states = args[0]
        if hidden_states is None:
            raise ValueError("Gemma pre-encoder audio state is missing")
        tower._iris_aggregator.add(hidden_states, 0)

    def tower_hook(module, args, kwargs, output):
        mixed = module._iris_aggregator.result(num_states)
        output.last_hidden_state = module.output_proj(mixed)
        module._iris_aggregator = None
        module._iris_layer_weights = None
        return output

    model.register_forward_pre_hook(model_pre_hook, with_kwargs=True)
    tower.register_forward_pre_hook(tower_pre_hook, with_kwargs=True)
    tower.layers[0].register_forward_pre_hook(first_layer_pre_hook, with_kwargs=True)
    for index, layer in enumerate(tower.layers, start=1):
        layer.register_forward_hook(layer_hook(index))
    tower.register_forward_hook(tower_hook, with_kwargs=True)
    return model
