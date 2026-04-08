from configs.default import ModelConfig, LoraConfig, TrainingConfig, DataConfig, APIConfig


class TestModelConfig:

    def test_defaults(self):
        config = ModelConfig()
        assert "llama" in config.model_id.lower() or "Llama" in config.model_id
        assert config.load_in_4bit is True
        assert config.bnb_4bit_quant_type == "nf4"
    def test_override(self):
        config = ModelConfig(model_id="gpt2", load_in_4bit=False)
        assert config.model_id == "gpt2"
        assert config.load_in_4bit is False


class TestLoraConfig:
    def test_defaults(self):
        config = LoraConfig()
        assert config.r == 16
        assert config.lora_alpha == 32
        assert "q_proj" in config.target_modules
        assert config.task_type == "CAUSAL_LM"


class TestTrainingConfig:
    def test_defaults(self):
        config = TrainingConfig()
        assert config.num_train_epochs == 3
        assert config.learning_rate == 2e-5
        assert config.bf16 is True


class TestDataConfig:
    def test_defaults(self):
        config = DataConfig()
        assert config.test_size + config.val_size < 1.0
        assert config.max_samples is None

    def test_max_samples(self):
        config = DataConfig(max_samples=100)
        assert config.max_samples == 100


class TestAPIConfig:
    def test_defaults(self):
        config = APIConfig()
        assert config.port == 8000
        assert config.max_new_tokens == 256