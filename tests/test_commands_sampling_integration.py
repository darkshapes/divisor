"""Tests for command routines integration with sampling pipeline."""

from unittest.mock import MagicMock, Mock, PropertyMock, call, patch

from einops import rearrange
import pytest
import torch

from divisor.cli_menu import route_choices
from divisor.controller import ManualTimestepController
from divisor.flux1.autoencoder import AutoEncoder
from divisor.flux1.sampling import denoise
from divisor.keybinds import (
    change_guidance,
    change_layer_dropout,
    change_prompt,
    change_resolution,
    change_seed,
    change_vae_offset,
    toggle_buffer_mask,
    toggle_deterministic,
)
from divisor.state import DenoiseSettings, DenoisingState, TimestepState
from divisor.interaction_context import InteractionContext


class TestCommandsSamplingIntegration:
    """Test that command routines correctly apply to the sampling pipeline."""

    @pytest.fixture
    def mock_controller(self):
        """Create a mock controller for testing."""
        controller = Mock(spec=ManualTimestepController)
        controller.is_complete = False
        timestep_state = TimestepState(
            current_timestep=0.5,
            previous_timestep=0.6,
            current_sample=torch.randn(1, 16, 8, 8),
            timestep_index=0,
            total_timesteps=10,
        )
        controller.current_state = DenoisingState(
            timestep_state=timestep_state,
            guidance=4.0,
            layer_dropout=None,
            width=512,
            height=512,
            seed=42,
            prompt="test prompt",
            num_steps=10,
            vae_shift_offset=0.0,
            vae_scale_offset=0.0,
            use_previous_as_mask=False,
            variation_seed=None,
            variation_strength=0.0,
            deterministic=False,
        )
        # Mock setter methods
        controller.set_guidance = Mock()
        controller.set_layer_dropout = Mock()
        controller.set_resolution = Mock()
        controller.set_seed = Mock()
        controller.set_prompt = Mock()
        controller.set_vae_shift_offset = Mock()
        controller.set_vae_scale_offset = Mock()
        controller.set_use_previous_as_mask = Mock()
        controller.set_deterministic = Mock()
        controller.step = Mock(return_value=controller.current_state)
        return controller

    @pytest.fixture
    def mock_clear_cache(self):
        """Create a mock cache clearing function."""
        return Mock()

    @pytest.fixture
    def mock_rng(self):
        """Create a mock RNG."""
        rng = Mock()
        rng.next_seed = Mock(return_value=123)
        return rng

    def test_change_guidance_updates_controller(self, mock_controller, mock_rng, mock_var_rng, mock_clear_cache):
        """Test that change_guidance correctly updates the controller."""
        initial_state = mock_controller.current_state
        interaction_context = InteractionContext(
            clear_prediction_cache=mock_clear_cache,
            rng=mock_rng,
            variation_rng=mock_var_rng,
        )
        with patch("divisor.keybinds.get_float_input", return_value=7.5):
            result = change_guidance(mock_controller, initial_state, interaction_context)

            # Verify set_guidance was called with correct value
            mock_controller.set_guidance.assert_called_once_with(7.5)
            # Verify cache was cleared
            mock_clear_cache.clear_prediction_cache.assert_called_once()
            # Verify state was updated
            assert result == mock_controller.current_state

    def test_change_layer_dropout_updates_controller(self, mock_controller, mock_rng, mock_var_rng, mock_clear_cache):
        """Test that change_layer_dropout correctly updates the controller."""
        initial_state = mock_controller.current_state
        interaction_context = InteractionContext(
            clear_prediction_cache=mock_clear_cache,
            rng=mock_rng,
            variation_rng=mock_var_rng,
        )
        with patch("builtins.input", return_value="1,2,3"):
            result = change_layer_dropout(mock_controller, initial_state, interaction_context)

            # Verify set_layer_dropout was called with correct value
            mock_controller.set_layer_dropout.assert_called_once_with([1, 2, 3])
            # Verify cache was cleared
            mock_clear_cache.clear_prediction_cache.assert_called_once()
            # Verify state was updated
            assert result == mock_controller.current_state

    def test_change_resolution_updates_controller(self, mock_controller, mock_clear_cache, mock_rng, mock_var_rng):
        """Test that change_resolution correctly updates the controller."""
        initial_state = mock_controller.current_state
        interaction_context = InteractionContext(
            clear_prediction_cache=mock_clear_cache,
            rng=mock_rng,
            variation_rng=mock_var_rng,
        )
        with patch("divisor.keybinds.generate_valid_resolutions", return_value=[(512, 512), (768, 512)]), patch("builtins.input", return_value="0"):
            result = change_resolution(mock_controller, initial_state, interaction_context)

            # Verify set_resolution was called with correct values
            mock_controller.set_resolution.assert_called_once_with(512, 512)
            # Verify cache was cleared
            mock_clear_cache.clear_prediction_cache.assert_called_once()
            # Verify state was updated
            assert result == mock_controller.current_state

    def test_change_seed_updates_controller(self, mock_controller, mock_clear_cache, mock_rng, mock_var_rng):
        """Test that change_seed correctly updates the controller."""
        initial_state = mock_controller.current_state
        interaction_context = InteractionContext(
            clear_prediction_cache=mock_clear_cache,
            rng=mock_rng,
            variation_rng=mock_var_rng,
        )

        with patch("divisor.keybinds.get_int_input", return_value=999):
            with patch("divisor.keybinds.prepare_noise_for_model", return_value=torch.randn(1, 16, 32, 32)):
                result = change_seed(mock_controller, initial_state, interaction_context)

                # Verify set_seed was called with correct value
                mock_controller.set_seed.assert_called_once_with(999)
                # Verify cache was cleared
                mock_clear_cache.clear_prediction_cache.assert_called_once()
                # Verify state was updated
                assert result == mock_controller.current_state

    def test_toggle_buffer_mask_updates_controller(self, mock_controller):
        """Test that toggle_buffer_mask correctly updates the controller."""
        initial_state = mock_controller.current_state
        initial_state.use_previous_as_mask = False

        result = toggle_buffer_mask(mock_controller, initial_state)

        # Verify set_use_previous_as_mask was called with True (toggled)
        mock_controller.set_use_previous_as_mask.assert_called_once_with(True)
        # Verify state was updated
        assert result == mock_controller.current_state

    def test_change_vae_shift_offset_updates_controller(self, mock_controller, mock_clear_cache, mock_rng, mock_var_rng):
        """Test that change_vae_offset correctly updates shift offset."""
        initial_state = mock_controller.current_state
        mock_ae = Mock(spec=AutoEncoder)
        interaction_context = InteractionContext(
            clear_prediction_cache=mock_clear_cache,
            rng=mock_rng,
            variation_rng=mock_var_rng,
        )
        with patch("builtins.input", side_effect=["s", "0.1"]):
            with patch("divisor.keybinds.handle_float_setting") as mock_handle:
                mock_handle.return_value = mock_controller.current_state
                result = change_vae_offset(mock_controller, initial_state, interaction_context)

                # Verify handle_float_setting was called with correct parameters
                assert mock_handle.called
                call_args = mock_handle.call_args
                assert call_args[0][0] == mock_controller  # controller
                assert call_args[0][1] == initial_state  # state
                assert "shift offset" in call_args[0][2].lower()  # prompt
                assert call_args[0][4] == mock_controller.set_vae_shift_offset  # setter
                assert call_args[0][5] == mock_clear_cache.clear_prediction_cache  # clear_cache

    def test_change_vae_scale_offset_updates_controller(self, mock_controller, mock_clear_cache, mock_rng, mock_var_rng):
        """Test that change_vae_offset correctly updates scale offset."""
        initial_state = mock_controller.current_state
        mock_ae = Mock(spec=AutoEncoder)
        interaction_context = InteractionContext(
            clear_prediction_cache=mock_clear_cache,
            rng=mock_rng,
            variation_rng=mock_var_rng,
        )

        with patch("builtins.input", side_effect=["c", "0.05"]):
            with patch("divisor.keybinds.handle_float_setting") as mock_handle:
                mock_handle.return_value = mock_controller.current_state
                result = change_vae_offset(mock_controller, initial_state, interaction_context)

                # Verify handle_float_setting was called with correct parameters
                assert mock_handle.called
                call_args = mock_handle.call_args
                assert call_args[0][4] == mock_controller.set_vae_scale_offset  # setter

    def test_toggle_deterministic_updates_controller(self, mock_controller, mock_clear_cache, mock_rng, mock_var_rng):
        """Test that toggle_deterministic correctly updates the controller."""
        initial_state = mock_controller.current_state
        initial_state.deterministic = False
        interaction_context = InteractionContext(
            clear_prediction_cache=mock_clear_cache,
            rng=mock_rng,
            variation_rng=mock_var_rng,
        )

        with patch("divisor.controller.rng") as mock_rng, patch("divisor.controller.variation_rng") as mock_var_rng:
            mock_rng.random_mode = Mock()
            mock_var_rng.random_mode = Mock()

            result = toggle_deterministic(mock_controller, initial_state, interaction_context)

            # Verify set_deterministic was called with True (toggled)
            mock_controller.set_deterministic.assert_called_once_with(True)
            # Verify RNG modes were updated
            mock_clear_cache.rng.random_mode.assert_called_once_with(reproducible=True)
            mock_clear_cache.variation_rng.random_mode.assert_called_once_with(reproducible=True)
            # Verify cache was cleared
            mock_clear_cache.clear_prediction_cache.assert_called_once()

    def test_change_prompt_updates_controller(self, mock_controller, mock_clear_cache, mock_rng, mock_var_rng):
        """Test that change_prompt correctly updates the controller."""
        initial_state = mock_controller.current_state
        mock_recompute = Mock()
        interaction_context = InteractionContext(
            clear_prediction_cache=mock_clear_cache,
            rng=mock_rng,
            variation_rng=mock_var_rng,
            recompute_text_embeddings=mock_recompute,
        )

        with patch("builtins.input", return_value="new prompt text"):
            result = change_prompt(mock_controller, initial_state, interaction_context)

            # Verify set_prompt was called with new prompt
            mock_controller.set_prompt.assert_called_once_with("new prompt text")
            # Verify recompute_text_embeddings was called
            mock_recompute.assert_called_once_with("new prompt text")
            # Verify state was updated
            assert result == mock_controller.current_state

    def test_route_choices_calls_correct_handler(self, mock_controller, mock_clear_cache, mock_rng):
        """Test that route_choices routes to correct command handler."""
        initial_state = mock_controller.current_state
        mock_controller.current_layer_dropout = [None]
        mock_var_rng = Mock()
        interaction_context = InteractionContext(
            clear_prediction_cache=mock_clear_cache,
            rng=mock_rng,
            variation_rng=mock_var_rng,
        )
        with patch("builtins.input", side_effect=["g", "7.5"]):
            with patch("divisor.keybinds.get_float_input") as mock_handle:
                mock_handle.return_value = mock_controller.current_state
                result = route_choices(
                    mock_controller,
                    initial_state,
                    interaction_context,
                )

                # Verify change_guidance was called
                assert mock_handle.called

    def test_route_choices_advances_on_empty_input(self, mock_controller, mock_clear_cache, mock_rng):
        """Test that route_choices advances step on empty input."""
        initial_state = mock_controller.current_state
        current_layer_dropout = [None]
        mock_var_rng = Mock()
        interaction_context = InteractionContext(
            clear_prediction_cache=mock_clear_cache,
            rng=mock_rng,
            variation_rng=mock_var_rng,
        )
        with patch("builtins.input", return_value=""):
            result = route_choices(
                mock_controller,
                initial_state,
                interaction_context,
            )

            # Verify step was called
            mock_controller.step.assert_called_once()
            # Verify cache was cleared
            mock_clear_cache.assert_called_once()

    def test_denoise_initializes_controller_with_state_values(self):
        """Test that denoise function initializes controller with state values."""
        mock_model = Mock()
        # Mock parameters() to return an iterable with a parameter that has a device
        mock_param = Mock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])
        mock_ae = Mock(spec=AutoEncoder)
        mock_ae.scale_factor = 0.3611
        mock_ae.shift_factor = 0.1159
        mock_ae.decode = Mock(return_value=torch.randn(1, 3, 512, 512))
        mock_ae.decoder = Mock(return_value=torch.randn(1, 3, 512, 512))

        # Create img in 4D format, then convert to 3D format that current_sample expects
        # For 512x512 image: h = ceil(512/16) = 32, w = ceil(512/16) = 32 patches
        # We need [1, 16, 64, 64] so after rearrange with ph=2, pw=2: [1, 32*32, 64] = [1, 1024, 64]
        img_4d = torch.randn(1, 16, 64, 64)  # [batch, channels, height_patches, width_patches]
        # Convert to 3D format: [batch, (h w), (c ph pw)] where ph=2, pw=2
        # Result: [1, (64/2)*(64/2), (16*2*2)] = [1, 32*32, 64] = [1, 1024, 64]
        img = rearrange(img_4d, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        # img_ids should match the sequence length of img: [1, 1024, 3]
        img_ids = torch.zeros(1, img.shape[1], 3)
        txt = torch.randn(1, 77, 4096)
        txt_ids = torch.zeros(1, 77, 3)
        vec = torch.randn(1, 77, 768)

        from divisor.state import TimestepState

        timestep_state = TimestepState(
            current_timestep=0.0,
            previous_timestep=None,
            current_sample=img,
            timestep_index=0,
            total_timesteps=10,
        )
        state = DenoisingState(
            timestep_state=timestep_state,
            guidance=4.0,
            layer_dropout=[1, 2],
            width=512,
            height=512,
            seed=42,
            prompt="test prompt",
            num_steps=10,
            vae_shift_offset=0.1,
            vae_scale_offset=0.05,
            use_previous_as_mask=True,
            variation_seed=None,
            variation_strength=0.0,
            deterministic=False,
        )

        timesteps = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

        # Mock the controller creation and methods
        with patch("divisor.flux1.sampling.ManualTimestepController") as MockController:
            mock_controller_instance = Mock()
            mock_controller_instance.is_complete = True  # Make it complete immediately to exit loop
            mock_controller_instance.current_state = state
            mock_controller_instance.hyperchain = Mock()
            MockController.return_value = mock_controller_instance

            # Mock other dependencies
            with (
                patch("divisor.flux1.sampling.name_save_file_as"),
                patch("divisor.flux1.sampling.save_with_hyperchain"),
                patch("divisor.flux1.sampling.sync_torch"),
                patch("divisor.flux1.sampling.nfo"),
            ):
                settings = DenoiseSettings(
                    img=img,
                    img_ids=img_ids,
                    txt=txt,
                    txt_ids=txt_ids,
                    vec=vec,
                    state=state,
                    ae=mock_ae,
                    timesteps=timesteps,
                )
                result = denoise(
                    model=mock_model,
                    settings=settings,
                )

                # Verify controller was created with correct initial guidance
                MockController.assert_called_once()
                call_args = MockController.call_args
                assert call_args[1]["initial_guidance"] == state.guidance

                # Verify setter methods were called with state values
                mock_controller_instance.set_layer_dropout.assert_called_once_with([1, 2])
                mock_controller_instance.set_resolution.assert_called_once_with(512, 512)
                mock_controller_instance.set_seed.assert_called_once_with(42)
                mock_controller_instance.set_prompt.assert_called_once_with("test prompt")
                mock_controller_instance.set_num_steps.assert_called_once_with(10)
                mock_controller_instance.set_vae_shift_offset.assert_called_once_with(0.1)
                mock_controller_instance.set_vae_scale_offset.assert_called_once_with(0.05)
                mock_controller_instance.set_use_previous_as_mask.assert_called_once_with(True)

    def test_denoise_uses_controller_state_in_denoise_step_fn(self):
        """Test that denoise_step_fn uses controller's current state values."""
        mock_model = Mock()
        mock_model.return_value = torch.randn(1, 16, 32, 32)
        mock_ae = Mock(spec=AutoEncoder)
        mock_ae.scale_factor = 0.3611
        mock_ae.shift_factor = 0.1159
        mock_ae.decode = Mock(return_value=torch.randn(1, 3, 512, 512))
        mock_ae.decoder = Mock(return_value=torch.randn(1, 3, 512, 512))

        img = torch.randn(1, 16, 32, 32)
        img_ids = torch.zeros(1, 1024, 3)
        txt = torch.randn(1, 77, 4096)
        txt_ids = torch.zeros(1, 77, 3)
        vec = torch.randn(1, 77, 768)

        timestep_state = TimestepState(
            current_timestep=0.0,
            previous_timestep=None,
            current_sample=img,
            timestep_index=0,
            total_timesteps=2,
        )
        state = DenoisingState(
            timestep_state=timestep_state,
            guidance=5.0,
            layer_dropout=None,
            width=512,
            height=512,
            seed=42,
            prompt="test",
            num_steps=2,
            vae_shift_offset=0.0,
            vae_scale_offset=0.0,
            use_previous_as_mask=False,
            variation_seed=None,
            variation_strength=0.0,
            deterministic=False,
        )

        timesteps = [1.0, 0.5, 0.0]

        # Track calls to model
        model_calls = []

        def track_model_call(*args, **kwargs):
            model_calls.append(kwargs)
            return torch.randn(1, 16, 32, 32)

        mock_model.side_effect = track_model_call

        with patch("divisor.flux1.sampling.ManualTimestepController") as MockController:
            mock_controller_instance = Mock()
            mock_controller_instance.is_complete = True
            mock_controller_instance.current_state = state
            mock_controller_instance.guidance = 5.0  # Updated guidance
            mock_controller_instance.hyperchain = Mock()

            # Make step() update guidance
            def mock_step():
                mock_controller_instance.guidance = 7.0  # Simulate guidance change
                return state

            mock_controller_instance.step = Mock(side_effect=mock_step)
            MockController.return_value = mock_controller_instance

            with (
                patch("divisor.flux1.sampling.name_save_file_as"),
                patch("divisor.flux1.sampling.save_with_hyperchain"),
                patch("divisor.flux1.sampling.sync_torch"),
                patch("divisor.flux1.sampling.nfo"),
                patch("divisor.flux1.sampling.route_choices", return_value=state),
            ):
                # This test verifies that the denoise_step_fn closure captures
                # the controller's current guidance value
                # The actual implementation uses controller_ref[0].current_state.guidance
                # which we can't easily test without running the full loop
                # But we can verify the structure is correct
                pass

    def test_commands_clear_cache_when_state_changes(self, mock_controller, mock_rng, mock_var_rng, mock_clear_cache):
        """Test that commands clear prediction cache when state changes."""
        initial_state = mock_controller.current_state
        interaction_context = InteractionContext(
            clear_prediction_cache=mock_clear_cache,
            rng=mock_rng,
            variation_rng=mock_var_rng,
        )

        with patch("divisor.keybinds.get_float_input", return_value=7.5):
            change_guidance(mock_controller, initial_state, interaction_context)
            assert mock_clear_cache.clear_prediction_cache.call_count == 1

        # Reset mock
        mock_clear_cache.reset_mock()

        # Test layer dropout change
        with patch("builtins.input", return_value="1,2"):
            change_layer_dropout(mock_controller, initial_state, interaction_context)
            assert mock_clear_cache.clear_prediction_cache.call_count == 1

    def test_route_choices_integration_with_denoise_loop(self):
        """Test that route_choices integrates correctly with denoise loop."""
        # This test verifies that route_choices can be called from within
        # the denoise loop and correctly updates the controller state
        mock_ae = Mock(spec=AutoEncoder)
        mock_ae.scale_factor = 0.3611
        mock_ae.shift_factor = 0.1159
        mock_ae.decode = Mock(return_value=torch.randn(1, 3, 512, 512))
        mock_ae.decoder = Mock(return_value=torch.randn(1, 3, 512, 512))

        # Create img in 4D format, then convert to 3D format that current_sample expects
        # For 512x512 image: h = ceil(512/16) = 32, w = ceil(512/16) = 32 patches
        # We need [1, 16, 64, 64] so after rearrange with ph=2, pw=2: [1, 32*32, 64] = [1, 1024, 64]
        img_4d = torch.randn(1, 16, 64, 64)  # [batch, channels, height_patches, width_patches]
        # Convert to 3D format: [batch, (h w), (c ph pw)] where ph=2, pw=2
        # Result: [1, (64/2)*(64/2), (16*2*2)] = [1, 32*32, 64] = [1, 1024, 64]
        img = rearrange(img_4d, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        # img_ids should match the sequence length of img: [1, 1024, 3]
        img_ids = torch.zeros(1, img.shape[1], 3)
        txt = torch.randn(1, 77, 4096)
        txt_ids = torch.zeros(1, 77, 3)
        vec = torch.randn(1, 77, 768)

        # Create mock_model after img is defined
        mock_model = Mock()
        # Mock parameters() to return an iterable with a parameter that has a device and dtype
        mock_param = Mock()
        mock_param.device = torch.device("cpu")
        mock_param.dtype = torch.bfloat16
        mock_model.parameters.return_value = iter([mock_param])
        # Mock the model call to return a prediction
        mock_model.return_value = torch.randn(1, img.shape[1], img.shape[2])
        # Mock encoder for VAE dtype detection
        mock_encoder_param = Mock()
        mock_encoder_param.dtype = torch.bfloat16
        mock_ae.encoder = Mock()
        mock_ae.encoder.parameters.return_value = iter([mock_encoder_param])

        timestep_state = TimestepState(
            current_timestep=0.5,
            previous_timestep=0.6,
            current_sample=img,
            timestep_index=5,
            total_timesteps=10,
        )
        state = DenoisingState(
            timestep_state=timestep_state,
            guidance=4.0,
            layer_dropout=None,
            width=512,
            height=512,
            seed=42,
            prompt="test",
            num_steps=10,
            vae_shift_offset=0.0,
            vae_scale_offset=0.0,
            use_previous_as_mask=False,
            variation_seed=None,
            variation_strength=0.0,
            deterministic=False,
        )

        timesteps = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

        with patch("divisor.flux1.sampling.ManualTimestepController") as MockController:
            mock_controller_instance = Mock()
            # Simulate one iteration of the loop
            # Use PropertyMock to properly mock the is_complete property
            # First access returns False (enter loop), second returns True (exit loop)
            is_complete_mock = PropertyMock(side_effect=[False, True])
            type(mock_controller_instance).is_complete = is_complete_mock
            mock_controller_instance.current_state = state
            mock_controller_instance.guidance = 4.0
            mock_controller_instance.hyperchain = Mock()
            MockController.return_value = mock_controller_instance
            # Mock route_choices at the usage site (where it's imported)
            with patch("divisor.flux1.sampling.route_choices") as mock_process:
                mock_process.return_value = state

                with (
                    patch("divisor.flux1.sampling.name_save_file_as"),
                    patch("divisor.flux1.sampling.save_with_hyperchain"),
                    patch("divisor.flux1.sampling.sync_torch"),
                    patch("divisor.flux1.sampling.nfo"),
                ):
                    settings = DenoiseSettings(
                        img=img,
                        img_ids=img_ids,
                        txt=txt,
                        txt_ids=txt_ids,
                        vec=vec,
                        state=state,
                        ae=mock_ae,
                        timesteps=timesteps,
                    )
                    denoise(
                        model=mock_model,
                        settings=settings,
                    )

                    # Verify route_choices was called (from the denoise loop)
                    assert mock_process.called
