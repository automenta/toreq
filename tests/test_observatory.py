"""
Tests for the TorEq Dynamic Observatory module.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import torch
import numpy as np

from src.observatory import SynapseHeatmap, DynamicsCapture, ObservatoryMetrics
from src.observatory.heatmap import LayerState
from src.observatory.renderer import HeadlessRenderer, RendererConfig


class TestSynapseHeatmap:
    """Tests for SynapseHeatmap class."""
    
    def test_grid_size_computation(self):
        """Test that grid sizes are computed correctly."""
        heatmap = SynapseHeatmap()
        
        assert heatmap.compute_grid_size(64) == (8, 8)
        assert heatmap.compute_grid_size(256) == (16, 16)
        assert heatmap.compute_grid_size(1024) == (32, 32)
        # Non-perfect squares should round up
        assert heatmap.compute_grid_size(100) == (10, 10)
        assert heatmap.compute_grid_size(65) == (9, 9)
    
    def test_reshape_to_grid(self):
        """Test reshaping tensors to grids."""
        heatmap = SynapseHeatmap()
        
        # Perfect square
        tensor = torch.randn(4, 64)
        grid = heatmap.reshape_to_grid(tensor)
        assert grid.shape == (4, 8, 8)
        
        # Non-perfect square (should pad)
        tensor = torch.randn(4, 60)
        grid = heatmap.reshape_to_grid(tensor)
        assert grid.shape == (4, 8, 8)  # Rounds up to 64
    
    def test_activation_to_red(self):
        """Test red channel generation from activation."""
        heatmap = SynapseHeatmap()
        
        activation = torch.randn(4, 64)
        red = heatmap.activation_to_red(activation)
        
        assert red.shape == (8, 8)
        assert red.dtype == np.uint8
        assert red.min() >= 0
        assert red.max() <= 255
    
    def test_velocity_to_green(self):
        """Test green channel generation from velocity."""
        heatmap = SynapseHeatmap()
        
        velocity = torch.randn(4, 64) * 0.1
        green = heatmap.velocity_to_green(velocity)
        
        assert green.shape == (8, 8)
        assert green.dtype == np.uint8
    
    def test_nudge_to_blue(self):
        """Test blue channel generation from nudge."""
        heatmap = SynapseHeatmap()
        
        nudge = torch.randn(4, 64) * 0.2
        blue = heatmap.nudge_to_blue(nudge)
        
        assert blue.shape == (8, 8)
        assert blue.dtype == np.uint8
    
    def test_generate_rgb(self):
        """Test full RGB generation from LayerState."""
        heatmap = SynapseHeatmap()
        
        state = LayerState(
            activation=torch.randn(4, 64),
            velocity=torch.randn(4, 64) * 0.1,
            nudge=torch.randn(4, 64) * 0.2,
        )
        
        rgb = heatmap.generate_rgb(state)
        
        assert rgb.shape == (8, 8, 3)
        assert rgb.dtype == np.uint8
    
    def test_generate_rgb_without_optional(self):
        """Test RGB generation with only activation."""
        heatmap = SynapseHeatmap()
        
        state = LayerState(activation=torch.randn(4, 64))
        rgb = heatmap.generate_rgb(state)
        
        assert rgb.shape == (8, 8, 3)
        # Green and blue should be zero
        assert rgb[:, :, 1].max() == 0  # Green
        assert rgb[:, :, 2].max() == 0  # Blue


class TestDynamicsCapture:
    """Tests for DynamicsCapture class."""
    
    def test_record_step(self):
        """Test recording a forward step."""
        capture = DynamicsCapture()
        
        h_old = torch.zeros(4, 64)
        h_new = torch.randn(4, 64)
        
        capture.record_step("layer_0", h_new, h_old)
        
        assert len(capture.history) == 1
        assert "layer_0" in capture.history[0]
        assert capture.history[0]["layer_0"].velocity is not None
    
    def test_record_equilibrium(self):
        """Test recording free and nudged equilibrium."""
        capture = DynamicsCapture()
        
        free_state = {"layer_0": torch.randn(4, 64)}
        nudged_state = {"layer_0": torch.randn(4, 64)}
        
        capture.record_free_equilibrium(free_state)
        capture.record_nudged_equilibrium(nudged_state)
        
        assert capture.free_equilibrium is not None
        assert capture.nudged_equilibrium is not None
    
    def test_history_limit(self):
        """Test that history is limited."""
        capture = DynamicsCapture(max_history=5)
        
        for i in range(10):
            capture.record_step(f"step_{i}", torch.randn(4, 64), torch.zeros(4, 64))
        
        assert len(capture.history) == 5
    
    def test_clear(self):
        """Test clearing capture."""
        capture = DynamicsCapture()
        capture.record_step("layer_0", torch.randn(4, 64), torch.zeros(4, 64))
        
        capture.clear()
        
        assert len(capture.history) == 0


class TestObservatoryMetrics:
    """Tests for ObservatoryMetrics class."""
    
    def test_settling_time(self):
        """Test settling time computation."""
        metrics = ObservatoryMetrics(velocity_threshold=0.01)
        
        # Velocity decreasing over time
        history = [
            torch.ones(4, 64) * 0.5,
            torch.ones(4, 64) * 0.1,
            torch.ones(4, 64) * 0.005,  # Below threshold
        ]
        
        t = metrics.compute_settling_time(history)
        assert t == 3
    
    def test_nudge_depth(self):
        """Test nudge depth computation."""
        metrics = ObservatoryMetrics(nudge_visibility_threshold=0.01)
        
        layer_nudges = {
            "layer_0": torch.ones(4, 64) * 0.1,
            "layer_1": torch.ones(4, 64) * 0.05,
            "layer_2": torch.ones(4, 64) * 0.001,  # Below threshold
        }
        
        depth = metrics.compute_nudge_depth(layer_nudges, ["layer_0", "layer_1", "layer_2"])
        assert depth == 2  # layer_1 and layer_0 visible
    
    def test_skip_ratio(self):
        """Test FLOP savings tracking."""
        metrics = ObservatoryMetrics()
        
        for _ in range(7):
            metrics.record_update(was_skipped=False)
        for _ in range(3):
            metrics.record_update(was_skipped=True)
        
        assert metrics.skip_ratio == 0.3
        assert metrics.flop_savings_percent == 30.0
    
    def test_summary(self):
        """Test summary dict generation."""
        metrics = ObservatoryMetrics()
        summary = metrics.summary()
        
        assert 'mean_settling_time' in summary
        assert 'mean_nudge_depth' in summary
        assert 'flop_savings_percent' in summary


class TestHeadlessRenderer:
    """Tests for HeadlessRenderer class."""
    
    def test_render_frame(self):
        """Test headless frame rendering."""
        renderer = HeadlessRenderer()
        renderer.init()
        
        heatmaps = {
            "layer_0": np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8),
            "layer_1": np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8),
        }
        
        renderer.render_frame(heatmaps, metrics={}, epoch=0, step=0)
        
        assert len(renderer.frames) == 1
        assert renderer.frames[0].shape[1] == 128  # Two 64-wide heatmaps


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
