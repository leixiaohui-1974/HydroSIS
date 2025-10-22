"""Tests for the visualization capabilities of HydroSIS."""
from __future__ import annotations

import unittest
from typing import Dict, List
from unittest.mock import patch, MagicMock

from hydrosis.visualization.charts import (
    create_hydrograph_chart,
    create_comparison_chart,
    create_metrics_chart,
    create_dashboard_html,
    create_map_visualization,
)



class VisualizationTests(unittest.TestCase):
    """Test visualization functionality."""

    def setUp(self) -> None:
        """Set up test data."""
        self.sample_data = {
            "Series 1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "Series 2": [0.5, 1.5, 2.5, 3.5, 4.5],
            "Series 3": [0.0, 1.0, 2.0, 3.0, 4.0],
        }
        
        self.baseline_data = {
            "S1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "S2": [0.5, 1.5, 2.5, 3.5, 4.5],
            "S3": [0.0, 1.0, 2.0, 3.0, 4.0],
        }
        
        self.scenario_data = {
            "Scenario 1": {
                "S1": [1.2, 2.2, 3.2, 4.2, 5.2],
                "S2": [0.7, 1.7, 2.7, 3.7, 4.7],
                "S3": [0.2, 1.2, 2.2, 3.2, 4.2],
            },
            "Scenario 2": {
                "S1": [0.8, 1.8, 2.8, 3.8, 4.8],
                "S2": [0.3, 1.3, 2.3, 3.3, 4.3],
                "S3": [0.1, 1.1, 2.1, 3.1, 4.1],
            },
        }
        
        self.metrics_data = {
            "Model 1": {"nse": 0.85, "rmse": 2.5, "mae": 1.8},
            "Model 2": {"nse": 0.78, "rmse": 3.2, "mae": 2.1},
            "Model 3": {"nse": 0.92, "rmse": 1.8, "mae": 1.2},
        }

    @patch('hydrosis.visualization.charts.HAS_PLOTLY', True)
    @patch('hydrosis.visualization.charts.HAS_MPL', True)
    def test_create_hydrograph_chart_with_plotly(self) -> None:
        """Test creating a hydrograph chart with Plotly."""
        with patch('hydrosis.visualization.charts._create_plotly_hydrograph') as mock_plotly:
            mock_plotly.return_value = "<html>Mock Plotly Chart</html>"
            
            result = create_hydrograph_chart(self.sample_data, use_plotly=True)
            
            self.assertEqual(result, "<html>Mock Plotly Chart</html>")
            mock_plotly.assert_called_once()

    @patch('hydrosis.visualization.charts.HAS_PLOTLY', False)
    @patch('hydrosis.visualization.charts.HAS_MPL', True)
    def test_create_hydrograph_chart_with_matplotlib(self) -> None:
        """Test creating a hydrograph chart with Matplotlib."""
        with patch('hydrosis.visualization.charts._create_mpl_hydrograph') as mock_mpl:
            mock_mpl.return_value = b"Mock Matplotlib Image"
            
            result = create_hydrograph_chart(self.sample_data, use_plotly=False)
            
            self.assertEqual(result, b"Mock Matplotlib Image")
            mock_mpl.assert_called_once()

    @patch('hydrosis.visualization.charts.HAS_PLOTLY', False)
    @patch('hydrosis.visualization.charts.HAS_MPL', False)
    def test_create_hydrograph_chart_no_library(self) -> None:
        """Test creating a hydrograph chart with no visualization library."""
        with self.assertRaises(ImportError):
            create_hydrograph_chart(self.sample_data)

    @patch('hydrosis.visualization.charts.HAS_PLOTLY', True)
    def test_create_comparison_chart(self) -> None:
        """Test creating a comparison chart."""
        with patch('hydrosis.visualization.charts.create_hydrograph_chart') as mock_chart:
            mock_chart.return_value = "<html>Mock Comparison Chart</html>"
            
            result = create_comparison_chart(
                self.baseline_data,
                self.scenario_data,
                subbasin_id="S1"
            )
            
            self.assertEqual(result, "<html>Mock Comparison Chart</html>")
            mock_chart.assert_called_once()

    @patch('hydrosis.visualization.charts.HAS_PLOTLY', True)
    def test_create_metrics_chart(self) -> None:
        """Test creating a metrics chart."""
        with patch('hydrosis.visualization.charts._create_plotly_metrics') as mock_plotly:
            mock_plotly.return_value = "<html>Mock Metrics Chart</html>"
            
            result = create_metrics_chart(self.metrics_data)
            
            self.assertEqual(result, "<html>Mock Metrics Chart</html>")
            mock_plotly.assert_called_once()

    def test_create_dashboard_html(self) -> None:
        """Test creating a dashboard HTML."""
        hydrograph_html = "<div>Hydrograph</div>"
        comparison_html = "<div>Comparison</div>"
        metrics_html = "<div>Metrics</div>"
        map_html = "<div>Map</div>"
        
        result = create_dashboard_html(
            hydrograph_html,
            comparison_html,
            metrics_html,
            map_html=map_html,
            title="Test Dashboard"
        )
        
        self.assertIn("Test Dashboard", result)
        self.assertIn(hydrograph_html, result)
        self.assertIn(comparison_html, result)
        self.assertIn(metrics_html, result)
        self.assertIn(map_html, result)
        self.assertIn("grid-template-columns", result)

    def test_create_dashboard_html_without_map(self) -> None:
        """Test creating a dashboard HTML without a map."""
        hydrograph_html = "<div>Hydrograph</div>"
        comparison_html = "<div>Comparison</div>"
        metrics_html = "<div>Metrics</div>"
        
        result = create_dashboard_html(
            hydrograph_html,
            comparison_html,
            metrics_html,
            title="Test Dashboard"
        )
        
        self.assertIn("Test Dashboard", result)
        self.assertIn(hydrograph_html, result)
        self.assertIn(comparison_html, result)
        self.assertIn(metrics_html, result)
        self.assertNotIn("流域地图", result)

    @patch('hydrosis.visualization.charts.HAS_PLOTLY', True)
    def test_create_map_visualization(self) -> None:
        """Test creating a map visualization."""
        geojson_data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": "test_feature",
                    "properties": {"name": "Test"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
                    }
                }
            ]
        }
        
        with patch('hydrosis.visualization.charts.pio.to_html') as mock_plotly:
            mock_plotly.return_value = "<html>Mock Map</html>"
            
            result = create_map_visualization(geojson_data)
            
            self.assertEqual(result, "<html>Mock Map</html>")
            mock_plotly.assert_called_once()

    @patch('hydrosis.visualization.charts.HAS_PLOTLY', False)
    def test_create_map_visualization_no_plotly(self) -> None:
        """Test creating a map visualization without Plotly."""
        geojson_data = {"type": "FeatureCollection", "features": []}
        
        with self.assertRaises(ImportError):
            create_map_visualization(geojson_data)


if __name__ == "__main__":
    unittest.main()
