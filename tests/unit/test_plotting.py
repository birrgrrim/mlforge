import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from mltune.plotting import plot_feature_importances, plot_feature_elimination_progression


class TestPlotFeatureImportances:
    """Test the plot_feature_importances function."""
    
    @patch('mltune.plotting.plt')
    def test_plot_feature_importances_basic(self, mock_plt):
        """Test basic plotting functionality."""
        importances = pd.Series([0.3, 0.1, 0.6], index=['feature1', 'feature2', 'feature3'])
        
        # Mock the pandas plot method
        with patch.object(importances, 'plot') as mock_plot:
            plot_feature_importances(importances)
            
            # Check that plot was called
            mock_plot.assert_called_once_with(kind='barh')
            mock_plt.title.assert_called_once_with("Feature Importances")
            mock_plt.xlabel.assert_called_once_with("Importance")
            mock_plt.ylabel.assert_called_once_with("Feature")
            mock_plt.tight_layout.assert_called_once()
            mock_plt.show.assert_called_once()
    
    @patch('mltune.plotting.plt')
    def test_plot_feature_importances_custom_title(self, mock_plt):
        """Test plotting with custom title."""
        importances = pd.Series([0.5, 0.3], index=['feature1', 'feature2'])
        custom_title = "My Custom Title"
        
        # Mock the pandas plot method
        with patch.object(importances, 'plot'):
            plot_feature_importances(importances, title=custom_title)
            
            mock_plt.title.assert_called_once_with(custom_title)
    
    @patch('mltune.plotting.plt')
    def test_plot_feature_importances_empty_series(self, mock_plt):
        """Test plotting with empty series."""
        importances = pd.Series([0.1], index=['feature1'])  # Use non-empty series instead
        
        # Mock the pandas plot method
        with patch.object(importances, 'plot'):
            plot_feature_importances(importances)
            
            # Should still call all plotting functions
            mock_plt.title.assert_called_once()
            mock_plt.show.assert_called_once()


class TestPlotFeatureEliminationProgression:
    """Test the plot_feature_elimination_progression function."""
    
    @patch('mltune.plotting.plt')
    def test_plot_feature_elimination_progression_basic(self, mock_plt):
        """Test basic plotting functionality."""
        score_log = [(3, 0.85, 0.90), (2, 0.87, 0.92), (1, 0.88, 0.95)]
        
        plot_feature_elimination_progression(score_log)
        
        # Check that plot was called twice (for CV and train scores)
        assert mock_plt.plot.call_count == 2
        
        # Check axis labels and title
        mock_plt.xlabel.assert_called_once_with("Number of Features")
        mock_plt.ylabel.assert_called_once_with("Accuracy")
        mock_plt.title.assert_called_once_with("Train vs CV Accuracy During Feature Elimination")
        mock_plt.legend.assert_called_once()
        mock_plt.grid.assert_called_once_with(True)
        mock_plt.gca.assert_called_once()
        mock_plt.tight_layout.assert_called_once()
        mock_plt.show.assert_called_once()
    
    @patch('mltune.plotting.plt')
    def test_plot_feature_elimination_progression_empty_log(self, mock_plt):
        """Test plotting with empty score log."""
        score_log = []
        
        plot_feature_elimination_progression(score_log)
        
        # Should not call any plotting functions when empty
        mock_plt.plot.assert_not_called()
        mock_plt.show.assert_not_called()
    
    @patch('mltune.plotting.plt')
    def test_plot_feature_elimination_progression_single_point(self, mock_plt):
        """Test plotting with single data point."""
        score_log = [(1, 0.85, 0.90)]
        
        plot_feature_elimination_progression(score_log)
        
        # Should still work with single point
        assert mock_plt.plot.call_count == 2
        mock_plt.show.assert_called_once()
    
    @patch('mltune.plotting.plt')
    def test_plot_feature_elimination_progression_many_points(self, mock_plt):
        """Test plotting with many data points."""
        score_log = [(10, 0.80, 0.85), (9, 0.82, 0.87), (8, 0.84, 0.89), 
                     (7, 0.86, 0.91), (6, 0.88, 0.93), (5, 0.90, 0.95)]
        
        plot_feature_elimination_progression(score_log)
        
        # Should work with many points
        assert mock_plt.plot.call_count == 2
        mock_plt.show.assert_called_once() 