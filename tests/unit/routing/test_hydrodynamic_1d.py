"""Unit tests for Hydrodynamic 1D routing wrapper (deprecated)."""
import pytest
import warnings


class TestHydrodynamic1DWrapper:
    """Test the deprecated hydrodynamic_1d wrapper module."""

    def test_import_triggers_deprecation_warning(self):
        """Test that importing hydrodynamic_1d triggers deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Import the module
            from hydrosis.routing import hydrodynamic_1d

            # Check that a warning was raised
            assert len(w) >= 1
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
            assert any("deprecated" in str(warning.message).lower() for warning in w)
            assert any("hydrodynamics" in str(warning.message).lower() for warning in w)

    def test_exports_are_available(self):
        """Test that all expected exports are available."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            from hydrosis.routing import hydrodynamic_1d

            # Check that expected classes are available
            assert hasattr(hydrodynamic_1d, 'SaintVenantSolver')
            assert hasattr(hydrodynamic_1d, 'RiverReach')
            assert hasattr(hydrodynamic_1d, 'BoundaryCondition')
            assert hasattr(hydrodynamic_1d, 'HydrodynamicRoutingModel')

    def test_classes_are_importable(self):
        """Test that classes can be imported from the wrapper."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            from hydrosis.routing.hydrodynamic_1d import (
                SaintVenantSolver,
                RiverReach,
                BoundaryCondition,
                HydrodynamicRoutingModel
            )

            # Verify they are classes/types
            assert SaintVenantSolver is not None
            assert RiverReach is not None
            assert BoundaryCondition is not None
            assert HydrodynamicRoutingModel is not None

    def test_all_list_contains_expected_exports(self):
        """Test that __all__ contains the expected exports."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            from hydrosis.routing import hydrodynamic_1d

            assert hasattr(hydrodynamic_1d, '__all__')
            expected_exports = [
                'SaintVenantSolver',
                'RiverReach',
                'BoundaryCondition',
                'HydrodynamicRoutingModel'
            ]

            for export in expected_exports:
                assert export in hydrodynamic_1d.__all__
