"""High-level analysis utilities exposed for external integrations."""

from .channel_profile import (
    ChannelProfileConfig,
    PreprocessResult,
    ChannelProfileResult,
    apply_monotonic_corrections,
    build_zone_grid,
    extract_centerline,
    generate_channel_profiles,
    load_cross_sections,
    preprocess_cross_sections,
    run_channel_profile_model,
)

from .metrics import (
    nash_sutcliffe_efficiency,
    root_mean_square_error,
    mean_absolute_error,
    percent_bias,
    kling_gupta_efficiency,
    log_nash_sutcliffe,
    volume_error,
    peak_error,
    calculate_metrics,
    get_metric_interpretation,
    nse,
    rmse,
    mae,
    pbias,
    kge,
)

from .sensitivity import (
    one_at_a_time_sensitivity,
    sobol_sensitivity,
    morris_screening,
    visualize_sensitivity_results,
)

from .uncertainty import (
    monte_carlo_sampling,
    latin_hypercube_sampling,
    monte_carlo_analysis,
    glue_analysis,
)

from .calibration_optimization import (
    CalibrationResult,
    sce_ua_optimization,
    differential_evolution_optimization,
    calibrate_model,
)

from .visualization import (
    plot_hydrograph_comparison,
    plot_scatter,
    plot_uncertainty_envelope,
    plot_parameter_distributions,
    plot_convergence_history,
    plot_tornado_sensitivity,
    plot_multi_metric_comparison,
    create_analysis_report_figures,
)

__all__ = [
    # Channel profile analysis
    "ChannelProfileConfig",
    "PreprocessResult",
    "ChannelProfileResult",
    "generate_channel_profiles",
    "preprocess_cross_sections",
    "run_channel_profile_model",
    "apply_monotonic_corrections",
    "build_zone_grid",
    "extract_centerline",
    "load_cross_sections",
    # Performance metrics
    "nash_sutcliffe_efficiency",
    "root_mean_square_error",
    "mean_absolute_error",
    "percent_bias",
    "kling_gupta_efficiency",
    "log_nash_sutcliffe",
    "volume_error",
    "peak_error",
    "calculate_metrics",
    "get_metric_interpretation",
    "nse",
    "rmse",
    "mae",
    "pbias",
    "kge",
    # Sensitivity analysis
    "one_at_a_time_sensitivity",
    "sobol_sensitivity",
    "morris_screening",
    "visualize_sensitivity_results",
    # Uncertainty analysis
    "monte_carlo_sampling",
    "latin_hypercube_sampling",
    "monte_carlo_analysis",
    "glue_analysis",
    # Parameter calibration
    "CalibrationResult",
    "sce_ua_optimization",
    "differential_evolution_optimization",
    "calibrate_model",
    # Visualization
    "plot_hydrograph_comparison",
    "plot_scatter",
    "plot_uncertainty_envelope",
    "plot_parameter_distributions",
    "plot_convergence_history",
    "plot_tornado_sensitivity",
    "plot_multi_metric_comparison",
    "create_analysis_report_figures",
]
