"""优化算法模块

提供统一的优化接口，支持多种优化算法用于模型校准。
"""
from __future__ import annotations

from typing import Callable, Dict, List, Any, Optional, Tuple
import numpy as np

try:
    from scipy.optimize import differential_evolution, minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def calibrate_model(
    objective_function: Callable,
    param_bounds: Dict[str, List[float]],
    maximize: bool = True,
    method: str = "differential_evolution",
    seed: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """校准模型参数

    使用指定的优化算法找到最优参数。

    Parameters
    ----------
    objective_function : callable
        目标函数，接受参数数组返回标量值
    param_bounds : dict
        参数边界 {param_name: [min, max]}
    maximize : bool, default=True
        是否最大化目标函数
    method : str, default="differential_evolution"
        优化算法: "differential_evolution", "nelder_mead", "powell"
    seed : int, optional
        随机种子
    **kwargs
        传递给优化算法的额外参数

    Returns
    -------
    dict
        优化结果，包含:
        - best_params: 最优参数字典
        - best_score: 最优得分
        - success: 是否成功
        - message: 消息
        - n_evaluations: 评估次数

    Example
    -------
    >>> def objective(params):
    ...     return -(params['x']**2 + params['y']**2)  # 最小化
    >>> result = calibrate_model(
    ...     objective,
    ...     {'x': [-10, 10], 'y': [-10, 10]},
    ...     maximize=False
    ... )
    """
    if not HAS_SCIPY:
        raise ImportError(
            "scipy is required for model calibration. "
            "Install it with: pip install scipy"
        )

    # 提取参数名和边界
    param_names = list(param_bounds.keys())
    bounds = [param_bounds[name] for name in param_names]

    # 创建包装函数（数组 -> 字典）
    evaluation_count = [0]  # 使用列表以便在闭包中修改

    def wrapped_objective(param_array):
        """包装目标函数，转换参数格式"""
        evaluation_count[0] += 1
        params_dict = {name: value for name, value in zip(param_names, param_array)}
        score = objective_function(params_dict)

        # 如果是最大化，取负值
        if maximize:
            return -score
        return score

    # 选择优化算法
    if method in ["differential_evolution", "de"]:
        # 差分进化算法（全局优化）
        de_params = {
            'maxiter': kwargs.get('maxiter', 100),
            'popsize': kwargs.get('popsize', 15),
            'atol': kwargs.get('atol', 1e-4),
            'tol': kwargs.get('tol', 0.01),
            'seed': seed,
            'workers': kwargs.get('workers', 1),
            'updating': kwargs.get('updating', 'deferred'),
            'disp': kwargs.get('disp', False)
        }

        result = differential_evolution(
            wrapped_objective,
            bounds,
            **de_params
        )

        best_params = {name: value for name, value in zip(param_names, result.x)}
        best_score = -result.fun if maximize else result.fun

        return {
            'best_params': best_params,
            'best_score': best_score,
            'success': result.success,
            'message': result.message,
            'n_evaluations': evaluation_count[0]
        }

    elif method in ["nelder_mead", "nelder-mead", "simplex"]:
        # Nelder-Mead单纯形法（局部优化）
        # 需要初始猜测
        x0 = kwargs.get('x0', None)
        if x0 is None:
            # 使用边界中点作为初始值
            x0 = [(b[0] + b[1]) / 2 for b in bounds]

        nm_params = {
            'maxiter': kwargs.get('maxiter', 1000),
            'xatol': kwargs.get('xatol', 1e-4),
            'fatol': kwargs.get('fatol', 1e-4),
            'disp': kwargs.get('disp', False)
        }

        result = minimize(
            wrapped_objective,
            x0,
            method='Nelder-Mead',
            options=nm_params
        )

        best_params = {name: value for name, value in zip(param_names, result.x)}
        best_score = -result.fun if maximize else result.fun

        return {
            'best_params': best_params,
            'best_score': best_score,
            'success': result.success,
            'message': result.message if hasattr(result, 'message') else 'Optimization completed',
            'n_evaluations': evaluation_count[0]
        }

    elif method in ["powell"]:
        # Powell方法（局部优化）
        x0 = kwargs.get('x0', None)
        if x0 is None:
            x0 = [(b[0] + b[1]) / 2 for b in bounds]

        powell_params = {
            'maxiter': kwargs.get('maxiter', 1000),
            'xtol': kwargs.get('xtol', 1e-4),
            'ftol': kwargs.get('ftol', 1e-4),
            'disp': kwargs.get('disp', False)
        }

        result = minimize(
            wrapped_objective,
            x0,
            method='Powell',
            options=powell_params
        )

        best_params = {name: value for name, value in zip(param_names, result.x)}
        best_score = -result.fun if maximize else result.fun

        return {
            'best_params': best_params,
            'best_score': best_score,
            'success': result.success,
            'message': result.message if hasattr(result, 'message') else 'Optimization completed',
            'n_evaluations': evaluation_count[0]
        }

    else:
        raise ValueError(
            f"Unknown optimization method: {method}. "
            f"Available methods: differential_evolution, nelder_mead, powell"
        )


__all__ = ['calibrate_model']
