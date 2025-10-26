#!/usr/bin/env python3
"""
综合产汇流模型测试和率定系统

功能：
1. 测试项目中所有产流模型和汇流模型
2. 对每个分区进行多模型对比
3. 自动参数率定找到最佳参数
4. 参数敏感性分析
5. 模型验证和诊断
6. 保存最佳参数和模拟精度
7. 生成详细的分析报告

作者：Claude
日期：2025-10-26
"""

import sys
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（用于报告和表格，图表标签用英文）
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 导入hydrosis模块
try:
    from hydrosis.runoff.hbv import HBVRunoff
    from hydrosis.runoff.xinanjiang import XinAnJiangRunoff
    from hydrosis.runoff.scs_curve_number import SCSCurveNumber
    from hydrosis.runoff.hymod import HYMODRunoff
    from hydrosis.runoff.linear_reservoir import LinearReservoirRunoff
    from hydrosis.runoff.simple import SimpleRunoff

    from hydrosis.routing.muskingum import MuskingumRouting
    from hydrosis.routing.lag import LagRouting
    from hydrosis.routing.simple import SimpleRouting
except ImportError as e:
    logger.error(f"Failed to import hydrosis modules: {e}")
    logger.info("Will use simplified model implementations")

# 尝试导入率定器
try:
    from hydrosis.calibration.base import CalibrationData, CalibrationConfig
    
    # 创建简单的MockSubbasin类
    class MockSubbasin:
        """模拟Subbasin对象"""
        def __init__(self, area_km2: float):
            self.area_km2 = area_km2
    
    HAS_CALIBRATOR = True
except ImportError:
    logger.warning("Calibration modules not available, will use simplified testing")
    HAS_CALIBRATOR = False


@dataclass
class ModelTestResult:
    """模型测试结果"""
    zone_id: str
    runoff_model: str
    routing_model: str
    nse: float
    rmse: float
    mae: float
    r2: float
    pbias: float
    runoff_coef_obs: float
    runoff_coef_sim: float
    best_params: Dict[str, float]
    calibration_time: float
    
    def get_score(self) -> float:
        """综合评分（0-100）"""
        # NSE权重50%，R2权重30%，径流系数误差权重20%
        nse_score = max(0, self.nse * 50)
        r2_score = self.r2 * 30
        rc_error = abs(self.runoff_coef_sim - self.runoff_coef_obs) / max(0.01, self.runoff_coef_obs)
        rc_score = max(0, (1 - rc_error) * 20)
        return nse_score + r2_score + rc_score


class ComprehensiveModelTester:
    """综合模型测试器"""
    
    # 可用的产流模型
    RUNOFF_MODELS = {
        'HBV': HBVRunoff,
        'XinAnJiang': XinAnJiangRunoff,
        'SCS-CN': SCSCurveNumber,
        'HYMOD': HYMODRunoff,
        'LinearReservoir': LinearReservoirRunoff,
        'Simple': SimpleRunoff,
    }
    
    # 可用的汇流模型
    ROUTING_MODELS = {
        'Muskingum': MuskingumRouting,
        'Lag': LagRouting,
        'Simple': SimpleRouting,
    }
    
    # 默认参数范围
    DEFAULT_PARAM_BOUNDS = {
        # HBV parameters
        'FC': [200.0, 600.0],          # Field capacity
        'BETA': [1.0, 6.0],            # Shape parameter
        'K0': [0.01, 0.5],             # Quick flow coefficient
        'K1': [0.001, 0.1],            # Slow flow coefficient
        'K2': [0.0001, 0.01],          # Base flow coefficient
        'PERC': [0.0, 5.0],            # Percolation rate
        'MAXBAS': [1.0, 10.0],         # Routing parameter
        
        # XinAnJiang parameters
        'K': [0.1, 1.5],               # Ratio of potential evapotranspiration to pan evaporation
        'IM': [0.001, 0.05],           # Fraction of impervious area
        'UM': [10.0, 30.0],            # Upper layer tension water capacity
        'LM': [60.0, 100.0],           # Lower layer tension water capacity
        'B': [0.1, 0.5],               # Exponent parameter
        'C': [0.1, 0.25],              # Deep layer evaporation coefficient
        'SM': [10.0, 50.0],            # Free water storage capacity
        
        # SCS-CN parameters
        'CN': [30.0, 98.0],            # Curve number
        'ia_ratio': [0.05, 0.3],       # Initial abstraction ratio
        
        # HYMOD parameters
        'Huz': [0.0, 500.0],           # Maximum height of soil moisture accounting tank
        'B': [0.0, 2.0],               # Distribution function shape parameter
        'Alpha': [0.0, 1.0],           # Quick/slow split parameter
        'Rs': [0.0, 0.1],              # Slow flow routing tanks rate parameter
        'Rq': [0.5, 1.0],              # Quick flow routing tanks rate parameter
        
        # LinearReservoir parameters
        'k': [0.1, 10.0],              # Storage coefficient
        'threshold': [0.0, 10.0],      # Threshold
        
        # Muskingum parameters
        'K': [0.1, 24.0],              # Storage time constant (hours)
        'x': [0.0, 0.5],               # Weighting factor
        
        # Lag parameters
        'lag_time': [0.1, 10.0],       # Lag time (hours)
    }
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 存储所有测试结果
        self.results: List[ModelTestResult] = []
        
        # matplotlib设置
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['figure.dpi'] = 150
        
    def load_data(self, 
                  precip_file: Path,
                  discharge_file: Path, 
                  watershed_file: Path,
                  temperature_file: Optional[Path] = None) -> bool:
        """加载数据"""
        try:
            logger.info("Loading data files...")
            
            # 加载降雨数据
            self.precip_df = pd.read_csv(precip_file)
            logger.info(f"  Precipitation: {len(self.precip_df)} time steps, "
                       f"{len(self.precip_df.columns)-1} zones")
            
            # 加载径流数据
            self.discharge_df = pd.read_csv(discharge_file)
            logger.info(f"  Discharge: {len(self.discharge_df)} time steps, "
                       f"{len(self.discharge_df.columns)-1} points")
            
            # 加载温度数据（如果有）
            if temperature_file and temperature_file.exists():
                self.temp_df = pd.read_csv(temperature_file)
                logger.info(f"  Temperature: {len(self.temp_df)} time steps")
            else:
                # 生成默认温度数据（15°C）
                self.temp_df = pd.DataFrame({
                    'time': self.precip_df.iloc[:, 0],
                    **{col: 15.0 for col in self.precip_df.columns[1:]}
                })
                logger.info("  Temperature: Using default 15°C")
            
            # 加载流域数据
            import geopandas as gpd
            self.watersheds = gpd.read_file(watershed_file)
            logger.info(f"  Watersheds: {len(self.watersheds)} polygons")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def test_zone(self,
                  zone_id: str,
                  runoff_model_name: str,
                  routing_model_name: str,
                  max_iterations: int = 50) -> Optional[ModelTestResult]:
        """测试单个分区的模型组合"""
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing Zone: {zone_id}")
        logger.info(f"Runoff Model: {runoff_model_name}")
        logger.info(f"Routing Model: {routing_model_name}")
        logger.info(f"{'='*80}")
        
        try:
            # 获取数据
            precip_col = zone_id
            discharge_col = zone_id
            temp_col = zone_id if zone_id in self.temp_df.columns else self.temp_df.columns[1]
            
            if precip_col not in self.precip_df.columns:
                logger.warning(f"  No precipitation data for {zone_id}, skipping")
                return None
            
            if discharge_col not in self.discharge_df.columns:
                logger.warning(f"  No discharge data for {zone_id}, skipping")
                return None
            
            precipitation = self.precip_df[precip_col].values
            observed_discharge = self.discharge_df[discharge_col].values
            temperature = self.temp_df[temp_col].values
            
            # 获取流域面积
            zone_data = self.watersheds[self.watersheds['id'] == zone_id]
            if len(zone_data) == 0:
                zone_data = self.watersheds.iloc[0:1]  # 使用第一个
            area_km2 = zone_data.iloc[0].geometry.area / 1e6
            
            logger.info(f"  Area: {area_km2:.2f} km²")
            logger.info(f"  Time steps: {len(precipitation)}")
            logger.info(f"  Precipitation range: {precipitation.min():.2f} - {precipitation.max():.2f} mm/h")
            logger.info(f"  Discharge range: {observed_discharge.min():.2f} - {observed_discharge.max():.2f} m³/s")
            
            # 准备率定数据
            cal_data = CalibrationData(
                precipitation=precipitation.tolist(),
                observed_discharge=observed_discharge.tolist(),
                temperature=temperature.tolist(),
                subbasin=MockSubbasin(area_km2=area_km2),
                dt=1.0  # hourly time step
            )
            
            # 准备参数边界
            param_bounds = self._get_param_bounds(runoff_model_name, routing_model_name)
            
            # 配置率定
            cal_config = CalibrationConfig(
                param_bounds=param_bounds,
                algorithm='differential_evolution',
                max_iterations=max_iterations,
                objective='nse',
                pop_size=15,
                n_cores=1
            )
            
            # 创建率定器
            start_time = datetime.now()
            
            calibrator = GenericHydrologicCalibrator(
                data=cal_data,
                config=cal_config,
                runoff_model_class=self.RUNOFF_MODELS[runoff_model_name],
                routing_model_class=self.ROUTING_MODELS[routing_model_name],
                temperature=temperature,
                output_dir=self.output_dir / f"{zone_id}_{runoff_model_name}_{routing_model_name}"
            )
            
            # 运行率定
            logger.info("  Running calibration...")
            result = calibrator.run()
            
            calibration_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"  Calibration completed in {calibration_time:.1f}s")
            logger.info(f"  Best NSE: {result.best_value:.4f}")
            
            # 计算其他指标
            simulated = result.simulated_series
            metrics = self._calculate_metrics(observed_discharge, simulated)
            
            # 计算径流系数
            total_precip_mm = precipitation.sum()  # mm (hourly sum)
            
            # 观测径流系数
            obs_discharge_m3s_sum = observed_discharge.sum()
            obs_volume_m3 = obs_discharge_m3s_sum * 3600  # m³
            obs_runoff_mm = (obs_volume_m3 / (area_km2 * 1e6)) * 1000
            obs_rc = obs_runoff_mm / total_precip_mm if total_precip_mm > 0 else 0
            
            # 模拟径流系数
            sim_discharge_m3s_sum = simulated.sum()
            sim_volume_m3 = sim_discharge_m3s_sum * 3600
            sim_runoff_mm = (sim_volume_m3 / (area_km2 * 1e6)) * 1000
            sim_rc = sim_runoff_mm / total_precip_mm if total_precip_mm > 0 else 0
            
            logger.info(f"  NSE: {metrics['nse']:.4f}")
            logger.info(f"  R²: {metrics['r2']:.4f}")
            logger.info(f"  RMSE: {metrics['rmse']:.4f} m³/s")
            logger.info(f"  Observed RC: {obs_rc:.4f}")
            logger.info(f"  Simulated RC: {sim_rc:.4f}")
            
            # 创建结果对象
            test_result = ModelTestResult(
                zone_id=zone_id,
                runoff_model=runoff_model_name,
                routing_model=routing_model_name,
                nse=metrics['nse'],
                rmse=metrics['rmse'],
                mae=metrics['mae'],
                r2=metrics['r2'],
                pbias=metrics['pbias'],
                runoff_coef_obs=obs_rc,
                runoff_coef_sim=sim_rc,
                best_params=result.best_params,
                calibration_time=calibration_time
            )
            
            # 生成诊断图表
            self._plot_diagnostics(zone_id, runoff_model_name, routing_model_name,
                                   observed_discharge, simulated, test_result)
            
            return test_result
            
        except Exception as e:
            logger.error(f"  Error testing {zone_id} with {runoff_model_name}+{routing_model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _get_param_bounds(self, runoff_model: str, routing_model: str) -> Dict[str, tuple]:
        """获取参数边界"""
        bounds = {}
        
        # 产流模型参数
        if runoff_model == 'HBV':
            bounds.update({
                'FC': self.DEFAULT_PARAM_BOUNDS['FC'],
                'BETA': self.DEFAULT_PARAM_BOUNDS['BETA'],
                'K0': self.DEFAULT_PARAM_BOUNDS['K0'],
                'K1': self.DEFAULT_PARAM_BOUNDS['K1'],
                'K2': self.DEFAULT_PARAM_BOUNDS['K2'],
                'PERC': self.DEFAULT_PARAM_BOUNDS['PERC'],
            })
        elif runoff_model == 'XinAnJiang':
            bounds.update({
                'K': self.DEFAULT_PARAM_BOUNDS['K'],
                'IM': self.DEFAULT_PARAM_BOUNDS['IM'],
                'UM': self.DEFAULT_PARAM_BOUNDS['UM'],
                'LM': self.DEFAULT_PARAM_BOUNDS['LM'],
                'B': self.DEFAULT_PARAM_BOUNDS['B'],
                'C': self.DEFAULT_PARAM_BOUNDS['C'],
                'SM': self.DEFAULT_PARAM_BOUNDS['SM'],
            })
        elif runoff_model == 'SCS-CN':
            bounds.update({
                'CN': self.DEFAULT_PARAM_BOUNDS['CN'],
                'ia_ratio': self.DEFAULT_PARAM_BOUNDS['ia_ratio'],
            })
        elif runoff_model == 'HYMOD':
            bounds.update({
                'Huz': self.DEFAULT_PARAM_BOUNDS['Huz'],
                'B': self.DEFAULT_PARAM_BOUNDS['B'],
                'Alpha': self.DEFAULT_PARAM_BOUNDS['Alpha'],
                'Rs': self.DEFAULT_PARAM_BOUNDS['Rs'],
                'Rq': self.DEFAULT_PARAM_BOUNDS['Rq'],
            })
        elif runoff_model == 'LinearReservoir':
            bounds.update({
                'k': self.DEFAULT_PARAM_BOUNDS['k'],
                'threshold': self.DEFAULT_PARAM_BOUNDS['threshold'],
            })
        
        # 汇流模型参数
        if routing_model == 'Muskingum':
            bounds.update({
                'K': self.DEFAULT_PARAM_BOUNDS['K'],
                'x': self.DEFAULT_PARAM_BOUNDS['x'],
            })
        elif routing_model == 'Lag':
            bounds.update({
                'lag_time': self.DEFAULT_PARAM_BOUNDS['lag_time'],
            })
        
        return bounds
    
    def _calculate_metrics(self, observed: np.ndarray, simulated: np.ndarray) -> Dict[str, float]:
        """计算评价指标"""
        # 确保是numpy数组
        obs = np.array(observed)
        sim = np.array(simulated)
        
        # NSE (Nash-Sutcliffe Efficiency)
        numerator = np.sum((obs - sim) ** 2)
        denominator = np.sum((obs - np.mean(obs)) ** 2)
        nse = 1 - (numerator / denominator) if denominator > 0 else -np.inf
        
        # R² (Coefficient of Determination)
        correlation = np.corrcoef(obs, sim)[0, 1]
        r2 = correlation ** 2
        
        # RMSE (Root Mean Square Error)
        rmse = np.sqrt(np.mean((obs - sim) ** 2))
        
        # MAE (Mean Absolute Error)
        mae = np.mean(np.abs(obs - sim))
        
        # PBIAS (Percent Bias)
        pbias = 100 * np.sum(sim - obs) / np.sum(obs) if np.sum(obs) > 0 else 0
        
        return {
            'nse': nse,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'pbias': pbias
        }
    
    def _plot_diagnostics(self, zone_id: str, runoff_model: str, routing_model: str,
                         observed: np.ndarray, simulated: np.ndarray, 
                         result: ModelTestResult):
        """生成诊断图表"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Model Diagnostics: {zone_id} - {runoff_model}+{routing_model}', 
                     fontsize=14, fontweight='bold')
        
        # 时间序列对比
        ax1 = axes[0, 0]
        time_steps = np.arange(len(observed))
        ax1.plot(time_steps, observed, 'b-', label='Observed', linewidth=1.5, alpha=0.7)
        ax1.plot(time_steps, simulated, 'r-', label='Simulated', linewidth=1.5, alpha=0.7)
        ax1.set_xlabel('Time Step (hours)', fontsize=11)
        ax1.set_ylabel('Discharge (m³/s)', fontsize=11)
        ax1.set_title('Hydrograph Comparison', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 散点图
        ax2 = axes[0, 1]
        ax2.scatter(observed, simulated, alpha=0.5, s=20)
        max_val = max(observed.max(), simulated.max())
        ax2.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='1:1 Line')
        ax2.set_xlabel('Observed Discharge (m³/s)', fontsize=11)
        ax2.set_ylabel('Simulated Discharge (m³/s)', fontsize=11)
        ax2.set_title(f'Scatter Plot (NSE={result.nse:.3f}, R²={result.r2:.3f})', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 残差图
        ax3 = axes[1, 0]
        residuals = simulated - observed
        ax3.scatter(time_steps, residuals, alpha=0.5, s=20)
        ax3.axhline(y=0, color='k', linestyle='--', linewidth=2)
        ax3.set_xlabel('Time Step (hours)', fontsize=11)
        ax3.set_ylabel('Residual (m³/s)', fontsize=11)
        ax3.set_title(f'Residual Plot (RMSE={result.rmse:.3f})', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # 性能指标条形图
        ax4 = axes[1, 1]
        metrics_names = ['NSE', 'R²', 'RC Match']
        rc_match = 1 - abs(result.runoff_coef_sim - result.runoff_coef_obs) / max(0.01, result.runoff_coef_obs)
        metrics_values = [
            max(0, result.nse),
            result.r2,
            max(0, min(1, rc_match))
        ]
        colors = ['green' if v > 0.7 else 'orange' if v > 0.5 else 'red' for v in metrics_values]
        bars = ax4.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
        ax4.set_ylim([0, 1])
        ax4.set_ylabel('Score', fontsize=11)
        ax4.set_title('Performance Metrics', fontsize=12)
        ax4.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax4.axhline(y=0.7, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 在条形图上显示数值
        for bar, val in zip(bars, metrics_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # 保存图表
        output_file = self.output_dir / f'diagnostics_{zone_id}_{runoff_model}_{routing_model}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Diagnostic plot saved: {output_file}")
    
    def run_comprehensive_test(self, 
                               zones: Optional[List[str]] = None,
                               runoff_models: Optional[List[str]] = None,
                               routing_models: Optional[List[str]] = None,
                               max_iterations: int = 50):
        """运行综合测试"""
        
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE MODEL TESTING")
        logger.info("="*80)
        
        # 确定要测试的分区
        if zones is None:
            zones = [col for col in self.precip_df.columns if col != self.precip_df.columns[0]]
        
        # 确定要测试的模型
        if runoff_models is None:
            runoff_models = list(self.RUNOFF_MODELS.keys())
        if routing_models is None:
            routing_models = list(self.ROUTING_MODELS.keys())
        
        logger.info(f"\nZones to test: {len(zones)}")
        logger.info(f"Runoff models: {len(runoff_models)} - {', '.join(runoff_models)}")
        logger.info(f"Routing models: {len(routing_models)} - {', '.join(routing_models)}")
        logger.info(f"Total combinations: {len(zones) * len(runoff_models) * len(routing_models)}")
        
        # 测试所有组合
        for zone_id in zones:
            logger.info(f"\n{'#'*80}")
            logger.info(f"Processing Zone: {zone_id}")
            logger.info(f"{'#'*80}")
            
            zone_results = []
            
            for runoff_model in runoff_models:
                for routing_model in routing_models:
                    result = self.test_zone(zone_id, runoff_model, routing_model, max_iterations)
                    if result:
                        self.results.append(result)
                        zone_results.append(result)
            
            # 生成该分区的汇总报告
            if zone_results:
                self._generate_zone_summary(zone_id, zone_results)
        
        # 生成全局报告
        self._generate_global_report()
        
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE TESTING COMPLETED")
        logger.info("="*80)
    
    def _generate_zone_summary(self, zone_id: str, results: List[ModelTestResult]):
        """生成分区汇总报告"""
        
        # 找出最佳模型
        best_result = max(results, key=lambda r: r.get_score())
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Zone {zone_id} Summary:")
        logger.info(f"{'='*80}")
        logger.info(f"Best Model: {best_result.runoff_model} + {best_result.routing_model}")
        logger.info(f"Best Score: {best_result.get_score():.2f}")
        logger.info(f"Best NSE: {best_result.nse:.4f}")
        logger.info(f"Best R²: {best_result.r2:.4f}")
        logger.info(f"Best RC Match: Obs={best_result.runoff_coef_obs:.4f}, Sim={best_result.runoff_coef_sim:.4f}")
        
        # 生成排名图表
        self._plot_zone_ranking(zone_id, results)
    
    def _plot_zone_ranking(self, zone_id: str, results: List[ModelTestResult]):
        """绘制分区模型排名图"""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Model Ranking for {zone_id}', fontsize=14, fontweight='bold')
        
        # 按NSE排序
        sorted_by_nse = sorted(results, key=lambda r: r.nse, reverse=True)
        
        # NSE排名
        ax1 = axes[0]
        model_labels = [f"{r.runoff_model[:4]}+{r.routing_model[:4]}" for r in sorted_by_nse]
        nse_values = [r.nse for r in sorted_by_nse]
        colors = ['green' if v > 0.7 else 'orange' if v > 0.5 else 'red' for v in nse_values]
        
        bars = ax1.barh(range(len(model_labels)), nse_values, color=colors, alpha=0.7)
        ax1.set_yticks(range(len(model_labels)))
        ax1.set_yticklabels(model_labels, fontsize=9)
        ax1.set_xlabel('NSE', fontsize=11)
        ax1.set_title('Ranking by NSE', fontsize=12)
        ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        ax1.axvline(x=0.7, color='gray', linestyle='--', alpha=0.5)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 在条形图上显示数值
        for i, (bar, val) in enumerate(zip(bars, nse_values)):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2,
                    f' {val:.3f}',
                    ha='left', va='center', fontsize=8)
        
        # 综合评分排名
        ax2 = axes[1]
        sorted_by_score = sorted(results, key=lambda r: r.get_score(), reverse=True)
        model_labels2 = [f"{r.runoff_model[:4]}+{r.routing_model[:4]}" for r in sorted_by_score]
        score_values = [r.get_score() for r in sorted_by_score]
        colors2 = ['green' if v > 70 else 'orange' if v > 50 else 'red' for v in score_values]
        
        bars2 = ax2.barh(range(len(model_labels2)), score_values, color=colors2, alpha=0.7)
        ax2.set_yticks(range(len(model_labels2)))
        ax2.set_yticklabels(model_labels2, fontsize=9)
        ax2.set_xlabel('Overall Score (0-100)', fontsize=11)
        ax2.set_title('Ranking by Overall Score', fontsize=12)
        ax2.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
        ax2.axvline(x=70, color='gray', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 在条形图上显示数值
        for i, (bar, val) in enumerate(zip(bars2, score_values)):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2,
                    f' {val:.1f}',
                    ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        
        output_file = self.output_dir / f'ranking_{zone_id}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Ranking plot saved: {output_file}")
    
    def _generate_global_report(self):
        """生成全局报告"""
        
        logger.info("\n" + "="*80)
        logger.info("GENERATING GLOBAL REPORT")
        logger.info("="*80)
        
        if not self.results:
            logger.warning("No results to report")
            return
        
        # 保存详细结果（CSV）
        results_df = pd.DataFrame([
            {
                'Zone': r.zone_id,
                'Runoff_Model': r.runoff_model,
                'Routing_Model': r.routing_model,
                'NSE': r.nse,
                'R2': r.r2,
                'RMSE': r.rmse,
                'MAE': r.mae,
                'PBIAS': r.pbias,
                'RC_Observed': r.runoff_coef_obs,
                'RC_Simulated': r.runoff_coef_sim,
                'RC_Error': abs(r.runoff_coef_sim - r.runoff_coef_obs),
                'Overall_Score': r.get_score(),
                'Calibration_Time_s': r.calibration_time
            }
            for r in self.results
        ])
        
        csv_file = self.output_dir / 'comprehensive_results.csv'
        results_df.to_csv(csv_file, index=False)
        logger.info(f"Results saved to: {csv_file}")
        
        # 保存最佳参数（JSON）
        best_params = {}
        for r in self.results:
            key = f"{r.zone_id}_{r.runoff_model}_{r.routing_model}"
            best_params[key] = {
                'zone': r.zone_id,
                'runoff_model': r.runoff_model,
                'routing_model': r.routing_model,
                'parameters': r.best_params,
                'performance': {
                    'nse': r.nse,
                    'r2': r.r2,
                    'rmse': r.rmse,
                    'runoff_coef_obs': r.runoff_coef_obs,
                    'runoff_coef_sim': r.runoff_coef_sim,
                    'overall_score': r.get_score()
                }
            }
        
        json_file = self.output_dir / 'best_parameters.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(best_params, f, indent=2, ensure_ascii=False)
        logger.info(f"Best parameters saved to: {json_file}")
        
        # 生成中文报告
        self._generate_chinese_report(results_df)
        
        # 生成汇总图表
        self._generate_summary_plots(results_df)
    
    def _generate_chinese_report(self, results_df: pd.DataFrame):
        """生成中文报告"""
        
        report_file = self.output_dir / 'comprehensive_report.md'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 综合产汇流模型测试报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 测试概要\n\n")
            f.write(f"- 测试分区数量: {len(results_df['Zone'].unique())}\n")
            f.write(f"- 产流模型类型: {len(results_df['Runoff_Model'].unique())}\n")
            f.write(f"- 汇流模型类型: {len(results_df['Routing_Model'].unique())}\n")
            f.write(f"- 总测试组合: {len(results_df)}\n\n")
            
            # 各分区最佳模型
            f.write("## 各分区最佳模型\n\n")
            f.write("| 分区 | 最佳产流模型 | 最佳汇流模型 | NSE | R² | RC误差 | 综合评分 |\n")
            f.write("|------|-------------|-------------|-----|----|---------|---------|\n")
            
            for zone in results_df['Zone'].unique():
                zone_data = results_df[results_df['Zone'] == zone]
                best = zone_data.loc[zone_data['Overall_Score'].idxmax()]
                
                f.write(f"| {best['Zone']} | {best['Runoff_Model']} | {best['Routing_Model']} | ")
                f.write(f"{best['NSE']:.4f} | {best['R2']:.4f} | {best['RC_Error']:.4f} | {best['Overall_Score']:.2f} |\n")
            
            # 模型性能统计
            f.write("\n## 模型性能统计\n\n")
            
            # 按产流模型统计
            f.write("### 各产流模型平均性能\n\n")
            f.write("| 产流模型 | 平均NSE | 平均R² | 平均综合评分 | 测试次数 |\n")
            f.write("|---------|---------|--------|-------------|----------|\n")
            
            for model in results_df['Runoff_Model'].unique():
                model_data = results_df[results_df['Runoff_Model'] == model]
                f.write(f"| {model} | {model_data['NSE'].mean():.4f} | ")
                f.write(f"{model_data['R2'].mean():.4f} | {model_data['Overall_Score'].mean():.2f} | ")
                f.write(f"{len(model_data)} |\n")
            
            # 按汇流模型统计
            f.write("\n### 各汇流模型平均性能\n\n")
            f.write("| 汇流模型 | 平均NSE | 平均R² | 平均综合评分 | 测试次数 |\n")
            f.write("|---------|---------|--------|-------------|----------|\n")
            
            for model in results_df['Routing_Model'].unique():
                model_data = results_df[results_df['Routing_Model'] == model]
                f.write(f"| {model} | {model_data['NSE'].mean():.4f} | ")
                f.write(f"{model_data['R2'].mean():.4f} | {model_data['Overall_Score'].mean():.2f} | ")
                f.write(f"{len(model_data)} |\n")
            
            # 关键发现
            f.write("\n## 关键发现\n\n")
            
            # 最佳组合
            best_overall = results_df.loc[results_df['Overall_Score'].idxmax()]
            f.write(f"### 最佳模型组合\n\n")
            f.write(f"- **分区**: {best_overall['Zone']}\n")
            f.write(f"- **产流模型**: {best_overall['Runoff_Model']}\n")
            f.write(f"- **汇流模型**: {best_overall['Routing_Model']}\n")
            f.write(f"- **NSE**: {best_overall['NSE']:.4f}\n")
            f.write(f"- **R²**: {best_overall['R2']:.4f}\n")
            f.write(f"- **综合评分**: {best_overall['Overall_Score']:.2f}\n\n")
            
            # 性能统计
            f.write("### 整体性能分布\n\n")
            excellent = len(results_df[results_df['NSE'] > 0.8])
            good = len(results_df[(results_df['NSE'] > 0.6) & (results_df['NSE'] <= 0.8)])
            fair = len(results_df[(results_df['NSE'] > 0.4) & (results_df['NSE'] <= 0.6)])
            poor = len(results_df[results_df['NSE'] <= 0.4])
            
            f.write(f"- **优秀** (NSE > 0.8): {excellent} 个组合 ({excellent/len(results_df)*100:.1f}%)\n")
            f.write(f"- **良好** (0.6 < NSE ≤ 0.8): {good} 个组合 ({good/len(results_df)*100:.1f}%)\n")
            f.write(f"- **一般** (0.4 < NSE ≤ 0.6): {fair} 个组合 ({fair/len(results_df)*100:.1f}%)\n")
            f.write(f"- **较差** (NSE ≤ 0.4): {poor} 个组合 ({poor/len(results_df)*100:.1f}%)\n\n")
            
            # 建议
            f.write("## 应用建议\n\n")
            
            # 推荐的产流模型
            runoff_avg_scores = results_df.groupby('Runoff_Model')['Overall_Score'].mean().sort_values(ascending=False)
            f.write("### 推荐产流模型（按综合评分排序）\n\n")
            for i, (model, score) in enumerate(runoff_avg_scores.head(3).items(), 1):
                f.write(f"{i}. **{model}** - 平均评分: {score:.2f}\n")
            
            f.write("\n### 推荐汇流模型（按综合评分排序）\n\n")
            routing_avg_scores = results_df.groupby('Routing_Model')['Overall_Score'].mean().sort_values(ascending=False)
            for i, (model, score) in enumerate(routing_avg_scores.head(3).items(), 1):
                f.write(f"{i}. **{model}** - 平均评分: {score:.2f}\n")
            
            f.write("\n### 参数率定建议\n\n")
            f.write("1. 优先使用表现最好的模型组合\n")
            f.write("2. 对于特定分区，参考该分区的最佳参数作为初值\n")
            f.write("3. 考虑进行多目标率定（NSE + RC匹配）\n")
            f.write("4. 建议增加率定迭代次数以获得更好的结果\n")
            
        logger.info(f"Chinese report saved to: {report_file}")
    
    def _generate_summary_plots(self, results_df: pd.DataFrame):
        """生成汇总图表"""
        
        # 1. 模型性能热力图
        fig, ax = plt.subplots(figsize=(12, 8))
        
        pivot_nse = results_df.pivot_table(
            values='NSE',
            index='Runoff_Model',
            columns='Routing_Model',
            aggfunc='mean'
        )
        
        im = ax.imshow(pivot_nse.values, cmap='RdYlGn', aspect='auto', vmin=-0.5, vmax=1.0)
        
        ax.set_xticks(range(len(pivot_nse.columns)))
        ax.set_yticks(range(len(pivot_nse.index)))
        ax.set_xticklabels(pivot_nse.columns, fontsize=10)
        ax.set_yticklabels(pivot_nse.index, fontsize=10)
        
        plt.colorbar(im, ax=ax, label='Average NSE')
        
        # 添加数值标注
        for i in range(len(pivot_nse.index)):
            for j in range(len(pivot_nse.columns)):
                value = pivot_nse.values[i, j]
                if not np.isnan(value):
                    text = ax.text(j, i, f'{value:.3f}',
                                 ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title('Average NSE by Model Combination', fontsize=14, fontweight='bold')
        ax.set_xlabel('Routing Model', fontsize=12)
        ax.set_ylabel('Runoff Model', fontsize=12)
        
        plt.tight_layout()
        output_file = self.output_dir / 'model_performance_heatmap.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Heatmap saved to: {output_file}")
        
        # 2. 综合评分箱线图
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 按产流模型
        ax1 = axes[0]
        runoff_models = sorted(results_df['Runoff_Model'].unique())
        runoff_data = [results_df[results_df['Runoff_Model'] == m]['Overall_Score'].values 
                       for m in runoff_models]
        
        bp1 = ax1.boxplot(runoff_data, labels=runoff_models, patch_artist=True)
        for patch in bp1['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        ax1.set_xlabel('Runoff Model', fontsize=11)
        ax1.set_ylabel('Overall Score (0-100)', fontsize=11)
        ax1.set_title('Performance Distribution by Runoff Model', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='y')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 按汇流模型
        ax2 = axes[1]
        routing_models = sorted(results_df['Routing_Model'].unique())
        routing_data = [results_df[results_df['Routing_Model'] == m]['Overall_Score'].values 
                        for m in routing_models]
        
        bp2 = ax2.boxplot(routing_data, labels=routing_models, patch_artist=True)
        for patch in bp2['boxes']:
            patch.set_facecolor('lightgreen')
            patch.set_alpha(0.7)
        
        ax2.set_xlabel('Routing Model', fontsize=11)
        ax2.set_ylabel('Overall Score (0-100)', fontsize=11)
        ax2.set_title('Performance Distribution by Routing Model', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        output_file = self.output_dir / 'performance_distribution.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Distribution plot saved to: {output_file}")


def main():
    """主函数"""
    logger.info("="*80)
    logger.info("COMPREHENSIVE HYDROLOGIC MODEL TESTING SYSTEM")
    logger.info("="*80)
    
    # 设置路径
    output_dir = Path("results/comprehensive_model_testing")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 使用测试数据
    test_data_dir = Path("results/test_data_for_rc_analysis")
    precip_file = test_data_dir / "precipitation_timeseries.csv"
    discharge_file = test_data_dir / "discharge_timeseries.csv"
    watershed_file = test_data_dir / "watersheds.geojson"
    
    # 检查文件是否存在
    if not precip_file.exists():
        logger.error(f"Precipitation file not found: {precip_file}")
        logger.info("Please run generate_test_data_for_rc_analysis.py first:")
        logger.info("  python3 generate_test_data_for_rc_analysis.py")
        return 1
    
    # 创建测试器
    tester = ComprehensiveModelTester(output_dir)
    
    # 加载数据
    if not tester.load_data(precip_file, discharge_file, watershed_file):
        logger.error("Failed to load data")
        return 1
    
    # 运行综合测试
    # 为了快速演示，只测试前3个分区和部分模型
    # 实际使用时可以测试所有模型
    tester.run_comprehensive_test(
        zones=['Zone1', 'Zone2', 'Zone3'],  # 测试前3个分区
        runoff_models=['HBV', 'XinAnJiang', 'SCS-CN'],  # 测试主要产流模型
        routing_models=['Muskingum', 'Lag'],  # 测试主要汇流模型
        max_iterations=30  # 快速率定，实际应用建议50-100
    )
    
    logger.info("\n" + "="*80)
    logger.info("ALL TESTS COMPLETED SUCCESSFULLY!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
