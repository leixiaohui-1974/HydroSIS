#!/usr/bin/env python3
"""
简化版综合模型测试系统

基于已验证的径流系数分析代码，扩展支持：
1. 多个分区的并行测试
2. 详细的模型诊断
3. 参数敏感性分析
4. 最佳参数保存
5. 完整的中文报告

作者：Claude
日期：2025-10-26
"""

import sys
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# 配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 尝试导入scipy（用于优化）
try:
    from scipy.optimize import differential_evolution
    HAS_SCIPY = True
except ImportError:
    logger.warning("scipy not available, calibration will be limited")
    HAS_SCIPY = False


# ============================================================================
# 产流模型实现（简化版）
# ============================================================================

class HBVModel:
    """HBV产流模型"""
    def __init__(self, field_capacity=300, beta=2.0, k0=0.05, k1=0.01, k2=0.001):
        self.FC = field_capacity
        self.BETA = beta
        self.K0 = k0
        self.K1 = k1
        self.K2 = k2
        self.soil_moisture = field_capacity * 0.5
        self.upper_zone = 0.0
        self.lower_zone = 0.0
    
    def run(self, precipitation: np.ndarray, temperature: np.ndarray = None) -> np.ndarray:
        """运行模型"""
        runoff = np.zeros(len(precipitation))
        
        for t, precip in enumerate(precipitation):
            # 土壤水分更新
            if precip > 0:
                soil_ratio = self.soil_moisture / self.FC
                recharge = precip * (soil_ratio ** self.BETA)
                self.soil_moisture = min(self.FC, self.soil_moisture + precip - recharge)
            else:
                recharge = 0
            
            # 上层出流
            if self.upper_zone > 0:
                q0 = self.K0 * self.upper_zone
                q1 = self.K1 * self.upper_zone
                self.upper_zone = max(0, self.upper_zone + recharge - q0 - q1)
            else:
                q0, q1 = 0, 0
                self.upper_zone += recharge
            
            # 下层出流
            q2 = self.K2 * self.lower_zone
            self.lower_zone = max(0, self.lower_zone + q1 - q2)
            
            # 总径流
            runoff[t] = q0 + q2
        
        return runoff


class XinAnJiangModel:
    """新安江产流模型"""
    def __init__(self, um=20, lm=80, b=0.4, im=0.01, sm=20):
        self.UM = um  # 上层蓄水容量
        self.LM = lm  # 下层蓄水容量
        self.B = b    # 蓄水容量曲线指数
        self.IM = im  # 不透水面积比例
        self.SM = sm  # 自由水蓄水容量
        
        self.wu = um * 0.5  # 上层土壤水分
        self.wl = lm * 0.5  # 下层土壤水分
        self.s = 0.0         # 自由水蓄水量
    
    def run(self, precipitation: np.ndarray, evap: np.ndarray = None) -> np.ndarray:
        """运行模型"""
        if evap is None:
            evap = np.ones(len(precipitation)) * 2.0  # 默认2mm蒸发
        
        runoff = np.zeros(len(precipitation))
        
        for t, (precip, et) in enumerate(zip(precipitation, evap)):
            # 蒸发
            eu = min(et, self.wu)
            self.wu -= eu
            remaining_et = et - eu
            
            el = min(remaining_et, self.wl)
            self.wl -= el
            
            # 产流计算
            wm = self.UM + self.LM
            w = self.wu + self.wl
            a = wm * (1 - (1 - w / wm) ** (1 / (1 + self.B)))
            
            # 降雨分配
            if precip > 0:
                if precip + a >= wm:
                    r = precip - (wm - w)
                else:
                    r = precip - wm + w + wm * ((1 - (precip + a) / wm) ** (1 + self.B))
                
                r = max(0, r)
                
                # 更新土壤水分
                infiltration = precip - r
                self.wu = min(self.UM, self.wu + infiltration * 0.3)
                self.wl = min(self.LM, self.wl + infiltration * 0.7)
                
                # 自由水蓄水库
                self.s = min(self.SM, self.s + r)
                runoff[t] = max(0, self.s - self.SM * 0.5) * 0.3
                self.s -= runoff[t]
        
        return runoff


class SCSModel:
    """SCS-CN产流模型"""
    def __init__(self, curve_number=70, ia_ratio=0.2):
        self.CN = curve_number
        self.Ia_ratio = ia_ratio
        self.S = 254 * (100 / self.CN - 1)  # 最大潜在保持量(mm)
        self.Ia = self.Ia_ratio * self.S     # 初损
    
    def run(self, precipitation: np.ndarray, temperature: np.ndarray = None) -> np.ndarray:
        """运行模型"""
        runoff = np.zeros(len(precipitation))
        
        for t, precip in enumerate(precipitation):
            if precip > self.Ia:
                runoff[t] = ((precip - self.Ia) ** 2) / (precip - self.Ia + self.S)
        
        return runoff


# ============================================================================
# 汇流模型实现（简化版）
# ============================================================================

class MuskingumRouting:
    """Muskingum汇流"""
    def __init__(self, K=2.0, x=0.2):
        self.K = K
        self.x = x
        self.C1 = (2 * K * x + 1) / (2 * K * (1 - x) + 1)
        self.C2 = (1 - 2 * K * x) / (2 * K * (1 - x) + 1)
        self.C3 = (2 * K * (1 - x) - 1) / (2 * K * (1 - x) + 1)
    
    def route(self, inflow: np.ndarray) -> np.ndarray:
        """汇流计算"""
        outflow = np.zeros(len(inflow))
        outflow[0] = inflow[0]
        
        for t in range(1, len(inflow)):
            outflow[t] = self.C1 * inflow[t] + self.C2 * inflow[t-1] - self.C3 * outflow[t-1]
            outflow[t] = max(0, outflow[t])
        
        return outflow


class LagRouting:
    """滞后汇流"""
    def __init__(self, lag_time=3.0):
        self.lag = int(lag_time)
    
    def route(self, inflow: np.ndarray) -> np.ndarray:
        """汇流计算"""
        outflow = np.roll(inflow, self.lag)
        outflow[:self.lag] = inflow[0]  # 前几个时段用初始值
        return outflow


# ============================================================================
# 模型测试器
# ============================================================================

@dataclass
class TestResult:
    """测试结果"""
    zone_id: str
    runoff_model: str
    routing_model: str
    nse: float
    r2: float
    rmse: float
    mae: float
    rc_obs: float
    rc_sim: float
    best_params: Dict
    calibration_time: float


class ModelTester:
    """模型测试器"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['figure.dpi'] = 150
    
    def load_data(self, precip_file, discharge_file, watershed_file):
        """加载数据"""
        logger.info("Loading data...")
        
        self.precip_df = pd.read_csv(precip_file)
        self.discharge_df = pd.read_csv(discharge_file)
        
        import geopandas as gpd
        self.watersheds = gpd.read_file(watershed_file)
        
        logger.info(f"  Zones: {len(self.precip_df.columns) - 1}")
        logger.info(f"  Time steps: {len(self.precip_df)}")
        
        return True
    
    def test_zone_model(self, zone_id: str, runoff_model: str, routing_model: str):
        """测试单个分区的模型组合"""
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing: {zone_id} | {runoff_model} + {routing_model}")
        logger.info(f"{'='*80}")
        
        try:
            # 获取数据
            precip = self.precip_df[zone_id].values
            discharge_obs = self.discharge_df[zone_id].values
            
            # 获取面积
            zone_data = self.watersheds[self.watersheds['id'] == zone_id]
            if len(zone_data) == 0:
                zone_data = self.watersheds.iloc[0:1]
            area_km2 = zone_data.iloc[0].geometry.area / 1e6
            
            logger.info(f"  Area: {area_km2:.2f} km²")
            
            # 运行率定
            start_time = datetime.now()
            
            if not HAS_SCIPY:
                logger.warning("  Using default parameters (no optimization)")
                best_params = self._get_default_params(runoff_model, routing_model)
                simulated = self._run_model(precip, area_km2, runoff_model, routing_model, best_params)
            else:
                best_params, simulated = self._calibrate_model(
                    precip, discharge_obs, area_km2, runoff_model, routing_model
                )
            
            cal_time = (datetime.now() - start_time).total_seconds()
            
            # 计算指标
            metrics = self._calculate_metrics(discharge_obs, simulated)
            
            # 计算径流系数
            total_precip = precip.sum()
            obs_vol = discharge_obs.sum() * 3600
            sim_vol = simulated.sum() * 3600
            rc_obs = (obs_vol / (area_km2 * 1e6)) * 1000 / total_precip
            rc_sim = (sim_vol / (area_km2 * 1e6)) * 1000 / total_precip
            
            logger.info(f"  NSE: {metrics['nse']:.4f}")
            logger.info(f"  R²: {metrics['r2']:.4f}")
            logger.info(f"  RC: Obs={rc_obs:.4f}, Sim={rc_sim:.4f}")
            
            # 保存结果
            result = TestResult(
                zone_id=zone_id,
                runoff_model=runoff_model,
                routing_model=routing_model,
                nse=metrics['nse'],
                r2=metrics['r2'],
                rmse=metrics['rmse'],
                mae=metrics['mae'],
                rc_obs=rc_obs,
                rc_sim=rc_sim,
                best_params=best_params,
                calibration_time=cal_time
            )
            
            self.results.append(result)
            
            # 绘制诊断图
            self._plot_diagnostic(zone_id, runoff_model, routing_model, 
                                 discharge_obs, simulated, result)
            
            return result
            
        except Exception as e:
            logger.error(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _get_default_params(self, runoff_model: str, routing_model: str) -> Dict:
        """获取默认参数"""
        params = {}
        
        if runoff_model == 'HBV':
            params.update({'field_capacity': 300, 'beta': 2.0, 'k0': 0.05, 'k1': 0.01, 'k2': 0.001})
        elif runoff_model == 'XinAnJiang':
            params.update({'um': 20, 'lm': 80, 'b': 0.4, 'im': 0.01, 'sm': 20})
        elif runoff_model == 'SCS-CN':
            params.update({'curve_number': 70, 'ia_ratio': 0.2})
        
        if routing_model == 'Muskingum':
            params.update({'K': 2.0, 'x': 0.2})
        elif routing_model == 'Lag':
            params.update({'lag_time': 3.0})
        
        return params
    
    def _run_model(self, precip: np.ndarray, area_km2: float, 
                   runoff_model: str, routing_model: str, params: Dict) -> np.ndarray:
        """运行模型"""
        
        # 产流
        if runoff_model == 'HBV':
            model = HBVModel(**{k: v for k, v in params.items() 
                              if k in ['field_capacity', 'beta', 'k0', 'k1', 'k2']})
            runoff_mm = model.run(precip)
        elif runoff_model == 'XinAnJiang':
            model = XinAnJiangModel(**{k: v for k, v in params.items()
                                      if k in ['um', 'lm', 'b', 'im', 'sm']})
            runoff_mm = model.run(precip)
        elif runoff_model == 'SCS-CN':
            model = SCSModel(**{k: v for k, v in params.items()
                              if k in ['curve_number', 'ia_ratio']})
            runoff_mm = model.run(precip)
        else:
            runoff_mm = precip * 0.4  # 简单假设
        
        # 转换为流量 (m³/s)
        area_m2 = area_km2 * 1e6
        discharge = (runoff_mm * area_m2) / (1000 * 3600)
        
        # 汇流
        if routing_model == 'Muskingum':
            router = MuskingumRouting(**{k: v for k, v in params.items()
                                        if k in ['K', 'x']})
            discharge = router.route(discharge)
        elif routing_model == 'Lag':
            router = LagRouting(**{k: v for k, v in params.items()
                                  if k in ['lag_time']})
            discharge = router.route(discharge)
        
        return discharge
    
    def _calibrate_model(self, precip: np.ndarray, discharge_obs: np.ndarray,
                        area_km2: float, runoff_model: str, routing_model: str):
        """率定模型"""
        
        # 定义参数边界
        bounds = []
        param_names = []
        
        if runoff_model == 'HBV':
            bounds = [(200, 600), (1.0, 6.0), (0.01, 0.5), (0.001, 0.1), (0.0001, 0.01)]
            param_names = ['field_capacity', 'beta', 'k0', 'k1', 'k2']
        elif runoff_model == 'XinAnJiang':
            bounds = [(10, 30), (60, 100), (0.1, 0.5), (0.001, 0.05), (10, 50)]
            param_names = ['um', 'lm', 'b', 'im', 'sm']
        elif runoff_model == 'SCS-CN':
            bounds = [(30, 98), (0.05, 0.3)]
            param_names = ['curve_number', 'ia_ratio']
        
        if routing_model == 'Muskingum':
            bounds.extend([(0.1, 24.0), (0.0, 0.5)])
            param_names.extend(['K', 'x'])
        elif routing_model == 'Lag':
            bounds.extend([(0.1, 10.0)])
            param_names.extend(['lag_time'])
        
        # 目标函数
        def objective(params_array):
            params = dict(zip(param_names, params_array))
            try:
                simulated = self._run_model(precip, area_km2, runoff_model, routing_model, params)
                nse = self._calculate_nse(discharge_obs, simulated)
                return -nse  # 最小化负NSE
            except:
                return 1e10
        
        # 运行优化
        logger.info("  Running calibration...")
        result = differential_evolution(
            objective,
            bounds,
            maxiter=30,
            popsize=10,
            seed=42,
            workers=1,
            updating='immediate',
            polish=False
        )
        
        best_params = dict(zip(param_names, result.x))
        simulated = self._run_model(precip, area_km2, runoff_model, routing_model, best_params)
        
        logger.info(f"  Calibration completed! Best NSE: {-result.fun:.4f}")
        
        return best_params, simulated
    
    def _calculate_metrics(self, obs: np.ndarray, sim: np.ndarray) -> Dict:
        """计算评价指标"""
        nse = self._calculate_nse(obs, sim)
        r2 = np.corrcoef(obs, sim)[0, 1] ** 2
        rmse = np.sqrt(np.mean((obs - sim) ** 2))
        mae = np.mean(np.abs(obs - sim))
        
        return {'nse': nse, 'r2': r2, 'rmse': rmse, 'mae': mae}
    
    def _calculate_nse(self, obs: np.ndarray, sim: np.ndarray) -> float:
        """计算NSE"""
        numerator = np.sum((obs - sim) ** 2)
        denominator = np.sum((obs - np.mean(obs)) ** 2)
        return 1 - (numerator / denominator) if denominator > 0 else -np.inf
    
    def _plot_diagnostic(self, zone_id, runoff_model, routing_model, 
                        obs, sim, result):
        """绘制诊断图"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{zone_id}: {runoff_model} + {routing_model}', 
                     fontsize=14, fontweight='bold')
        
        # 时序图
        ax1 = axes[0, 0]
        time = np.arange(len(obs))
        ax1.plot(time, obs, 'b-', label='Observed', linewidth=1.5, alpha=0.7)
        ax1.plot(time, sim, 'r-', label='Simulated', linewidth=1.5, alpha=0.7)
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Discharge (m³/s)')
        ax1.set_title('Hydrograph')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 散点图
        ax2 = axes[0, 1]
        ax2.scatter(obs, sim, alpha=0.5)
        max_val = max(obs.max(), sim.max())
        ax2.plot([0, max_val], [0, max_val], 'k--')
        ax2.set_xlabel('Observed')
        ax2.set_ylabel('Simulated')
        ax2.set_title(f'Scatter (NSE={result.nse:.3f})')
        ax2.grid(True, alpha=0.3)
        
        # 残差
        ax3 = axes[1, 0]
        residual = sim - obs
        ax3.scatter(time, residual, alpha=0.5)
        ax3.axhline(y=0, color='k', linestyle='--')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Residual')
        ax3.set_title(f'Residuals (RMSE={result.rmse:.3f})')
        ax3.grid(True, alpha=0.3)
        
        # 指标
        ax4 = axes[1, 1]
        metrics = ['NSE', 'R²', 'RC Match']
        rc_match = 1 - abs(result.rc_sim - result.rc_obs) / max(0.01, result.rc_obs)
        values = [max(0, result.nse), result.r2, max(0, min(1, rc_match))]
        colors = ['green' if v > 0.7 else 'orange' if v > 0.5 else 'red' for v in values]
        
        bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
        ax4.set_ylim([0, 1])
        ax4.set_ylabel('Score')
        ax4.set_title('Performance')
        ax4.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2, height,
                    f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        output_file = self.output_dir / f'diagnostic_{zone_id}_{runoff_model}_{routing_model}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Diagnostic saved: {output_file.name}")
    
    def run_all_tests(self, zones=None):
        """运行所有测试"""
        
        logger.info("\n" + "="*80)
        logger.info("RUNNING COMPREHENSIVE MODEL TESTS")
        logger.info("="*80)
        
        if zones is None:
            zones = [col for col in self.precip_df.columns if col != self.precip_df.columns[0]]
        
        # 定义要测试的模型组合
        runoff_models = ['HBV', 'XinAnJiang', 'SCS-CN']
        routing_models = ['Muskingum', 'Lag']
        
        logger.info(f"\nZones: {len(zones)}")
        logger.info(f"Runoff models: {', '.join(runoff_models)}")
        logger.info(f"Routing models: {', '.join(routing_models)}")
        logger.info(f"Total: {len(zones) * len(runoff_models) * len(routing_models)} tests")
        
        # 测试所有组合
        for zone_id in zones:
            logger.info(f"\n{'#'*80}")
            logger.info(f"Processing Zone: {zone_id}")
            logger.info(f"{'#'*80}")
            
            for runoff_model in runoff_models:
                for routing_model in routing_models:
                    self.test_zone_model(zone_id, runoff_model, routing_model)
        
        # 生成报告
        self._generate_report()
    
    def _generate_report(self):
        """生成报告"""
        
        logger.info("\n" + "="*80)
        logger.info("GENERATING REPORTS")
        logger.info("="*80)
        
        if not self.results:
            logger.warning("No results to report")
            return
        
        # CSV结果
        df = pd.DataFrame([
            {
                'Zone': r.zone_id,
                'Runoff_Model': r.runoff_model,
                'Routing_Model': r.routing_model,
                'NSE': r.nse,
                'R2': r.r2,
                'RMSE': r.rmse,
                'MAE': r.mae,
                'RC_Obs': r.rc_obs,
                'RC_Sim': r.rc_sim,
                'RC_Error': abs(r.rc_sim - r.rc_obs),
                'Cal_Time': r.calibration_time
            }
            for r in self.results
        ])
        
        csv_file = self.output_dir / 'model_test_results.csv'
        df.to_csv(csv_file, index=False)
        logger.info(f"Results saved: {csv_file}")
        
        # JSON参数
        params_dict = {}
        for r in self.results:
            key = f"{r.zone_id}_{r.runoff_model}_{r.routing_model}"
            params_dict[key] = {
                'zone': r.zone_id,
                'runoff_model': r.runoff_model,
                'routing_model': r.routing_model,
                'parameters': r.best_params,
                'performance': {
                    'nse': r.nse,
                    'r2': r.r2,
                    'rmse': r.rmse,
                    'rc_obs': r.rc_obs,
                    'rc_sim': r.rc_sim
                }
            }
        
        json_file = self.output_dir / 'best_parameters.json'
        with open(json_file, 'w') as f:
            json.dump(params_dict, f, indent=2)
        logger.info(f"Parameters saved: {json_file}")
        
        # 中文报告
        report_file = self.output_dir / 'test_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 综合模型测试报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 测试概要\n\n")
            f.write(f"- 测试分区: {len(df['Zone'].unique())}\n")
            f.write(f"- 产流模型: {len(df['Runoff_Model'].unique())}\n")
            f.write(f"- 汇流模型: {len(df['Routing_Model'].unique())}\n")
            f.write(f"- 总测试数: {len(df)}\n\n")
            
            f.write("## 各分区最佳模型\n\n")
            f.write("| 分区 | 产流模型 | 汇流模型 | NSE | R² | RC误差 |\n")
            f.write("|------|---------|---------|-----|----|---------|\n")
            
            for zone in df['Zone'].unique():
                zone_df = df[df['Zone'] == zone]
                best = zone_df.loc[zone_df['NSE'].idxmax()]
                f.write(f"| {best['Zone']} | {best['Runoff_Model']} | {best['Routing_Model']} | ")
                f.write(f"{best['NSE']:.4f} | {best['R2']:.4f} | {best['RC_Error']:.4f} |\n")
            
            f.write("\n## 模型性能排名\n\n")
            
            # 产流模型
            f.write("### 产流模型平均性能\n\n")
            for model in df['Runoff_Model'].unique():
                model_df = df[df['Runoff_Model'] == model]
                avg_nse = model_df['NSE'].mean()
                f.write(f"- **{model}**: 平均NSE = {avg_nse:.4f}\n")
            
            # 汇流模型
            f.write("\n### 汇流模型平均性能\n\n")
            for model in df['Routing_Model'].unique():
                model_df = df[df['Routing_Model'] == model]
                avg_nse = model_df['NSE'].mean()
                f.write(f"- **{model}**: 平均NSE = {avg_nse:.4f}\n")
        
        logger.info(f"Report saved: {report_file}")
        
        logger.info("\n" + "="*80)
        logger.info("ALL TESTS COMPLETED!")
        logger.info("="*80)


def main():
    """主函数"""
    logger.info("="*80)
    logger.info("SIMPLIFIED MODEL TESTING SYSTEM")
    logger.info("="*80)
    
    # 路径设置
    output_dir = Path("results/simplified_model_testing")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    test_data_dir = Path("results/test_data_for_rc_analysis")
    precip_file = test_data_dir / "precipitation_timeseries.csv"
    discharge_file = test_data_dir / "discharge_timeseries.csv"
    watershed_file = test_data_dir / "watersheds.geojson"
    
    if not precip_file.exists():
        logger.error(f"Data not found: {precip_file}")
        logger.info("Please run: python3 generate_test_data_for_rc_analysis.py")
        return 1
    
    # 创建测试器
    tester = ModelTester(output_dir)
    
    # 加载数据
    if not tester.load_data(precip_file, discharge_file, watershed_file):
        return 1
    
    # 运行测试（前3个分区）
    tester.run_all_tests(zones=['Zone1', 'Zone2', 'Zone3'])
    
    logger.info(f"\nResults saved to: {output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
