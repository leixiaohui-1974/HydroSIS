#!/usr/bin/env python3
"""Comprehensive Test Scenario Runner - With All Visualization and Validation

This script runs all test scenarios and generates:
1. Flow direction maps (correctly differentiated from accumulation)
2. Pour points distribution map (3 mainstream + 3 tributary = 6 points)
3. Rain gauge distribution map (50 gauges)
4. Subbasin areal precipitation animated GIF
5. Time series plots for each rain gauge
6. Time series plots for each pour point discharge
7. Automatic runoff coefficient calculation
8. Precipitation-runoff comparison plots
"""
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensiveVisualizer:
    """Comprehensive Visualizer - Generate all required charts"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Import necessary libraries
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap, BoundaryNorm
            self.plt = plt
            self.ListedColormap = ListedColormap
            self.BoundaryNorm = BoundaryNorm
        except ImportError:
            logger.warning("matplotlib not installed, some visualization features will be unavailable")
            self.plt = None
    
    def plot_flow_direction_correct(self, flow_dir_path: Path, output_path: Path):
        """Plot flow direction correctly (different from accumulation)
        
        Uses D8 encoding with 8 directions, each with different color
        """
        if not self.plt:
            return None
        
        try:
            import rasterio
            
            with rasterio.open(flow_dir_path) as src:
                flow_dir = src.read(1)
                
                # D8 direction encoding: 1=E, 2=NE, 3=N, 4=NW, 5=W, 6=SW, 7=S, 8=SE
                # Use different colors for different directions
                colors = ['#808080', '#ff0000', '#ff7f00', '#ffff00', '#7fff00', 
                         '#00ff00', '#00ff7f', '#00ffff', '#007fff']
                cmap = self.ListedColormap(colors)
                bounds = np.arange(-0.5, 9.5, 1.0)
                norm = self.BoundaryNorm(bounds, cmap.N)
                
                fig, ax = self.plt.subplots(figsize=(12, 10))
                im = ax.imshow(flow_dir, cmap=cmap, norm=norm, origin='upper')
                ax.set_title('Flow Direction (D8 Encoding)', fontsize=14, fontweight='bold')
                ax.set_xlabel('Column Index', fontsize=12)
                ax.set_ylabel('Row Index', fontsize=12)
                
                # Add colorbar with labels
                cbar = self.plt.colorbar(im, ax=ax, ticks=range(9))
                cbar.ax.set_yticklabels(['NoData', 'E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'])
                cbar.set_label('Flow Direction', fontsize=12)
                
                # Add statistics
                valid_data = flow_dir[(flow_dir >= 1) & (flow_dir <= 8)]
                stats_text = (
                    f"Valid cells: {len(valid_data)}\n"
                    f"Directions: {len(np.unique(valid_data))}\n"
                    f"Mode: {int(np.median(valid_data))}"
                )
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor='white', alpha=0.8), fontsize=10)
                
                self.plt.tight_layout()
                self.plt.savefig(output_path, dpi=200, bbox_inches='tight')
                self.plt.close(fig)
                
                logger.info(f"✓ 流向图已保存: {output_path}")
                return output_path
        
        except Exception as e:
            logger.error(f"Failed to plot flow direction: {e}")
            return None
    
    def plot_pour_points_distribution(
        self, 
        pour_points_path: Path, 
        dem_path: Path,
        output_path: Path,
        mainstream_count: int = 3,
        tributary_count: int = 3
    ):
        """Plot pour points distribution (distinguish mainstream and tributary)"""
        if not self.plt:
            return None
        
        try:
            import geopandas as gpd
            import rasterio
            from rasterio.plot import show
            
            # Read pour points
            gdf = gpd.read_file(pour_points_path)
            
            # Sort by accumulation, first N are mainstream
            if 'accumulation' in gdf.columns:
                gdf = gdf.sort_values('accumulation', ascending=False)
            
            total_points = len(gdf)
            mainstream_points = gdf.head(mainstream_count)
            tributary_points = gdf.iloc[mainstream_count:mainstream_count+tributary_count]
            
            fig, ax = self.plt.subplots(figsize=(14, 12))
            
            # Draw DEM basemap
            with rasterio.open(dem_path) as src:
                show(src, ax=ax, cmap='terrain', alpha=0.6)
            
            # Plot mainstream pour points (large red dots)
            if len(mainstream_points) > 0:
                mainstream_points.plot(
                    ax=ax, color='red', markersize=200, 
                    marker='o', edgecolor='darkred', linewidth=2,
                    label=f'Mainstream Pour Points ({len(mainstream_points)})',
                    zorder=10
                )
                
                # 添加标签
                for idx, row in mainstream_points.iterrows():
                    geom = row.geometry
                    label = row.get('id', f'M{idx}')
                    ax.annotate(label, xy=(geom.x, geom.y), xytext=(5, 5),
                               textcoords='offset points', fontsize=11,
                               bbox=dict(boxstyle='round', facecolor='red', alpha=0.7),
                               color='white', fontweight='bold')
            
            # Plot tributary pour points (blue triangles)
            if len(tributary_points) > 0:
                tributary_points.plot(
                    ax=ax, color='blue', markersize=150,
                    marker='^', edgecolor='darkblue', linewidth=2,
                    label=f'Tributary Pour Points ({len(tributary_points)})',
                    zorder=9
                )
                
                # 添加标签
                for idx, row in tributary_points.iterrows():
                    geom = row.geometry
                    label = row.get('id', f'T{idx}')
                    ax.annotate(label, xy=(geom.x, geom.y), xytext=(5, -15),
                               textcoords='offset points', fontsize=10,
                               bbox=dict(boxstyle='round', facecolor='blue', alpha=0.7),
                               color='white', fontweight='bold')
            
            ax.set_title(f'Pour Points Distribution\n{mainstream_count} Mainstream + {tributary_count} Tributary',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('X Coordinate', fontsize=12)
            ax.set_ylabel('Y Coordinate', fontsize=12)
            ax.legend(loc='upper right', fontsize=11)
            ax.grid(True, alpha=0.3)
            
            self.plt.tight_layout()
            self.plt.savefig(output_path, dpi=200, bbox_inches='tight')
            self.plt.close(fig)
            
            logger.info(f"Pour points distribution map saved: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Failed to plot pour points distribution: {e}")
            return None
    
    def plot_rain_gauges_distribution(
        self,
        gauges_path: Path,
        watershed_path: Path,
        output_path: Path,
        expected_count: int = 50
    ):
        """Plot rain gauge distribution"""
        if not self.plt:
            return None
        
        try:
            import geopandas as gpd
            
            # Read rain gauges and watersheds
            gauges_gdf = gpd.read_file(gauges_path)
            watershed_gdf = gpd.read_file(watershed_path)
            
            fig, ax = self.plt.subplots(figsize=(14, 12))
            
            # 绘制流域底图
            watershed_gdf.boundary.plot(ax=ax, edgecolor='gray', linewidth=1.5)
            watershed_gdf.plot(ax=ax, alpha=0.3, cmap='Set3')
            
            # 绘制雨量站
            gauges_gdf.plot(ax=ax, color='darkgreen', markersize=80,
                           marker='o', edgecolor='yellow', linewidth=1.5,
                           label=f'Rain Gauges ({len(gauges_gdf)})', zorder=5)
            
            # 标注部分雨量站
            sample_size = min(10, len(gauges_gdf))
            for idx, row in gauges_gdf.head(sample_size).iterrows():
                geom = row.geometry
                label = row.get('id', f'G{idx}')
                ax.annotate(label, xy=(geom.x, geom.y), xytext=(3, 3),
                           textcoords='offset points', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            ax.set_title(f'Rain Gauge Distribution\nTotal: {len(gauges_gdf)} gauges (Expected: {expected_count})',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('X Coordinate', fontsize=12)
            ax.set_ylabel('Y Coordinate', fontsize=12)
            ax.legend(loc='upper right', fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # 添加密度统计
            if len(watershed_gdf) > 0:
                total_area = watershed_gdf.geometry.area.sum() / 1e6  # km²
                density = len(gauges_gdf) / total_area if total_area > 0 else 0
                stats_text = f"Density: {density:.2f} gauges/km²\nArea: {total_area:.2f} km²"
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round',
                       facecolor='white', alpha=0.8), fontsize=10)
            
            self.plt.tight_layout()
            self.plt.savefig(output_path, dpi=200, bbox_inches='tight')
            self.plt.close(fig)
            
            logger.info(f"Rain gauge distribution map saved: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Failed to plot rain gauge distribution: {e}")
            return None
    
    def plot_timeseries_all_gauges(
        self,
        precipitation_csv: Path,
        output_dir: Path
    ):
        """绘制所有雨量站的时间序列图"""
        if not self.plt:
            return []
        
        try:
            import pandas as pd
            
            df = pd.read_csv(precipitation_csv)
            
            # 假设第一列是时间，其他列是各雨量站
            time_col = df.columns[0]
            gauge_cols = [col for col in df.columns if col != time_col]
            
            output_paths = []
            
            # 生成总览图（所有雨量站）
            fig, ax = self.plt.subplots(figsize=(14, 8))
            for col in gauge_cols[:10]:  # 只显示前10个，避免太乱
                ax.plot(df.index, df[col], label=col, linewidth=1.5, alpha=0.7)
            
            ax.set_title('Precipitation Time Series (First 10 Gauges)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time Step', fontsize=12)
            ax.set_ylabel('Precipitation (mm)', fontsize=12)
            ax.legend(loc='upper right', ncol=2, fontsize=9)
            ax.grid(True, alpha=0.3)
            
            overview_path = output_dir / 'precipitation_timeseries_overview.png'
            self.plt.tight_layout()
            self.plt.savefig(overview_path, dpi=150, bbox_inches='tight')
            self.plt.close(fig)
            output_paths.append(overview_path)
            
            # 生成单个雨量站图（前20个）
            for col in gauge_cols[:20]:
                fig, ax = self.plt.subplots(figsize=(12, 6))
                ax.plot(df.index, df[col], linewidth=2, color='blue')
                ax.fill_between(df.index, 0, df[col], alpha=0.3)
                
                ax.set_title(f'Precipitation at {col}', fontsize=13, fontweight='bold')
                ax.set_xlabel('Time Step', fontsize=11)
                ax.set_ylabel('Precipitation (mm)', fontsize=11)
                ax.grid(True, alpha=0.3)
                
                # 添加统计
                stats_text = (
                    f"Total: {df[col].sum():.2f} mm\n"
                    f"Mean: {df[col].mean():.2f} mm\n"
                    f"Max: {df[col].max():.2f} mm"
                )
                ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                       fontsize=9)
                
                gauge_path = output_dir / f'precipitation_{col}.png'
                self.plt.tight_layout()
                self.plt.savefig(gauge_path, dpi=150, bbox_inches='tight')
                self.plt.close(fig)
                output_paths.append(gauge_path)
            
            logger.info(f"Generated {len(output_paths)} rain gauge time series plots")
            return output_paths
        
        except Exception as e:
            logger.error(f"Failed to plot rain gauge time series: {e}")
            return []
    
    def plot_discharge_timeseries(
        self,
        discharge_csv: Path,
        output_dir: Path,
        pour_points: Optional[List[str]] = None
    ):
        """绘制各汇水点的径流时间序列"""
        if not self.plt:
            return []
        
        try:
            import pandas as pd
            
            df = pd.read_csv(discharge_csv)
            
            # 假设第一列是时间，其他列是各汇水点
            time_col = df.columns[0]
            discharge_cols = [col for col in df.columns if col != time_col]
            
            output_paths = []
            
            # 生成总览图
            fig, ax = self.plt.subplots(figsize=(14, 8))
            for col in discharge_cols:
                ax.plot(df.index, df[col], label=col, linewidth=2)
            
            ax.set_title('Discharge Time Series at All Pour Points', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time Step', fontsize=12)
            ax.set_ylabel('Discharge (m³/s)', fontsize=12)
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            overview_path = output_dir / 'discharge_timeseries_overview.png'
            self.plt.tight_layout()
            self.plt.savefig(overview_path, dpi=150, bbox_inches='tight')
            self.plt.close(fig)
            output_paths.append(overview_path)
            
            # 生成单个汇水点图
            for col in discharge_cols:
                fig, ax = self.plt.subplots(figsize=(12, 6))
                ax.plot(df.index, df[col], linewidth=2, color='navy')
                ax.fill_between(df.index, 0, df[col], alpha=0.3, color='skyblue')
                
                ax.set_title(f'Discharge at {col}', fontsize=13, fontweight='bold')
                ax.set_xlabel('Time Step', fontsize=11)
                ax.set_ylabel('Discharge (m³/s)', fontsize=11)
                ax.grid(True, alpha=0.3)
                
                # 添加统计
                stats_text = (
                    f"Peak: {df[col].max():.2f} m³/s\n"
                    f"Mean: {df[col].mean():.2f} m³/s\n"
                    f"Volume: {df[col].sum():.2f} m³"
                )
                ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                       fontsize=9)
                
                point_path = output_dir / f'discharge_{col}.png'
                self.plt.tight_layout()
                self.plt.savefig(point_path, dpi=150, bbox_inches='tight')
                self.plt.close(fig)
                output_paths.append(point_path)
            
            logger.info(f"Generated {len(output_paths)} discharge time series plots")
            return output_paths
        
        except Exception as e:
            logger.error(f"Failed to plot discharge time series: {e}")
            return []
    
    def calculate_runoff_coefficients(
        self,
        precipitation_csv: Path,
        discharge_csv: Path,
        watersheds_geojson: Path,
        output_path: Path
    ):
        """计算各分区的径流系数"""
        try:
            import pandas as pd
            import geopandas as gpd
            
            # 读取数据
            precip_df = pd.read_csv(precipitation_csv)
            discharge_df = pd.read_csv(discharge_csv)
            watersheds = gpd.read_file(watersheds_geojson)
            
            # 计算径流系数
            coefficients = {}
            
            # 获取列名（去掉时间列）
            precip_cols = [col for col in precip_df.columns if col != precip_df.columns[0]]
            discharge_cols = [col for col in discharge_df.columns if col != discharge_df.columns[0]]
            
            # 对每个流域计算
            for i, (pcol, dcol) in enumerate(zip(precip_cols, discharge_cols)):
                if pcol in precip_df.columns and dcol in discharge_df.columns:
                    # 总降雨量 (mm)
                    total_precip = precip_df[pcol].sum()
                    
                    # 总径流量 (m³) -> mm
                    total_discharge_m3 = discharge_df[dcol].sum()
                    
                    # 获取流域面积 (m²)
                    if i < len(watersheds):
                        area_m2 = watersheds.iloc[i].geometry.area
                        area_km2 = area_m2 / 1e6
                        
                        # 转换径流深 (mm)
                        if area_m2 > 0:
                            runoff_depth_mm = (total_discharge_m3 / area_m2) * 1000
                        else:
                            runoff_depth_mm = 0
                    else:
                        area_km2 = 100.0  # 默认值
                        runoff_depth_mm = total_discharge_m3 / 100000.0
                    
                    # 径流系数
                    if total_precip > 0:
                        coeff = runoff_depth_mm / total_precip
                    else:
                        coeff = 0.0
                    
                    coefficients[dcol] = {
                        'precipitation_mm': float(total_precip),
                        'runoff_mm': float(runoff_depth_mm),
                        'area_km2': float(area_km2),
                        'runoff_coefficient': float(coeff)
                    }
            
            # 保存到JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(coefficients, f, indent=2, ensure_ascii=False)
            
            # 生成可视化
            if self.plt and coefficients:
                fig, ax = self.plt.subplots(figsize=(12, 8))
                
                names = list(coefficients.keys())
                values = [coefficients[n]['runoff_coefficient'] for n in names]
                
                bars = ax.bar(range(len(names)), values, color='steelblue', alpha=0.7)
                ax.set_xticks(range(len(names)))
                ax.set_xticklabels(names, rotation=45, ha='right')
                ax.set_ylabel('Runoff Coefficient', fontsize=12)
                ax.set_title('Runoff Coefficients by Watershed', fontsize=14, fontweight='bold')
                ax.grid(True, axis='y', alpha=0.3)
                ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Reference: 0.5')
                ax.legend()
                
                # 添加数值标签
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=9)
                
                chart_path = output_path.with_suffix('.png')
                self.plt.tight_layout()
                self.plt.savefig(chart_path, dpi=150, bbox_inches='tight')
                self.plt.close(fig)
            
            logger.info(f"Runoff coefficients calculated and saved: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Failed to calculate runoff coefficients: {e}")
            return None
    
    def create_precipitation_runoff_comparison(
        self,
        precipitation_csv: Path,
        discharge_csv: Path,
        output_dir: Path
    ):
        """创建降雨径流对比图"""
        if not self.plt:
            return []
        
        try:
            import pandas as pd
            
            precip_df = pd.read_csv(precipitation_csv)
            discharge_df = pd.read_csv(discharge_csv)
            
            output_paths = []
            
            # 获取列名
            precip_cols = [col for col in precip_df.columns if col != precip_df.columns[0]]
            discharge_cols = [col for col in discharge_df.columns if col != discharge_df.columns[0]]
            
            # 对每个配对生成对比图
            for pcol, dcol in zip(precip_cols[:6], discharge_cols[:6]):  # 前6个
                fig, (ax1, ax2) = self.plt.subplots(2, 1, figsize=(14, 10), sharex=True)
                
                # 降雨图
                ax1.bar(precip_df.index, precip_df[pcol], color='blue', alpha=0.6)
                ax1.set_ylabel('Precipitation (mm)', fontsize=11, color='blue')
                ax1.set_title(f'Precipitation vs Runoff: {dcol}', fontsize=13, fontweight='bold')
                ax1.grid(True, alpha=0.3)
                ax1.tick_params(axis='y', labelcolor='blue')
                
                # 径流图
                ax2.plot(discharge_df.index, discharge_df[dcol], color='red', linewidth=2)
                ax2.fill_between(discharge_df.index, 0, discharge_df[dcol], alpha=0.3, color='red')
                ax2.set_xlabel('Time Step', fontsize=11)
                ax2.set_ylabel('Discharge (m³/s)', fontsize=11, color='red')
                ax2.grid(True, alpha=0.3)
                ax2.tick_params(axis='y', labelcolor='red')
                
                comparison_path = output_dir / f'precip_runoff_comparison_{dcol}.png'
                self.plt.tight_layout()
                self.plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
                self.plt.close(fig)
                output_paths.append(comparison_path)
            
            logger.info(f"Generated {len(output_paths)} precipitation-runoff comparison plots")
            return output_paths
        
        except Exception as e:
            logger.error(f"Failed to create precipitation-runoff comparison: {e}")
            return []
    
    def create_areal_precipitation_gif(
        self,
        areal_precip_csv: Path,
        watersheds_geojson: Path,
        output_path: Path,
        duration: float = 0.5
    ):
        """创建子流域面雨量动态GIF"""
        if not self.plt:
            return None
        
        try:
            import pandas as pd
            import geopandas as gpd
            import imageio
            from PIL import Image
            import tempfile
            
            # 读取数据
            df = pd.read_csv(areal_precip_csv)
            watersheds = gpd.read_file(watersheds_geojson)
            
            # 创建临时目录存储帧
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                frame_paths = []
                
                # 获取数据列（去掉时间列）
                data_cols = [col for col in df.columns if col != df.columns[0]]
                
                # 限制帧数，避免文件太大
                max_frames = min(50, len(df))
                step = max(1, len(df) // max_frames)
                
                for idx in range(0, len(df), step):
                    fig, ax = self.plt.subplots(figsize=(12, 10))
                    
                    # 获取当前时刻的降雨数据
                    current_precip = df.iloc[idx][data_cols].values
                    
                    # 确保数据长度匹配
                    if len(current_precip) == len(watersheds):
                        watersheds_copy = watersheds.copy()
                        watersheds_copy['precipitation'] = current_precip
                        
                        # 绘制
                        watersheds_copy.plot(
                            column='precipitation',
                            ax=ax,
                            cmap='Blues',
                            edgecolor='black',
                            linewidth=1.5,
                            legend=True,
                            vmin=0,
                            vmax=df[data_cols].max().max()
                        )
                    
                    ax.set_title(f'Areal Precipitation - Time Step {idx}',
                                fontsize=14, fontweight='bold')
                    ax.set_xlabel('X Coordinate', fontsize=12)
                    ax.set_ylabel('Y Coordinate', fontsize=12)
                    ax.grid(True, alpha=0.3)
                    
                    # 保存帧
                    frame_path = tmpdir / f'frame_{idx:04d}.png'
                    self.plt.tight_layout()
                    self.plt.savefig(frame_path, dpi=100, bbox_inches='tight')
                    self.plt.close(fig)
                    frame_paths.append(frame_path)
                
                # 创建GIF
                if frame_paths:
                    images = [Image.open(fp) for fp in frame_paths]
                    images[0].save(
                        output_path,
                        save_all=True,
                        append_images=images[1:],
                        duration=int(duration * 1000),
                        loop=0
                    )
                    
                    logger.info(f"Areal precipitation GIF saved: {output_path} ({len(frame_paths)} frames)")
                    return output_path
        
        except Exception as e:
            logger.error(f"Failed to create areal precipitation GIF: {e}")
            return None


class ComprehensiveTestRunner:
    """Comprehensive Test Runner"""
    
    def __init__(self, output_root: Path):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.test_results = {}
        self.visualizer = ComprehensiveVisualizer(self.output_root)
    
    def run_test_scenario(
        self,
        scenario_id: str,
        scenario_name: str,
        config_file: Path
    ) -> Dict[str, Any]:
        """运行单个测试场景"""
        logger.info("\n" + "=" * 80)
        logger.info(f"测试场景 {scenario_id}: {scenario_name}")
        logger.info("=" * 80)
        
        test_dir = self.output_root / f"{scenario_id}_{scenario_name.replace(' ', '_')}"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        result = {
            'scenario_id': scenario_id,
            'scenario_name': scenario_name,
            'config_file': str(config_file),
            'test_dir': str(test_dir),
            'start_time': datetime.now().isoformat(),
            'status': 'unknown',
            'visualizations': {}
        }
        
        try:
            if not config_file.exists():
                result['status'] = 'skipped'
                result['error'] = f"配置文件不存在: {config_file}"
                logger.warning(result['error'])
                return result
            
            # 加载并执行工作流
            from hydrosis.workflow_engine import WorkflowDefinition, WorkflowEngine
            
            logger.info(f"Loading workflow: {config_file}")
            workflow = WorkflowDefinition.from_yaml(config_file)
            
            engine = WorkflowEngine()
            run = engine.execute(workflow)
            
            result['status'] = run.status
            result['duration'] = time.time() - start_time
            result['run_id'] = run.run_id
            
            logger.info(f"\nWorkflow execution completed: {run.status}")
            
            # 生成所有可视化
            viz_dir = test_dir / "visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info("\nGenerating comprehensive visualizations...")
            result['visualizations'] = self._generate_comprehensive_visualizations(
                test_dir, viz_dir, run, workflow
            )
            
            logger.info(f"\nTest completed: {scenario_name}")
        
        except Exception as e:
            result['status'] = 'error'
            result['duration'] = time.time() - start_time
            result['error'] = str(e)
            logger.error(f"Test failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return result
    
    def _generate_comprehensive_visualizations(
        self,
        test_dir: Path,
        viz_dir: Path,
        run,
        workflow
    ) -> Dict[str, str]:
        """生成所有可视化"""
        visualizations = {}
        
        try:
            # 收集所有输出文件
            flow_dir_files = []
            pour_point_files = []
            gauge_files = []
            watershed_files = []
            precip_csv_files = []
            discharge_csv_files = []
            areal_precip_files = []
            dem_files = []
            
            for step_id, step_result in run.step_results.items():
                if step_result.status != 'completed' or not step_result.outputs:
                    continue
                
                for key, value in step_result.outputs.items():
                    if not isinstance(value, (str, Path)):
                        continue
                    
                    file_path = Path(value)
                    if not file_path.exists():
                        continue
                    
                    # 分类文件
                    if 'flow_direction' in key or 'flow_dir' in key:
                        flow_dir_files.append(file_path)
                    elif 'pour_point' in key and file_path.suffix == '.geojson':
                        pour_point_files.append(file_path)
                    elif 'rain_gauge' in key or 'gauge' in key:
                        if file_path.suffix == '.geojson':
                            gauge_files.append(file_path)
                    elif 'watershed' in key and file_path.suffix == '.geojson':
                        watershed_files.append(file_path)
                    elif 'precipitation' in key and file_path.suffix == '.csv':
                        if 'areal' in key:
                            areal_precip_files.append(file_path)
                        else:
                            precip_csv_files.append(file_path)
                    elif 'discharge' in key and file_path.suffix == '.csv':
                        discharge_csv_files.append(file_path)
                    elif 'elevation' in key or 'dem' in key:
                        if file_path.suffix == '.tif':
                            dem_files.append(file_path)
            
            # 1. 流向图
            for flow_dir_file in flow_dir_files:
                output_path = viz_dir / f'flow_direction_{flow_dir_file.stem}.png'
                result = self.visualizer.plot_flow_direction_correct(flow_dir_file, output_path)
                if result:
                    visualizations['flow_direction'] = str(result)
            
            # 2. 汇水点分布图（3干流+3支流）
            if pour_point_files and dem_files:
                output_path = viz_dir / 'pour_points_distribution.png'
                result = self.visualizer.plot_pour_points_distribution(
                    pour_point_files[0], dem_files[0], output_path,
                    mainstream_count=3, tributary_count=3
                )
                if result:
                    visualizations['pour_points'] = str(result)
            
            # 3. 雨量站分布图（50个）
            if gauge_files and watershed_files:
                output_path = viz_dir / 'rain_gauges_distribution.png'
                result = self.visualizer.plot_rain_gauges_distribution(
                    gauge_files[0], watershed_files[0], output_path, expected_count=50
                )
                if result:
                    visualizations['rain_gauges'] = str(result)
            
            # 4. 雨量站时间序列
            for precip_file in precip_csv_files:
                gauge_dir = viz_dir / 'gauge_timeseries'
                gauge_dir.mkdir(exist_ok=True)
                results = self.visualizer.plot_timeseries_all_gauges(precip_file, gauge_dir)
                if results:
                    visualizations['gauge_timeseries'] = [str(p) for p in results]
            
            # 5. 径流时间序列
            for discharge_file in discharge_csv_files:
                discharge_dir = viz_dir / 'discharge_timeseries'
                discharge_dir.mkdir(exist_ok=True)
                results = self.visualizer.plot_discharge_timeseries(discharge_file, discharge_dir)
                if results:
                    visualizations['discharge_timeseries'] = [str(p) for p in results]
            
            # 6. 径流系数
            if precip_csv_files and discharge_csv_files and watershed_files:
                output_path = viz_dir / 'runoff_coefficients.json'
                result = self.visualizer.calculate_runoff_coefficients(
                    precip_csv_files[0], discharge_csv_files[0],
                    watershed_files[0], output_path
                )
                if result:
                    visualizations['runoff_coefficients'] = str(result)
            
            # 7. 降雨径流对比
            if precip_csv_files and discharge_csv_files:
                comparison_dir = viz_dir / 'precip_runoff_comparison'
                comparison_dir.mkdir(exist_ok=True)
                results = self.visualizer.create_precipitation_runoff_comparison(
                    precip_csv_files[0], discharge_csv_files[0], comparison_dir
                )
                if results:
                    visualizations['precip_runoff_comparison'] = [str(p) for p in results]
            
            # 8. 面雨量动态GIF
            if areal_precip_files and watershed_files:
                output_path = viz_dir / 'areal_precipitation_animation.gif'
                result = self.visualizer.create_areal_precipitation_gif(
                    areal_precip_files[0], watershed_files[0], output_path
                )
                if result:
                    visualizations['areal_precip_animation'] = str(result)
            
            logger.info(f"   Generated {len(visualizations)} types of visualizations")
        
        except Exception as e:
            logger.error(f"Failed to generate comprehensive visualizations: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return visualizations
    
    def run_all_scenarios(self) -> bool:
        """运行所有测试场景"""
        logger.info("\n" + "#" * 80)
        logger.info("HydroSIS 综合测试场景运行器")
        logger.info("#" * 80)
        
        total_start = time.time()
        
        scenarios = [
            ("01", "Minimal_Terrain", "config/workflows/test_scenarios/01_minimal_terrain.yaml"),
            ("02", "Two_Step_Basic", "config/workflows/test_scenarios/02_two_step_basic.yaml"),
            ("03", "Watershed_Delineation", "config/workflows/test_scenarios/03_three_step_delineation.yaml"),
            ("04", "Precipitation_Analysis", "config/workflows/test_scenarios/04_precipitation_analysis.yaml"),
            ("05", "Hydrologic_Simulation", "config/workflows/test_scenarios/05_hydrologic_simulation.yaml"),
            ("06", "Calibration", "config/workflows/test_scenarios/06_calibration_workflow.yaml"),
            ("07", "Parallel_Analysis", "config/workflows/test_scenarios/07_parallel_analysis.yaml"),
            ("08", "Complete_Workflow", "config/workflows/test_scenarios/08_complete_eleven_steps.yaml"),
        ]
        
        for scenario_id, scenario_name, config_file in scenarios:
            result = self.run_test_scenario(scenario_id, scenario_name, Path(config_file))
            self.test_results[scenario_id] = result
        
        total_elapsed = time.time() - total_start
        
        # 生成总结
        passed = sum(1 for r in self.test_results.values() if r.get('status') == 'completed')
        total = len(self.test_results)
        
        logger.info("\n" + "#" * 80)
        logger.info("Test Summary")
        logger.info("#" * 80)
        logger.info(f"Total time: {total_elapsed:.2f} seconds")
        logger.info(f"Total tests: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {total - passed}")
        
        # 保存总结
        summary_path = self.output_root / "TEST_SUMMARY.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                'date': datetime.now().isoformat(),
                'total_time': total_elapsed,
                'total_tests': total,
                'passed': passed,
                'failed': total - passed,
                'results': self.test_results
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nSummary saved: {summary_path}")
        
        return passed == total


def main():
    """主函数"""
    output_dir = Path("results/comprehensive_test_scenarios")
    runner = ComprehensiveTestRunner(output_dir)
    
    success = runner.run_all_scenarios()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
