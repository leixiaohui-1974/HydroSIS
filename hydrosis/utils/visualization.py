"""可视化工具模块

用于生成测试结果的各种图表和动态GIF
"""
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
import json

logger = logging.getLogger(__name__)


class ResultVisualizer:
    """结果可视化器"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_raster(
        self,
        raster_path: Path,
        output_path: Path,
        title: str = "",
        cmap: str = "terrain",
        figsize: Tuple[int, int] = (10, 8)
    ) -> Path:
        """绘制栅格数据
        
        Args:
            raster_path: 栅格文件路径
            output_path: 输出图片路径
            title: 图片标题
            cmap: 颜色映射
            figsize: 图片大小
        
        Returns:
            输出文件路径
        """
        try:
            import rasterio
            from rasterio.plot import show
            
            with rasterio.open(raster_path) as src:
                data = src.read(1)
                
                # 处理nodata
                if src.nodata is not None:
                    data = np.ma.masked_equal(data, src.nodata)
                
                fig, ax = plt.subplots(figsize=figsize)
                
                # 绘制
                im = ax.imshow(data, cmap=cmap)
                ax.set_title(title, fontsize=14, pad=10)
                ax.set_xlabel("Column", fontsize=10)
                ax.set_ylabel("Row", fontsize=10)
                
                # 添加colorbar
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(title, fontsize=10)
                
                # 添加统计信息
                valid_data = data.compressed() if np.ma.is_masked(data) else data.flatten()
                stats_text = (
                    f"Min: {valid_data.min():.2f}\n"
                    f"Max: {valid_data.max():.2f}\n"
            f"Mean: {float(valid_data.mean()):.2f}\n"
            f"Std: {float(valid_data.std()):.2f}"
                )
                ax.text(
                    0.02, 0.98, stats_text,
                    transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=9
                )
                
                plt.tight_layout()
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                logger.info(f"栅格图已保存: {output_path}")
                return output_path
        
        except Exception as e:
            logger.error(f"绘制栅格失败: {e}")
            return None
    
    def plot_vector(
        self,
        vector_path: Path,
        output_path: Path,
        title: str = "",
        figsize: Tuple[int, int] = (10, 8),
        basemap_raster: Optional[Path] = None
    ) -> Path:
        """绘制矢量数据
        
        Args:
            vector_path: 矢量文件路径（GeoJSON）
            output_path: 输出图片路径
            title: 图片标题
            figsize: 图片大小
            basemap_raster: 底图栅格（可选）
        
        Returns:
            输出文件路径
        """
        try:
            import json
            from shapely.geometry import shape
            
            # 读取GeoJSON
            with open(vector_path, 'r') as f:
                geojson_data = json.load(f)
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # 如果有底图，先绘制底图
            if basemap_raster and basemap_raster.exists():
                import rasterio
                with rasterio.open(basemap_raster) as src:
                    data = src.read(1)
                    if src.nodata is not None:
                        data = np.ma.masked_equal(data, src.nodata)
                    ax.imshow(data, cmap='terrain', alpha=0.5)
            
            # 绘制矢量
            features = geojson_data.get('features', [])
            for i, feature in enumerate(features):
                geom = shape(feature['geometry'])
                
                if geom.geom_type == 'Point':
                    ax.plot(geom.x, geom.y, 'ro', markersize=8, alpha=0.7)
                    # 添加标签
                    props = feature.get('properties', {})
                    label = props.get('id', props.get('name', i))
                    ax.text(geom.x, geom.y, str(label), fontsize=8, ha='right')
                
                elif geom.geom_type == 'LineString':
                    x, y = geom.xy
                    ax.plot(x, y, 'b-', linewidth=2, alpha=0.7)
                
                elif geom.geom_type in ['Polygon', 'MultiPolygon']:
                    if geom.geom_type == 'Polygon':
                        polygons = [geom]
                    else:
                        polygons = list(geom.geoms)
                    
                    for poly in polygons:
                        x, y = poly.exterior.xy
                        ax.plot(x, y, 'g-', linewidth=1.5)
                        ax.fill(x, y, alpha=0.3)
            
            ax.set_title(title, fontsize=14, pad=10)
            ax.set_xlabel("X", fontsize=10)
            ax.set_ylabel("Y", fontsize=10)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # 添加统计信息
            stats_text = f"Features: {len(features)}"
            ax.text(
                0.02, 0.98, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=9
            )
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"矢量图已保存: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"绘制矢量失败: {e}")
            return None
    
    def plot_timeseries(
        self,
        csv_path: Path,
        output_path: Path,
        title: str = "",
        figsize: Tuple[int, int] = (12, 6)
    ) -> Path:
        """绘制时间序列
        
        Args:
            csv_path: CSV文件路径
            output_path: 输出图片路径
            title: 图片标题
            figsize: 图片大小
        
        Returns:
            输出文件路径
        """
        try:
            import pandas as pd
            
            # 读取CSV
            df = pd.read_csv(csv_path)
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # 绘制所有数值列
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                ax.plot(df.index, df[col], label=col, linewidth=1.5)
            
            ax.set_title(title, fontsize=14, pad=10)
            ax.set_xlabel("Time Step", fontsize=10)
            ax.set_ylabel("Value", fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # 添加统计信息
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                stats_text = (
                    f"Points: {len(df)}\n"
                    f"Min: {df[col].min():.2f}\n"
                    f"Max: {df[col].max():.2f}\n"
                    f"Mean: {df[col].mean():.2f}"
                )
                ax.text(
                    0.02, 0.98, stats_text,
                    transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=9
                )
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"时间序列图已保存: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"绘制时间序列失败: {e}")
            return None
    
    def create_comparison_plot(
        self,
        data_dict: Dict[str, np.ndarray],
        output_path: Path,
        title: str = "Comparison",
        figsize: Tuple[int, int] = (15, 5)
    ) -> Path:
        """创建对比图
        
        Args:
            data_dict: 数据字典 {name: data_array}
            output_path: 输出路径
            title: 标题
            figsize: 图片大小
        
        Returns:
            输出文件路径
        """
        try:
            n_plots = len(data_dict)
            fig, axes = plt.subplots(1, n_plots, figsize=figsize)
            
            if n_plots == 1:
                axes = [axes]
            
            for ax, (name, data) in zip(axes, data_dict.items()):
                im = ax.imshow(data, cmap='terrain')
                ax.set_title(name, fontsize=12)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            fig.suptitle(title, fontsize=14)
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"对比图已保存: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"创建对比图失败: {e}")
            return None
    
    def create_animation(
        self,
        image_paths: List[Path],
        output_path: Path,
        duration: float = 0.5
    ) -> Path:
        """创建动画GIF
        
        Args:
            image_paths: 图片路径列表
            output_path: 输出GIF路径
            duration: 每帧持续时间（秒）
        
        Returns:
            输出文件路径
        """
        try:
            import imageio
            from PIL import Image
            
            images = []
            for img_path in image_paths:
                if img_path.exists():
                    img = Image.open(img_path)
                    images.append(np.array(img))
            
            if images:
                imageio.mimsave(
                    output_path,
                    images,
                    duration=duration,
                    loop=0
                )
                logger.info(f"动画GIF已保存: {output_path}")
                return output_path
            else:
                logger.warning("没有图片可用于创建动画")
                return None
        
        except Exception as e:
            logger.error(f"创建动画失败: {e}")
            return None
    
    def create_summary_figure(
        self,
        test_name: str,
        results: Dict[str, Any],
        output_path: Path
    ) -> Path:
        """创建测试总结图
        
        Args:
            test_name: 测试名称
            results: 测试结果
            output_path: 输出路径
        
        Returns:
            输出文件路径
        """
        try:
            fig = plt.figure(figsize=(14, 10))
            
            # 标题
            fig.suptitle(f"Test Summary: {test_name}", fontsize=16, fontweight='bold')
            
            # 创建网格布局
            gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
            
            # 1. 测试状态
            ax1 = fig.add_subplot(gs[0, :])
            ax1.axis('off')
            
            status_text = f"""
Test Name: {test_name}
Status: {results.get('status', 'Unknown')}
Duration: {results.get('duration', 0):.2f} seconds
Steps: {results.get('step_count', 0)}
            """
            ax1.text(0.1, 0.5, status_text, fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            # 2-6. 预留给其他图表
            for i in range(4):
                ax = fig.add_subplot(gs[1 + i//2, i%2])
                ax.text(0.5, 0.5, f'Chart {i+2}', ha='center', va='center')
                ax.set_title(f"Placeholder {i+2}")
            
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"总结图已保存: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"创建总结图失败: {e}")
            return None


def create_test_report(
    test_dir: Path,
    test_name: str,
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    validation: Dict[str, Any],
    visualizations: Dict[str, Path]
) -> Path:
    """创建测试详细报告
    
    Args:
        test_dir: 测试目录
        test_name: 测试名称
        inputs: 输入信息
        outputs: 输出信息
        validation: 验证结果
        visualizations: 可视化文件
    
    Returns:
        报告文件路径
    """
    report_path = test_dir / "TEST_REPORT.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# {test_name} - 测试报告\n\n")
        
        f.write("## 测试概览\n\n")
        f.write(f"- **测试名称**: {test_name}\n")
        f.write(f"- **执行时间**: {validation.get('timestamp', 'N/A')}\n")
        f.write(f"- **状态**: {'✅ 通过' if validation.get('passed', False) else '❌ 失败'}\n")
        f.write(f"- **耗时**: {validation.get('duration', 0):.2f}秒\n\n")
        
        f.write("## 输入参数\n\n")
        f.write("```yaml\n")
        import yaml
        f.write(yaml.dump(inputs, default_flow_style=False, allow_unicode=True))
        f.write("```\n\n")
        
        f.write("## 输出结果\n\n")
        if outputs:
            f.write("| 输出项 | 路径 | 状态 |\n")
            f.write("|--------|------|------|\n")
            for key, value in outputs.items():
                exists = Path(value).exists() if isinstance(value, (str, Path)) else False
                status = "✅" if exists else "❌"
                f.write(f"| {key} | `{value}` | {status} |\n")
        f.write("\n")
        
        f.write("## 验证结果\n\n")
        if validation.get('errors'):
            f.write("### ❌ 错误\n\n")
            for error in validation['errors']:
                f.write(f"- {error}\n")
            f.write("\n")
        
        if validation.get('warnings'):
            f.write("### ⚠️ 警告\n\n")
            for warning in validation['warnings']:
                f.write(f"- {warning}\n")
            f.write("\n")
        
        if validation.get('metrics'):
            f.write("### 📊 指标\n\n")
            for key, value in validation['metrics'].items():
                f.write(f"- **{key}**: {value}\n")
            f.write("\n")
        
        f.write("## 可视化结果\n\n")
        for name, path in visualizations.items():
            if path and path.exists():
                rel_path = path.relative_to(test_dir)
                f.write(f"### {name}\n\n")
                f.write(f"![{name}]({rel_path})\n\n")
        
        f.write("## 结论\n\n")
        if validation.get('passed', False):
            f.write("✅ **测试通过** - 所有验证项都满足要求\n")
        else:
            f.write("❌ **测试失败** - 存在错误需要修复\n")
    
    logger.info(f"测试报告已生成: {report_path}")
    return report_path
