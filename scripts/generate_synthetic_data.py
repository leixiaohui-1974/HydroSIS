#!/usr/bin/env python3
"""
生成合成降雨和径流数据用于敏感性分析和参数率定

使用Upper Truckee River的气候特征
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def generate_precipitation(days=365, output_dir="results/synthetic_data"):
    """
    生成合成降雨序列
    
    基于Upper Truckee River（太浩湖流域）的气候特征：
    - 冬季降雪为主（11月-4月）
    - 夏季干燥（6月-9月）
    - 春季融雪（4月-6月）
    """
    print("="*80)
    print("生成合成降雨序列")
    print("="*80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 时间序列
    dates = pd.date_range('2020-01-01', periods=days, freq='D')
    
    # 月份
    months = dates.month
    
    # 基础降雨模式（mm/day）
    # Upper Truckee: 冬季湿润，夏季干燥
    monthly_mean = {
        1: 150,  # 1月 - 冬季降雪
        2: 130,  # 2月
        3: 100,  # 3月
        4: 60,   # 4月 - 融雪开始
        5: 40,   # 5月
        6: 20,   # 6月 - 干燥
        7: 10,   # 7月 - 最干燥
        8: 10,   # 8月
        9: 15,   # 9月
        10: 50,  # 10月 - 秋雨开始
        11: 100, # 11月
        12: 140  # 12月 - 冬季降雪
    }
    
    # 生成降雨
    precip = np.zeros(days)
    
    for i, (date, month) in enumerate(zip(dates, months)):
        # 月均降雨
        month_mean = monthly_mean[month]
        
        # 降雨事件概率（月度变化）
        if month in [6, 7, 8, 9]:  # 夏季
            rain_prob = 0.1
        elif month in [11, 12, 1, 2, 3]:  # 冬季
            rain_prob = 0.6
        else:  # 春秋
            rain_prob = 0.3
        
        # 是否降雨
        if np.random.random() < rain_prob:
            # 降雨强度（指数分布）
            intensity = np.random.exponential(month_mean / (rain_prob * 30))
            precip[i] = min(intensity, month_mean * 3)  # 限制最大值
    
    # 创建DataFrame
    df = pd.DataFrame({
        'date': dates,
        'precipitation_mm': precip
    })
    
    # 统计
    print(f"\n降雨统计:")
    print(f"  天数: {days}")
    print(f"  总降雨量: {precip.sum():.2f} mm")
    print(f"  年均降雨: {precip.sum():.2f} mm/year")
    print(f"  降雨天数: {np.sum(precip > 0)} 天")
    print(f"  最大日降雨: {precip.max():.2f} mm")
    print(f"  平均强度: {precip[precip > 0].mean():.2f} mm/day")
    
    # 月度统计
    df['month'] = df['date'].dt.month
    monthly_sum = df.groupby('month')['precipitation_mm'].sum()
    
    print(f"\n月度降雨分布:")
    for month in range(1, 13):
        print(f"  {month:2d}月: {monthly_sum[month]:6.1f} mm")
    
    # 保存
    output_path = output_dir / "precipitation.csv"
    df[['date', 'precipitation_mm']].to_csv(output_path, index=False)
    print(f"\n✅ 降雨数据已保存: {output_path}")
    
    # 可视化
    visualize_precipitation(df, output_dir)
    
    return df

def generate_runoff(precip_df, runoff_coeff=0.4, baseflow_index=0.3, 
                    output_dir="results/synthetic_data"):
    """
    基于降雨生成径流序列
    
    使用简化的水文模型:
    - 径流系数法生成地表径流
    - 指数衰减生成基流
    - 考虑融雪贡献
    """
    print("\n" + "="*80)
    print("生成合成径流序列")
    print("="*80)
    
    output_dir = Path(output_dir)
    
    precip = precip_df['precipitation_mm'].values
    dates = precip_df['date'].values
    n_days = len(precip)
    
    # 参数
    print(f"\n水文模型参数:")
    print(f"  径流系数: {runoff_coeff:.2f}")
    print(f"  基流指数: {baseflow_index:.2f}")
    print(f"  汇流时间: 1-2天")
    
    # 1. 地表径流（使用径流系数）
    surface_runoff = precip * runoff_coeff
    
    # 2. 地下水补给
    groundwater_recharge = precip * (1 - runoff_coeff) * 0.5
    
    # 3. 基流（指数衰减）
    baseflow = np.zeros(n_days)
    baseflow_storage = 10.0  # 初始基流存储（mm）
    recession_coeff = 0.95  # 衰退系数
    
    for i in range(n_days):
        # 补给
        baseflow_storage += groundwater_recharge[i]
        # 出流
        baseflow[i] = baseflow_storage * (1 - recession_coeff)
        # 更新存储
        baseflow_storage *= recession_coeff
    
    # 4. 融雪贡献（春季）
    snowmelt = np.zeros(n_days)
    months = pd.to_datetime(dates).month
    
    for i in range(n_days):
        if months[i] in [4, 5, 6]:  # 春季融雪
            # 温度相关的融雪（简化）
            day_of_month = pd.to_datetime(dates[i]).day
            snowmelt[i] = 20 * np.sin(np.pi * day_of_month / 30)  # mm/day
    
    # 5. 总径流
    total_runoff = surface_runoff + baseflow + snowmelt
    
    # 6. 汇流演算（简单的三角形单位线）
    routed_runoff = np.zeros(n_days)
    unit_hydrograph = np.array([0.2, 0.6, 0.2])  # 1天汇流时间
    
    for i in range(n_days):
        for j, weight in enumerate(unit_hydrograph):
            if i + j < n_days:
                routed_runoff[i + j] += total_runoff[i] * weight
    
    # 转换为流量（m³/s）
    # 假设流域面积100 km² (从控制面积)
    area_km2 = 100
    area_m2 = area_km2 * 1e6
    runoff_m3s = routed_runoff / 1000 * area_m2 / 86400  # mm/day -> m³/s
    
    # 创建DataFrame
    df = pd.DataFrame({
        'date': dates,
        'precipitation_mm': precip,
        'runoff_mm': routed_runoff,
        'runoff_m3s': runoff_m3s,
        'baseflow_mm': baseflow,
        'snowmelt_mm': snowmelt
    })
    
    # 统计
    print(f"\n径流统计:")
    print(f"  总径流量: {routed_runoff.sum():.2f} mm")
    print(f"  年均径流: {routed_runoff.sum():.2f} mm/year")
    print(f"  径流系数（实际）: {routed_runoff.sum() / precip.sum():.3f}")
    print(f"  最大流量: {runoff_m3s.max():.2f} m³/s")
    print(f"  平均流量: {runoff_m3s.mean():.2f} m³/s")
    print(f"  基流比例: {baseflow.sum() / routed_runoff.sum():.2f}")
    print(f"  融雪比例: {snowmelt.sum() / routed_runoff.sum():.2f}")
    
    # 保存
    output_path = output_dir / "runoff.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ 径流数据已保存: {output_path}")
    
    # 可视化
    visualize_runoff(df, output_dir)
    
    return df

def visualize_precipitation(df, output_dir):
    """可视化降雨"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 1. 日降雨
    ax = axes[0]
    ax.bar(df['date'], df['precipitation_mm'], width=1, edgecolor='none', alpha=0.7)
    ax.set_ylabel('Precipitation (mm/day)')
    ax.set_title('Daily Precipitation')
    ax.grid(True, alpha=0.3)
    
    # 2. 累积降雨
    ax = axes[1]
    ax.plot(df['date'], df['precipitation_mm'].cumsum(), 'b-', linewidth=2)
    ax.set_ylabel('Cumulative Precipitation (mm)')
    ax.set_title('Cumulative Precipitation')
    ax.grid(True, alpha=0.3)
    
    # 3. 月度降雨
    ax = axes[2]
    monthly = df.groupby(df['date'].dt.month)['precipitation_mm'].sum()
    ax.bar(monthly.index, monthly.values, color='steelblue', alpha=0.7)
    ax.set_xlabel('Month')
    ax.set_ylabel('Monthly Precipitation (mm)')
    ax.set_title('Monthly Precipitation Distribution')
    ax.set_xticks(range(1, 13))
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / 'precipitation_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ 降雨可视化: {output_path}")
    plt.close()

def visualize_runoff(df, output_dir):
    """可视化径流"""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    # 1. 降雨-径流对比
    ax = axes[0]
    ax2 = ax.twinx()
    ax.bar(df['date'], df['precipitation_mm'], width=1, color='blue', alpha=0.3, label='Precipitation')
    ax2.plot(df['date'], df['runoff_m3s'], 'r-', linewidth=1.5, label='Runoff')
    ax.set_ylabel('Precipitation (mm/day)', color='blue')
    ax2.set_ylabel('Runoff (m³/s)', color='red')
    ax.set_title('Precipitation-Runoff Relationship')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 2. 径流成分
    ax = axes[1]
    ax.fill_between(df['date'], 0, df['baseflow_mm'], alpha=0.5, label='Baseflow')
    ax.fill_between(df['date'], df['baseflow_mm'], 
                     df['baseflow_mm'] + df['snowmelt_mm'], 
                     alpha=0.5, label='Snowmelt')
    ax.fill_between(df['date'], df['baseflow_mm'] + df['snowmelt_mm'],
                     df['runoff_mm'], alpha=0.5, label='Surface runoff')
    ax.set_ylabel('Runoff components (mm/day)')
    ax.set_title('Runoff Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 流量过程线
    ax = axes[2]
    ax.plot(df['date'], df['runoff_m3s'], 'b-', linewidth=1.5)
    ax.fill_between(df['date'], 0, df['runoff_m3s'], alpha=0.3)
    ax.set_ylabel('Discharge (m³/s)')
    ax.set_title('Hydrograph')
    ax.grid(True, alpha=0.3)
    
    # 4. 流量历时曲线
    ax = axes[3]
    sorted_flow = np.sort(df['runoff_m3s'].values)[::-1]
    exceedance = np.arange(1, len(sorted_flow) + 1) / len(sorted_flow) * 100
    ax.semilogy(exceedance, sorted_flow, 'b-', linewidth=2)
    ax.set_xlabel('Exceedance Probability (%)')
    ax.set_ylabel('Discharge (m³/s)')
    ax.set_title('Flow Duration Curve')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    output_path = output_dir / 'runoff_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ 径流可视化: {output_path}")
    plt.close()

def main():
    output_dir = "results/synthetic_data"
    
    print("生成合成水文数据")
    print("="*80)
    print("目标: 为敏感性分析和参数率定提供测试数据")
    print("="*80)
    
    # 生成降雨
    precip_df = generate_precipitation(days=365, output_dir=output_dir)
    
    # 生成径流
    runoff_df = generate_runoff(
        precip_df, 
        runoff_coeff=0.4,  # 山地森林流域典型值
        baseflow_index=0.3,
        output_dir=output_dir
    )
    
    print("\n" + "="*80)
    print("数据生成完成")
    print("="*80)
    print(f"\n生成的文件:")
    print(f"  1. precipitation.csv - 降雨序列")
    print(f"  2. runoff.csv - 径流序列")
    print(f"  3. precipitation_visualization.png - 降雨图表")
    print(f"  4. runoff_visualization.png - 径流图表")
    
    return precip_df, runoff_df

if __name__ == "__main__":
    main()
