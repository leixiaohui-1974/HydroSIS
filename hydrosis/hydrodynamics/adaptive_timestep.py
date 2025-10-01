"""自适应时间步长控制器

提供多种策略动态调整时间步长，确保数值稳定性的同时提高计算效率：
- CFL 条件控制
- 水深变化率控制
- 牛顿迭代性能控制
- 复合策略
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np


class AdaptiveStrategy(Enum):
    """自适应策略枚举"""
    CFL_BASED = "cfl"                    # 基于CFL数
    DEPTH_CHANGE = "depth_change"        # 基于水深变化率
    CONVERGENCE = "convergence"          # 基于收敛性能
    HYBRID = "hybrid"                    # 复合策略


@dataclass
class TimeStepMetrics:
    """时间步长度量指标"""
    
    current_dt: float              # 当前时间步长 (s)
    cfl_number: float             # CFL数
    max_depth_change_rate: float  # 最大水深变化率 (m/s)
    newton_iterations: int        # 牛顿迭代次数
    convergence_achieved: bool    # 是否收敛
    suggested_dt: float           # 建议的下一步时间步长 (s)
    reason: str                   # 调整原因


class AdaptiveTimeStepController:
    """自适应时间步长控制器
    
    根据模拟状态动态调整时间步长，平衡稳定性与效率
    """
    
    def __init__(self, 
                 initial_dt: float = 60.0,
                 min_dt: float = 10.0,
                 max_dt: float = 600.0,
                 target_cfl: float = 0.5,
                 max_cfl: float = 0.8,
                 strategy: AdaptiveStrategy = AdaptiveStrategy.HYBRID,
                 safety_factor: float = 0.9):
        """
        参数:
            initial_dt: 初始时间步长 (s)
            min_dt: 最小允许步长 (s)
            max_dt: 最大允许步长 (s)
            target_cfl: 目标CFL数
            max_cfl: 最大允许CFL数
            strategy: 自适应策略
            safety_factor: 安全系数 (调整幅度的折减)
        """
        self.current_dt = initial_dt
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.target_cfl = target_cfl
        self.max_cfl = max_cfl
        self.strategy = strategy
        self.safety_factor = safety_factor
        
        # 历史记录用于平滑调整
        self.dt_history: List[float] = [initial_dt]
        self.cfl_history: List[float] = []
        self.convergence_history: List[bool] = []
        
        # 统计信息
        self.total_steps = 0
        self.adjustments = 0
        self.failed_steps = 0
    
    def compute_cfl_number(self, velocity: np.ndarray, 
                          dx: float, dt: float) -> float:
        """计算CFL数
        
        CFL = max(|v|) * dt / dx
        
        参数:
            velocity: 流速数组 (m/s)
            dx: 空间步长 (m)
            dt: 时间步长 (s)
        
        返回:
            CFL数
        """
        max_velocity = np.max(np.abs(velocity))
        return max_velocity * dt / dx if dx > 0 else 0.0
    
    def compute_depth_change_rate(self, depth_current: np.ndarray,
                                  depth_previous: np.ndarray,
                                  dt: float) -> float:
        """计算水深变化率
        
        参数:
            depth_current: 当前水深 (m)
            depth_previous: 上一步水深 (m)
            dt: 时间步长 (s)
        
        返回:
            最大水深变化率 (m/s)
        """
        depth_change = np.abs(depth_current - depth_previous)
        return np.max(depth_change) / dt if dt > 0 else 0.0
    
    def adjust_by_cfl(self, cfl_current: float) -> tuple[float, str]:
        """基于CFL条件调整时间步长
        
        返回:
            (建议步长, 调整原因)
        """
        if cfl_current > self.max_cfl:
            # CFL过大，减小步长
            reduction_factor = self.target_cfl / cfl_current
            new_dt = self.current_dt * reduction_factor * self.safety_factor
            reason = f"CFL={cfl_current:.3f} 超限，减小步长"
        elif cfl_current < self.target_cfl * 0.5:
            # CFL过小，增大步长以提高效率
            increase_factor = self.target_cfl / cfl_current
            new_dt = self.current_dt * increase_factor * self.safety_factor
            reason = f"CFL={cfl_current:.3f} 过小，增大步长"
        else:
            # 在合理范围内，保持不变
            new_dt = self.current_dt
            reason = f"CFL={cfl_current:.3f} 合适，维持"
        
        return new_dt, reason
    
    def adjust_by_depth_change(self, depth_change_rate: float,
                               max_allowed_rate: float = 0.5) -> tuple[float, str]:
        """基于水深变化率调整
        
        参数:
            depth_change_rate: 当前水深变化率 (m/s)
            max_allowed_rate: 最大允许变化率 (m/s)
        
        返回:
            (建议步长, 调整原因)
        """
        if depth_change_rate > max_allowed_rate:
            reduction_factor = max_allowed_rate / depth_change_rate
            new_dt = self.current_dt * reduction_factor * self.safety_factor
            reason = f"水深变化率 {depth_change_rate:.3f} m/s 过快"
        elif depth_change_rate < max_allowed_rate * 0.2 and depth_change_rate > 1e-6:
            increase_factor = min(2.0, max_allowed_rate / depth_change_rate)
            new_dt = self.current_dt * increase_factor * self.safety_factor
            reason = f"水深变化率 {depth_change_rate:.3f} m/s 缓慢"
        else:
            new_dt = self.current_dt
            reason = "水深变化率合适"
        
        return new_dt, reason
    
    def adjust_by_convergence(self, newton_iterations: int,
                              converged: bool,
                              max_iterations: int = 20) -> tuple[float, str]:
        """基于牛顿迭代性能调整
        
        参数:
            newton_iterations: 实际迭代次数
            converged: 是否收敛
            max_iterations: 最大允许迭代次数
        
        返回:
            (建议步长, 调整原因)
        """
        if not converged:
            # 未收敛，大幅减小步长
            new_dt = self.current_dt * 0.5
            reason = f"牛顿迭代未收敛，减半步长"
        elif newton_iterations > max_iterations * 0.8:
            # 迭代次数接近上限，预防性减小
            new_dt = self.current_dt * 0.8
            reason = f"迭代 {newton_iterations} 次接近上限"
        elif newton_iterations < max_iterations * 0.3 and len(self.convergence_history) > 5:
            # 收敛快且稳定，可增大步长
            if all(self.convergence_history[-5:]):
                new_dt = self.current_dt * 1.2
                reason = f"迭代 {newton_iterations} 次快速收敛"
            else:
                new_dt = self.current_dt
                reason = "收敛快但历史不稳定"
        else:
            new_dt = self.current_dt
            reason = f"迭代 {newton_iterations} 次正常"
        
        return new_dt, reason
    
    def update(self, 
               velocity: np.ndarray,
               depth_current: np.ndarray,
               depth_previous: np.ndarray,
               dx: float,
               newton_iterations: int = 0,
               converged: bool = True) -> TimeStepMetrics:
        """更新时间步长并返回度量信息
        
        参数:
            velocity: 当前流速场 (m/s)
            depth_current: 当前水深 (m)
            depth_previous: 上一步水深 (m)
            dx: 空间步长 (m)
            newton_iterations: 牛顿迭代次数
            converged: 是否收敛
        
        返回:
            TimeStepMetrics 对象
        """
        # 计算指标
        cfl = self.compute_cfl_number(velocity, dx, self.current_dt)
        depth_change_rate = self.compute_depth_change_rate(
            depth_current, depth_previous, self.current_dt
        )
        
        # 根据策略选择调整方法
        if self.strategy == AdaptiveStrategy.CFL_BASED:
            suggested_dt, reason = self.adjust_by_cfl(cfl)
        
        elif self.strategy == AdaptiveStrategy.DEPTH_CHANGE:
            suggested_dt, reason = self.adjust_by_depth_change(depth_change_rate)
        
        elif self.strategy == AdaptiveStrategy.CONVERGENCE:
            suggested_dt, reason = self.adjust_by_convergence(
                newton_iterations, converged
            )
        
        elif self.strategy == AdaptiveStrategy.HYBRID:
            # 复合策略：取最保守的建议
            dt_cfl, reason_cfl = self.adjust_by_cfl(cfl)
            dt_depth, reason_depth = self.adjust_by_depth_change(depth_change_rate)
            dt_conv, reason_conv = self.adjust_by_convergence(
                newton_iterations, converged
            )
            
            suggested_dt = min(dt_cfl, dt_depth, dt_conv)
            
            # 确定主要限制因素
            if suggested_dt == dt_cfl:
                reason = f"复合策略: CFL主导 ({reason_cfl})"
            elif suggested_dt == dt_depth:
                reason = f"复合策略: 水深变化主导 ({reason_depth})"
            else:
                reason = f"复合策略: 收敛性主导 ({reason_conv})"
        
        else:
            suggested_dt = self.current_dt
            reason = "未知策略"
        
        # 限制在允许范围内
        suggested_dt = np.clip(suggested_dt, self.min_dt, self.max_dt)
        
        # 平滑调整：避免剧烈变化
        if len(self.dt_history) > 0:
            max_change_ratio = 2.0
            suggested_dt = np.clip(
                suggested_dt,
                self.current_dt / max_change_ratio,
                self.current_dt * max_change_ratio
            )
        
        # 更新历史
        self.cfl_history.append(cfl)
        self.convergence_history.append(converged)
        self.total_steps += 1
        
        if abs(suggested_dt - self.current_dt) > 1.0:
            self.adjustments += 1
        
        if not converged:
            self.failed_steps += 1
        
        # 创建度量对象
        metrics = TimeStepMetrics(
            current_dt=self.current_dt,
            cfl_number=cfl,
            max_depth_change_rate=depth_change_rate,
            newton_iterations=newton_iterations,
            convergence_achieved=converged,
            suggested_dt=suggested_dt,
            reason=reason
        )
        
        # 应用新步长
        self.current_dt = suggested_dt
        self.dt_history.append(suggested_dt)
        
        # 限制历史长度
        if len(self.dt_history) > 100:
            self.dt_history = self.dt_history[-100:]
        if len(self.cfl_history) > 100:
            self.cfl_history = self.cfl_history[-100:]
        if len(self.convergence_history) > 100:
            self.convergence_history = self.convergence_history[-100:]
        
        return metrics
    
    def get_statistics(self) -> dict:
        """获取统计信息"""
        avg_dt = np.mean(self.dt_history) if self.dt_history else 0
        avg_cfl = np.mean(self.cfl_history) if self.cfl_history else 0
        success_rate = (self.total_steps - self.failed_steps) / max(self.total_steps, 1)
        
        return {
            'total_steps': self.total_steps,
            'adjustments': self.adjustments,
            'failed_steps': self.failed_steps,
            'current_dt': self.current_dt,
            'avg_dt': avg_dt,
            'min_dt_used': min(self.dt_history) if self.dt_history else 0,
            'max_dt_used': max(self.dt_history) if self.dt_history else 0,
            'avg_cfl': avg_cfl,
            'success_rate': success_rate * 100
        }
    
    def reset(self, new_initial_dt: Optional[float] = None):
        """重置控制器"""
        if new_initial_dt is not None:
            self.current_dt = new_initial_dt
        self.dt_history = [self.current_dt]
        self.cfl_history = []
        self.convergence_history = []
        self.total_steps = 0
        self.adjustments = 0
        self.failed_steps = 0


class VariableTimeStepSimulator:
    """可变时间步长模拟器包装类
    
    与固定步长求解器配合使用，自动管理时间步长调整
    """
    
    def __init__(self, controller: AdaptiveTimeStepController):
        self.controller = controller
        self.metrics_history: List[TimeStepMetrics] = []
    
    def run_adaptive_simulation(self,
                                solver,  # 原求解器对象
                                total_time: float,
                                boundary_conditions,
                                verbose: bool = True) -> dict:
        """运行自适应时间步长模拟
        
        参数:
            solver: 原求解器实例 (需有state, solve_timestep方法)
            total_time: 总模拟时间 (s)
            boundary_conditions: 边界条件对象
            verbose: 是否输出详细信息
        
        返回:
            结果字典 {'time': [...], 'discharge': [...], 'depth': [...]}
        """
        results = {
            'time': [],
            'discharge': [],
            'depth': [],
            'velocity': [],
            'dt': []
        }
        
        current_time = 0.0
        time_step_index = 0
        
        depth_prev = solver.state.depth.copy()
        
        if verbose:
            print(f"开始自适应模拟 (目标时间: {total_time}s)")
            print(f"初始步长: {self.controller.current_dt}s")
            print("-" * 70)
        
        while current_time < total_time:
            # 更新求解器时间步长
            solver.dt = self.controller.current_dt
            
            # 执行一步计算
            converged = solver.solve_timestep(boundary_conditions, time_step_index)
            
            # 获取当前状态
            velocity = solver.state.velocity
            depth_current = solver.state.depth
            dx = solver.reach.dx
            
            # 估算牛顿迭代次数 (如果求解器不提供，使用默认值)
            newton_iters = getattr(solver, 'last_newton_iterations', 10)
            
            # 更新时间步长控制器
            metrics = self.controller.update(
                velocity, depth_current, depth_prev,
                dx, newton_iters, converged
            )
            
            self.metrics_history.append(metrics)
            
            # 记录结果
            results['time'].append(current_time)
            results['discharge'].append(solver.state.discharge.copy())
            results['depth'].append(depth_current.copy())
            results['velocity'].append(velocity.copy())
            results['dt'].append(self.controller.current_dt)
            
            # 更新时间
            current_time += metrics.current_dt
            time_step_index += 1
            
            # 保存当前水深供下一步使用
            depth_prev = depth_current.copy()
            
            # 定期输出进度
            if verbose and time_step_index % 10 == 0:
                progress = current_time / total_time * 100
                print(f"步骤 {time_step_index:4d} | "
                      f"时间 {current_time:7.1f}s ({progress:5.1f}%) | "
                      f"dt={metrics.current_dt:5.1f}s | "
                      f"CFL={metrics.cfl_number:.3f} | "
                      f"{metrics.reason}")
        
        if verbose:
            print("-" * 70)
            stats = self.controller.get_statistics()
            print(f"模拟完成！")
            print(f"  总步数: {stats['total_steps']}")
            print(f"  步长调整次数: {stats['adjustments']}")
            print(f"  平均步长: {stats['avg_dt']:.1f}s")
            print(f"  步长范围: {stats['min_dt_used']:.1f} - {stats['max_dt_used']:.1f}s")
            print(f"  平均CFL: {stats['avg_cfl']:.3f}")
            print(f"  成功率: {stats['success_rate']:.1f}%")
        
        return results
    
    def plot_metrics(self, save_path: str = 'adaptive_metrics.png'):
        """绘制自适应控制指标图表"""
        if not self.dt_history:
            print("无历史数据可绘制")
            return
            
        import matplotlib.pyplot as plt
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        times = np.cumsum(self.dt_history)
        
        # 时间步长历史
        ax1.plot(times, self.dt_history, 'b-', linewidth=2)
        ax1.axhline(y=self.min_dt, color='r', linestyle='--', alpha=0.7, label=f'最小步长 {self.min_dt}s')
        ax1.axhline(y=self.max_dt, color='g', linestyle='--', alpha=0.7, label=f'最大步长 {self.max_dt}s')
        ax1.set_xlabel('累积时间 (s)')
        ax1.set_ylabel('时间步长 (s)')
        ax1.set_title('自适应时间步长历史')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # CFL数历史
        if self.cfl_history:
            ax2.plot(times, self.cfl_history, 'r-', linewidth=2)
            ax2.axhline(y=self.target_cfl, color='k', linestyle='--', alpha=0.7, label=f'目标CFL {self.target_cfl}')
            ax2.set_xlabel('累积时间 (s)')
            ax2.set_ylabel('CFL数')
            ax2.set_title('CFL数监控')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 调整策略分布
        if hasattr(self, 'strategy_history') and self.strategy_history:
            strategy_counts = {}
            for strategy in self.strategy_history:
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
            strategies = list(strategy_counts.keys())
            counts = list(strategy_counts.values())
            
            ax3.pie(counts, labels=strategies, autopct='%1.1f%%')
            ax3.set_title('调整策略分布')
        
        # 效率指标
        total_time = sum(self.dt_history)
        fixed_dt_time = len(self.dt_history) * self.initial_dt
        efficiency = fixed_dt_time / total_time if total_time > 0 else 1
        
        ax4.bar(['固定步长', '自适应步长'], [fixed_dt_time, total_time], 
                color=['lightblue', 'lightgreen'])
        ax4.set_ylabel('总计算时间 (s)')
        ax4.set_title(f'效率对比 (提升 {efficiency:.1f}x)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")


if __name__ == "__main__":
    # 测试示例
    print("="*70)
    print("自适应时间步长控制器测试")
    print("="*70)
    
    # 创建控制器
    controller = AdaptiveTimeStepController(
        initial_dt=60,
        min_dt=10,
        max_dt=300,
        target_cfl=0.5,
        strategy=AdaptiveStrategy.HYBRID
    )
    
    # 模拟一系列状态变化
    print("\n模拟场景: 洪水过程")
    print("-" * 70)
    
    # 模拟流速和水深变化
    num_sections = 20
    dx = 500.0
    
    for step in range(20):
        # 模拟洪水涨落过程
        t_normalized = step / 20.0
        peak_velocity = 2.0 + 3.0 * np.sin(t_normalized * np.pi)
        
        velocity = np.full(num_sections, peak_velocity) + \
                  np.random.normal(0, 0.2, num_sections)
        depth_current = np.full(num_sections, 2.0 + peak_velocity * 0.5)
        depth_previous = depth_current - 0.1 * np.random.random(num_sections)
        
        # 模拟收敛情况
        converged = np.random.random() > 0.1
        newton_iters = np.random.randint(5, 15) if converged else 25
        
        # 更新控制器
        metrics = controller.update(
            velocity, depth_current, depth_previous,
            dx, newton_iters, converged
        )
        
        if step % 5 == 0:
            print(f"步骤 {step:2d}: dt={metrics.current_dt:5.1f}s | "
                  f"CFL={metrics.cfl_number:.3f} | "
                  f"收敛={metrics.convergence_achieved} | "
                  f"{metrics.reason}")
    
    # 显示统计
    print("\n" + "="*70)
    stats = controller.get_statistics()
    print("统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")