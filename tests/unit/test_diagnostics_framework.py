"""诊断框架单元测试"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from hydrosis.diagnostics import (
    BaseDiagnostic,
    DiagnosticResult,
    DiagnosticIssue,
    IssueSeverity,
    WaterBalanceDiagnostic,
    PrecipitationDiagnostic
)


@pytest.mark.unit
@pytest.mark.diagnostics
class TestDiagnosticIssue:
    """测试 DiagnosticIssue 数据类"""

    def test_issue_creation(self):
        """测试问题创建"""
        issue = DiagnosticIssue(
            category="test",
            severity=IssueSeverity.WARNING,
            message="Test message",
            details={"key": "value"},
            suggestion="Test suggestion"
        )

        assert issue.category == "test"
        assert issue.severity == IssueSeverity.WARNING
        assert issue.message == "Test message"
        assert issue.details == {"key": "value"}
        assert issue.suggestion == "Test suggestion"

    def test_issue_severity_levels(self):
        """测试严重程度级别"""
        assert IssueSeverity.INFO.value == "info"
        assert IssueSeverity.WARNING.value == "warning"
        assert IssueSeverity.ERROR.value == "error"
        assert IssueSeverity.CRITICAL.value == "critical"


@pytest.mark.unit
@pytest.mark.diagnostics
class TestDiagnosticResult:
    """测试 DiagnosticResult 数据类"""

    def test_result_creation(self):
        """测试结果创建"""
        result = DiagnosticResult(diagnostic_name="Test Diagnostic")

        assert result.diagnostic_name == "Test Diagnostic"
        assert result.issues == []
        assert result.metrics == {}
        assert result.figures == {}
        assert result.recommendations == []

    def test_result_with_data(self):
        """测试包含数据的结果"""
        issue1 = DiagnosticIssue(
            category="test",
            severity=IssueSeverity.WARNING,
            message="Warning message"
        )
        issue2 = DiagnosticIssue(
            category="test",
            severity=IssueSeverity.ERROR,
            message="Error message"
        )

        result = DiagnosticResult(
            diagnostic_name="Test",
            issues=[issue1, issue2],
            metrics={"metric1": 100, "metric2": 200},
            figures={"fig1": Path("test.png")},
            recommendations=["Fix this", "Fix that"]
        )

        assert len(result.issues) == 2
        assert result.metrics["metric1"] == 100
        assert "fig1" in result.figures
        assert len(result.recommendations) == 2

    def test_has_issues(self):
        """测试是否有问题"""
        result = DiagnosticResult(diagnostic_name="Test")
        assert result.has_issues() is False

        result.issues.append(DiagnosticIssue(
            category="test",
            severity=IssueSeverity.INFO,
            message="Info"
        ))
        assert result.has_issues() is True

    def test_get_issues_by_severity(self):
        """测试按严重程度获取问题"""
        result = DiagnosticResult(diagnostic_name="Test")
        result.issues = [
            DiagnosticIssue(category="t1", severity=IssueSeverity.INFO, message="i1"),
            DiagnosticIssue(category="t2", severity=IssueSeverity.WARNING, message="w1"),
            DiagnosticIssue(category="t3", severity=IssueSeverity.ERROR, message="e1"),
            DiagnosticIssue(category="t4", severity=IssueSeverity.WARNING, message="w2"),
        ]

        warnings = result.get_issues_by_severity(IssueSeverity.WARNING)
        assert len(warnings) == 2

        errors = result.get_issues_by_severity(IssueSeverity.ERROR)
        assert len(errors) == 1


@pytest.mark.unit
@pytest.mark.diagnostics
class TestBaseDiagnostic:
    """测试 BaseDiagnostic 抽象基类"""

    @pytest.fixture
    def temp_output_dir(self):
        """创建临时输出目录"""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_cannot_instantiate_abstract_class(self):
        """测试不能实例化抽象类"""
        with pytest.raises(TypeError):
            BaseDiagnostic()

    def test_concrete_diagnostic(self, temp_output_dir):
        """测试具体诊断类实现"""
        class ConcreteDiagnostic(BaseDiagnostic):
            @property
            def diagnostic_name(self):
                return "Concrete Test"

            def load_data(self, **kwargs):
                self._data_loaded = True

            def analyze(self):
                result = DiagnosticResult(diagnostic_name=self.diagnostic_name)
                result.metrics["test_metric"] = 42
                return result

        diagnostic = ConcreteDiagnostic(output_dir=temp_output_dir, verbose=False)
        assert diagnostic.diagnostic_name == "Concrete Test"

        # 运行诊断
        result = diagnostic.run()

        assert result.diagnostic_name == "Concrete Test"
        assert result.metrics["test_metric"] == 42
        assert diagnostic._data_loaded is True

    def test_add_issue(self, temp_output_dir):
        """测试添加问题"""
        class TestDiagnostic(BaseDiagnostic):
            @property
            def diagnostic_name(self):
                return "Test"

            def load_data(self, **kwargs):
                pass

            def analyze(self):
                self._result = DiagnosticResult(diagnostic_name=self.diagnostic_name)
                self.add_issue(
                    category="test",
                    severity=IssueSeverity.WARNING,
                    message="Test warning"
                )
                return self._result

        diagnostic = TestDiagnostic(output_dir=temp_output_dir, verbose=False)
        result = diagnostic.run()

        assert len(result.issues) == 1
        assert result.issues[0].category == "test"
        assert result.issues[0].severity == IssueSeverity.WARNING

    def test_add_metric_and_recommendation(self, temp_output_dir):
        """测试添加指标和建议"""
        class TestDiagnostic(BaseDiagnostic):
            @property
            def diagnostic_name(self):
                return "Test"

            def load_data(self, **kwargs):
                pass

            def analyze(self):
                self._result = DiagnosticResult(diagnostic_name=self.diagnostic_name)
                self.add_metric("metric1", 100)
                self.add_recommendation("Fix issue 1")
                return self._result

        diagnostic = TestDiagnostic(output_dir=temp_output_dir, verbose=False)
        result = diagnostic.run()

        assert result.metrics["metric1"] == 100
        assert "Fix issue 1" in result.recommendations


@pytest.mark.unit
@pytest.mark.diagnostics
class TestWaterBalanceDiagnostic:
    """测试 WaterBalanceDiagnostic"""

    @pytest.fixture
    def temp_output_dir(self):
        """创建临时输出目录"""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def normal_water_balance_data(self):
        """正常的水量平衡数据（RC < 1）"""
        np.random.seed(42)
        hours = 720  # 30天
        precipitation = np.random.gamma(2, 2, size=hours)  # mm/h
        runoff = precipitation * 0.5  # RC = 0.5
        return {
            "precipitation": precipitation,
            "runoff": runoff,
            "initial_lower": 100.0,  # 合理的初始储量
            "k2": 0.02
        }

    @pytest.fixture
    def anomalous_water_balance_data(self):
        """异常的水量平衡数据（RC > 1）"""
        np.random.seed(42)
        hours = 720  # 30天
        precipitation = np.random.gamma(2, 2, size=hours)
        runoff = precipitation * 1.5  # RC = 1.5 (异常)
        return {
            "precipitation": precipitation,
            "runoff": runoff,
            "initial_lower": 5000.0,  # 过大的初始储量
            "k2": 0.02
        }

    def test_diagnostic_creation(self, temp_output_dir):
        """测试诊断器创建"""
        diagnostic = WaterBalanceDiagnostic(
            output_dir=temp_output_dir,
            verbose=False,
            target_runoff_coefficient=0.5
        )

        assert diagnostic.diagnostic_name == "水量平衡诊断"
        assert diagnostic.target_runoff_coefficient == 0.5

    def test_load_data(self, temp_output_dir, normal_water_balance_data):
        """测试数据加载"""
        diagnostic = WaterBalanceDiagnostic(output_dir=temp_output_dir, verbose=False)
        diagnostic.load_data(**normal_water_balance_data)

        assert diagnostic._precipitation is not None
        assert diagnostic._runoff is not None
        assert diagnostic._initial_lower == 100.0
        assert diagnostic._k2 == 0.02

    def test_load_data_length_mismatch(self, temp_output_dir):
        """测试降雨和径流长度不匹配"""
        diagnostic = WaterBalanceDiagnostic(output_dir=temp_output_dir, verbose=False)

        with pytest.raises(ValueError, match="降雨和径流序列长度不一致"):
            diagnostic.load_data(
                precipitation=np.array([1, 2, 3]),
                runoff=np.array([1, 2]),
                initial_lower=100,
                k2=0.02
            )

    def test_normal_water_balance_analysis(self, temp_output_dir, normal_water_balance_data):
        """测试正常水量平衡分析"""
        diagnostic = WaterBalanceDiagnostic(output_dir=temp_output_dir, verbose=False)
        diagnostic.load_data(**normal_water_balance_data)
        result = diagnostic.analyze()

        assert result.diagnostic_name == "水量平衡诊断"
        assert "runoff_coefficient" in result.metrics
        assert "total_precipitation_mm" in result.metrics

        # 检查径流系数
        rc = result.metrics["runoff_coefficient"]
        assert 0 < rc < 1  # 正常情况

        # 不应该有严重错误
        errors = result.get_issues_by_severity(IssueSeverity.ERROR)
        assert len(errors) == 0

    def test_anomalous_water_balance_analysis(self, temp_output_dir, anomalous_water_balance_data):
        """测试异常水量平衡分析"""
        diagnostic = WaterBalanceDiagnostic(output_dir=temp_output_dir, verbose=False)
        diagnostic.load_data(**anomalous_water_balance_data)
        result = diagnostic.analyze()

        # 检查径流系数
        rc = result.metrics["runoff_coefficient"]
        assert rc > 1.0  # 异常情况

        # 应该有错误
        errors = result.get_issues_by_severity(IssueSeverity.ERROR)
        assert len(errors) > 0

        # 应该有修正建议
        assert len(result.recommendations) > 0

    def test_run_complete_workflow(self, temp_output_dir, normal_water_balance_data):
        """测试完整诊断流程"""
        diagnostic = WaterBalanceDiagnostic(output_dir=temp_output_dir, verbose=False)
        result = diagnostic.run(**normal_water_balance_data)

        assert result is not None
        assert isinstance(result, DiagnosticResult)

        # 检查报告文件是否生成
        report_files = list(temp_output_dir.glob("*.txt"))
        assert len(report_files) > 0

    def test_visualization(self, temp_output_dir, normal_water_balance_data):
        """测试可视化生成"""
        diagnostic = WaterBalanceDiagnostic(output_dir=temp_output_dir, verbose=False)
        diagnostic.load_data(**normal_water_balance_data)
        diagnostic.analyze()
        diagnostic.visualize()

        # 检查图表是否生成
        fig_files = list(temp_output_dir.glob("*.png"))
        assert len(fig_files) > 0


@pytest.mark.unit
@pytest.mark.diagnostics
class TestPrecipitationDiagnostic:
    """测试 PrecipitationDiagnostic"""

    @pytest.fixture
    def temp_output_dir(self):
        """创建临时输出目录"""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def normal_precipitation_data(self):
        """正常的降雨数据"""
        np.random.seed(42)
        hours = 720
        subbasins = ['1', '2', '3', '4', '5']

        # 生成相似的降雨数据
        precip_df = pd.DataFrame({
            sb: np.random.gamma(2, 2, size=hours) for sb in subbasins
        })

        # 子流域信息
        subbasins_df = pd.DataFrame({
            'subzone_id': [1, 2, 3, 4, 5],
            'zone_id': [1, 1, 2, 2, 2],
            'area_km2': [100, 150, 120, 130, 110]
        })

        return {
            "precipitation_df": precip_df,
            "subbasins_df": subbasins_df
        }

    @pytest.fixture
    def anomalous_precipitation_data(self):
        """异常的降雨数据（zone 2 降雨明显偏低）"""
        np.random.seed(42)
        hours = 720
        subbasins = ['1', '2', '3', '4', '5']

        # Zone 1 正常降雨，Zone 2 降雨偏低
        precip_df = pd.DataFrame({
            '1': np.random.gamma(2, 2, size=hours),
            '2': np.random.gamma(2, 2, size=hours),
            '3': np.random.gamma(2, 2, size=hours) * 0.3,  # 降雨偏低
            '4': np.random.gamma(2, 2, size=hours) * 0.3,
            '5': np.random.gamma(2, 2, size=hours) * 0.3,
        })

        subbasins_df = pd.DataFrame({
            'subzone_id': [1, 2, 3, 4, 5],
            'zone_id': [1, 1, 2, 2, 2],
            'area_km2': [100, 150, 120, 130, 110]
        })

        return {
            "precipitation_df": precip_df,
            "subbasins_df": subbasins_df
        }

    def test_diagnostic_creation(self, temp_output_dir):
        """测试诊断器创建"""
        diagnostic = PrecipitationDiagnostic(
            output_dir=temp_output_dir,
            verbose=False,
            anomaly_threshold=0.3
        )

        assert diagnostic.diagnostic_name == "降雨空间分布诊断"
        assert diagnostic.anomaly_threshold == 0.3

    def test_load_data(self, temp_output_dir, normal_precipitation_data):
        """测试数据加载"""
        diagnostic = PrecipitationDiagnostic(output_dir=temp_output_dir, verbose=False)
        diagnostic.load_data(**normal_precipitation_data)

        assert diagnostic._precipitation_df is not None
        assert diagnostic._subbasins_df is not None

    def test_load_data_missing_columns(self, temp_output_dir):
        """测试缺少必需列"""
        diagnostic = PrecipitationDiagnostic(output_dir=temp_output_dir, verbose=False)

        bad_subbasins_df = pd.DataFrame({
            'subzone_id': [1, 2],
            # 缺少 zone_id 和 area_km2
        })

        with pytest.raises(ValueError, match="缺少必需列"):
            diagnostic.load_data(
                precipitation_df=pd.DataFrame(),
                subbasins_df=bad_subbasins_df
            )

    def test_normal_precipitation_analysis(self, temp_output_dir, normal_precipitation_data):
        """测试正常降雨分析"""
        diagnostic = PrecipitationDiagnostic(output_dir=temp_output_dir, verbose=False)
        diagnostic.load_data(**normal_precipitation_data)
        result = diagnostic.analyze()

        assert result.diagnostic_name == "降雨空间分布诊断"

        # 检查分区指标
        assert "zone_1_weighted_avg_mm" in result.metrics
        assert "zone_2_weighted_avg_mm" in result.metrics

        # 正常情况下，不应该有太多严重错误
        errors = result.get_issues_by_severity(IssueSeverity.ERROR)
        assert len(errors) <= 1  # 可能有一些小问题

    def test_anomalous_precipitation_analysis(self, temp_output_dir, anomalous_precipitation_data):
        """测试异常降雨分析"""
        diagnostic = PrecipitationDiagnostic(
            output_dir=temp_output_dir,
            verbose=False,
            anomaly_threshold=0.3
        )
        diagnostic.load_data(**anomalous_precipitation_data)
        result = diagnostic.analyze()

        # 应该检测到分区间降雨差异
        zone_anomaly_issues = [
            issue for issue in result.issues
            if issue.category == "zone_precipitation_anomaly"
        ]
        assert len(zone_anomaly_issues) > 0

        # 应该有修正建议
        assert len(result.recommendations) > 0

    def test_data_quality_check(self, temp_output_dir):
        """测试数据质量检查"""
        # 创建包含负值的数据
        precip_df = pd.DataFrame({
            '1': [1, 2, -1, 4],  # 包含负值
            '2': [2, 3, 4, 5]
        })

        subbasins_df = pd.DataFrame({
            'subzone_id': [1, 2],
            'zone_id': [1, 1],
            'area_km2': [100, 100]
        })

        diagnostic = PrecipitationDiagnostic(output_dir=temp_output_dir, verbose=False)
        diagnostic.load_data(precipitation_df=precip_df, subbasins_df=subbasins_df)
        result = diagnostic.analyze()

        # 应该检测到负值
        quality_issues = [
            issue for issue in result.issues
            if issue.category == "data_quality" and "负值" in issue.message
        ]
        assert len(quality_issues) > 0

    def test_run_complete_workflow(self, temp_output_dir, normal_precipitation_data):
        """测试完整诊断流程"""
        diagnostic = PrecipitationDiagnostic(output_dir=temp_output_dir, verbose=False)
        result = diagnostic.run(**normal_precipitation_data)

        assert result is not None
        assert isinstance(result, DiagnosticResult)

        # 检查报告文件
        report_files = list(temp_output_dir.glob("*.txt"))
        assert len(report_files) > 0

    def test_visualization(self, temp_output_dir, normal_precipitation_data):
        """测试可视化生成"""
        diagnostic = PrecipitationDiagnostic(output_dir=temp_output_dir, verbose=False)
        diagnostic.load_data(**normal_precipitation_data)
        diagnostic.analyze()
        diagnostic.visualize()

        # 检查图表是否生成
        fig_files = list(temp_output_dir.glob("*.png"))
        assert len(fig_files) > 0
