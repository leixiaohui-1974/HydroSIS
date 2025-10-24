"""ConfigManager的单元测试

测试配置管理器的所有功能：
- 单例模式
- 配置加载和缓存
- 环境变量替换
- 路径解析
- 配置验证
"""
import pytest
import os
import tempfile
from pathlib import Path

from hydrosis.config.manager import ConfigManager


@pytest.fixture
def temp_config_dir():
    """创建临时配置目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config_file(temp_config_dir):
    """创建示例配置文件"""
    config_content = """
project:
  name: test_project
  version: 1.0

directories:
  base_results: results
  data: data/input
  
paths:
  precipitation: data/precip.csv
  dem: ${DEM_PATH}/elevation.tif
  
model:
  hbv:
    FC: 300
    BETA: 2.0
  parameters:
    zones: 4
"""
    config_file = temp_config_dir / "test_config.yaml"
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def nested_config_file(temp_config_dir):
    """创建嵌套配置文件"""
    config_content = """
level1:
  level2:
    level3:
      value: deep_value
  simple: simple_value
"""
    config_file = temp_config_dir / "nested.yaml"
    config_file.write_text(config_content)
    return config_file


@pytest.fixture(autouse=True)
def reset_singleton():
    """每个测试前后重置ConfigManager单例"""
    yield
    ConfigManager.reset_instance()


class TestConfigManagerSingleton:
    """测试ConfigManager单例模式"""
    
    def test_singleton_same_instance(self):
        """测试多次实例化返回同一个对象"""
        mgr1 = ConfigManager()
        mgr2 = ConfigManager()
        
        assert mgr1 is mgr2
    
    def test_singleton_shared_cache(self, sample_config_file):
        """测试单例共享缓存"""
        mgr1 = ConfigManager()
        mgr2 = ConfigManager()
        
        # 第一个实例加载配置
        config1 = mgr1.load_config(sample_config_file)
        
        # 第二个实例应该能访问缓存
        cached = mgr2.get_cached_config(sample_config_file)
        
        assert cached is not None
        assert cached == config1
    
    def test_reset_instance(self):
        """测试重置单例"""
        mgr1 = ConfigManager()
        ConfigManager.reset_instance()
        mgr2 = ConfigManager()
        
        # 重置后应该是新实例
        assert mgr1 is not mgr2


class TestConfigLoading:
    """测试配置加载功能"""
    
    def test_load_valid_config(self, sample_config_file):
        """测试加载有效配置"""
        mgr = ConfigManager()
        config = mgr.load_config(sample_config_file)
        
        assert config is not None
        assert "project" in config
        assert config["project"]["name"] == "test_project"
        assert config["model"]["hbv"]["FC"] == 300
    
    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        mgr = ConfigManager()
        
        with pytest.raises(FileNotFoundError):
            mgr.load_config("nonexistent.yaml")
    
    def test_config_caching(self, sample_config_file):
        """测试配置缓存"""
        mgr = ConfigManager()
        
        # 第一次加载
        config1 = mgr.load_config(sample_config_file)
        
        # 第二次加载应该使用缓存（返回副本）
        config2 = mgr.load_config(sample_config_file)
        
        assert config1 == config2
        # 应该是不同的对象（副本）
        assert config1 is not config2
    
    def test_load_without_cache(self, sample_config_file):
        """测试不使用缓存加载"""
        mgr = ConfigManager()
        
        config1 = mgr.load_config(sample_config_file, use_cache=True)
        config2 = mgr.load_config(sample_config_file, use_cache=False)
        
        # 内容应该相同
        assert config1 == config2


class TestEnvironmentVariables:
    """测试环境变量替换"""
    
    def test_env_var_replacement(self, sample_config_file):
        """测试环境变量替换"""
        # 设置环境变量
        os.environ['DEM_PATH'] = '/data/dems'
        
        try:
            mgr = ConfigManager()
            config = mgr.load_config(sample_config_file)
            
            # 检查环境变量是否被替换
            assert config["paths"]["dem"] == "/data/dems/elevation.tif"
        finally:
            # 清理环境变量
            os.environ.pop('DEM_PATH', None)
    
    def test_env_var_not_set(self, sample_config_file):
        """测试环境变量未设置时保持原样"""
        # 确保环境变量未设置
        os.environ.pop('DEM_PATH', None)
        
        mgr = ConfigManager()
        config = mgr.load_config(sample_config_file)
        
        # 环境变量未设置，应该保持原样
        assert config["paths"]["dem"] == "${DEM_PATH}/elevation.tif"
    
    def test_disable_env_var_resolution(self, sample_config_file):
        """测试禁用环境变量替换"""
        os.environ['DEM_PATH'] = '/data/dems'
        
        try:
            mgr = ConfigManager()
            config = mgr.load_config(sample_config_file, resolve_env_vars=False)
            
            # 禁用替换，应该保持原样
            assert config["paths"]["dem"] == "${DEM_PATH}/elevation.tif"
        finally:
            os.environ.pop('DEM_PATH', None)


class TestPathResolution:
    """测试路径解析"""
    
    def test_relative_path_resolution(self, sample_config_file):
        """测试相对路径解析"""
        mgr = ConfigManager()
        config = mgr.load_config(sample_config_file)
        
        # 相对路径应该被解析为绝对路径
        base_path = sample_config_file.parent
        expected_precip = str((base_path / "data/precip.csv").resolve())
        
        assert Path(config["paths"]["precipitation"]).is_absolute()
        assert config["paths"]["precipitation"] == expected_precip
    
    def test_disable_path_resolution(self, sample_config_file):
        """测试禁用路径解析"""
        mgr = ConfigManager()
        config = mgr.load_config(sample_config_file, resolve_paths=False)
        
        # 禁用路径解析，应该保持原样
        assert config["directories"]["base_results"] == "results"
        assert config["paths"]["precipitation"] == "data/precip.csv"


class TestCacheManagement:
    """测试缓存管理"""
    
    def test_clear_specific_cache(self, sample_config_file, nested_config_file):
        """测试清除特定配置的缓存"""
        mgr = ConfigManager()
        
        # 加载两个配置
        mgr.load_config(sample_config_file)
        mgr.load_config(nested_config_file)
        
        # 清除第一个配置的缓存
        mgr.clear_cache(sample_config_file)
        
        # 第一个应该被清除，第二个应该还在
        assert mgr.get_cached_config(sample_config_file) is None
        assert mgr.get_cached_config(nested_config_file) is not None
    
    def test_clear_all_cache(self, sample_config_file, nested_config_file):
        """测试清除所有缓存"""
        mgr = ConfigManager()
        
        # 加载两个配置
        mgr.load_config(sample_config_file)
        mgr.load_config(nested_config_file)
        
        # 清除所有缓存
        mgr.clear_cache()
        
        # 两个都应该被清除
        assert mgr.get_cached_config(sample_config_file) is None
        assert mgr.get_cached_config(nested_config_file) is None
    
    def test_reload_config(self, sample_config_file):
        """测试重新加载配置"""
        mgr = ConfigManager()
        
        # 第一次加载
        config1 = mgr.load_config(sample_config_file)
        
        # 修改缓存中的值
        mgr.update_config_value(sample_config_file, "project.name", "modified")
        cached = mgr.get_cached_config(sample_config_file)
        assert cached["project"]["name"] == "modified"
        
        # 重新加载应该恢复原值
        config2 = mgr.reload_config(sample_config_file)
        assert config2["project"]["name"] == "test_project"


class TestConfigValidation:
    """测试配置验证"""
    
    def test_validate_required_keys(self):
        """测试必需键验证"""
        mgr = ConfigManager()
        
        config = {"key1": "value1", "key2": "value2"}
        
        # 所有必需键都存在
        is_valid, errors = mgr.validate_config(config, required_keys=["key1", "key2"])
        assert is_valid is True
        assert len(errors) == 0
        
        # 缺少必需键
        is_valid, errors = mgr.validate_config(config, required_keys=["key1", "key3"])
        assert is_valid is False
        assert len(errors) == 1
        assert "key3" in errors[0]
    
    def test_validate_schema_types(self):
        """测试schema类型验证"""
        mgr = ConfigManager()
        
        config = {
            "name": "test",
            "count": 10,
            "ratio": 0.5
        }
        
        schema = {
            "name": str,
            "count": int,
            "ratio": float
        }
        
        is_valid, errors = mgr.validate_config(config, schema=schema)
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_schema_type_mismatch(self):
        """测试schema类型不匹配"""
        mgr = ConfigManager()
        
        config = {
            "name": "test",
            "count": "not_a_number"  # 应该是int
        }
        
        schema = {
            "name": str,
            "count": int
        }
        
        is_valid, errors = mgr.validate_config(config, schema=schema)
        assert is_valid is False
        assert len(errors) == 1
        assert "count" in errors[0]
    
    def test_validate_schema_with_required(self):
        """测试schema中的required标记"""
        mgr = ConfigManager()
        
        config = {"name": "test"}
        
        schema = {
            "name": {"type": str, "required": True},
            "count": {"type": int, "required": True}
        }
        
        is_valid, errors = mgr.validate_config(config, schema=schema)
        assert is_valid is False
        assert len(errors) == 1
        assert "count" in errors[0]


class TestNestedKeyAccess:
    """测试嵌套键访问"""
    
    def test_get_nested_value(self, nested_config_file):
        """测试获取嵌套值"""
        mgr = ConfigManager()
        
        value = mgr.get_config_value(
            nested_config_file,
            "level1.level2.level3.value"
        )
        
        assert value == "deep_value"
    
    def test_get_simple_value(self, nested_config_file):
        """测试获取简单值"""
        mgr = ConfigManager()
        
        value = mgr.get_config_value(
            nested_config_file,
            "level1.simple"
        )
        
        assert value == "simple_value"
    
    def test_get_nonexistent_key(self, nested_config_file):
        """测试获取不存在的键"""
        mgr = ConfigManager()
        
        value = mgr.get_config_value(
            nested_config_file,
            "level1.nonexistent",
            default="default_value"
        )
        
        assert value == "default_value"
    
    def test_update_nested_value(self, nested_config_file):
        """测试更新嵌套值"""
        mgr = ConfigManager()
        
        # 加载配置
        mgr.load_config(nested_config_file)
        
        # 更新嵌套值
        mgr.update_config_value(
            nested_config_file,
            "level1.level2.level3.value",
            "new_value"
        )
        
        # 验证更新
        value = mgr.get_config_value(
            nested_config_file,
            "level1.level2.level3.value"
        )
        
        assert value == "new_value"
    
    def test_update_creates_path(self, nested_config_file):
        """测试更新时创建不存在的路径"""
        mgr = ConfigManager()
        
        # 加载配置
        mgr.load_config(nested_config_file)
        
        # 更新不存在的路径
        mgr.update_config_value(
            nested_config_file,
            "new.path.to.value",
            "created"
        )
        
        # 验证路径被创建
        value = mgr.get_config_value(
            nested_config_file,
            "new.path.to.value"
        )
        
        assert value == "created"
