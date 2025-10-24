"""配置管理器

提供统一的配置加载、缓存、验证和环境变量支持。
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union
from threading import Lock

try:
    import yaml
except ImportError:
    yaml = None


class ConfigManager:
    """配置管理器单例类
    
    提供以下功能：
    - 配置文件缓存（避免重复IO操作）
    - 环境变量替换（支持 ${VAR_NAME} 语法）
    - 路径自动解析（相对路径转绝对路径）
    - 线程安全的单例模式
    
    使用示例
    --------
    >>> config_mgr = ConfigManager()
    >>> config = config_mgr.load_config("config/workflow_config.yaml")
    >>> base_dir = config["directories"]["base_results"]
    
    >>> # 支持环境变量
    >>> # 在YAML中: precipitation: ${DATA_DIR}/precip.csv
    >>> # 自动替换为环境变量的值
    
    >>> # 清除缓存并重新加载
    >>> config_mgr.clear_cache()
    >>> config = config_mgr.load_config("config/workflow_config.yaml")
    """
    
    _instance: Optional[ConfigManager] = None
    _lock: Lock = Lock()
    
    def __new__(cls) -> ConfigManager:
        """单例模式实现（线程安全）"""
        if cls._instance is None:
            with cls._lock:
                # 双重检查锁定
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化配置管理器"""
        # 避免重复初始化
        if self._initialized:
            return
        
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._config_base_paths: Dict[str, Path] = {}
        self._env_var_pattern = re.compile(r'\$\{([A-Z_][A-Z0-9_]*)\}')
        self._initialized = True
    
    def load_config(
        self,
        config_path: Union[str, Path],
        use_cache: bool = True,
        resolve_paths: bool = True,
        resolve_env_vars: bool = True
    ) -> Dict[str, Any]:
        """加载配置文件
        
        Parameters
        ----------
        config_path : str or Path
            配置文件路径
        use_cache : bool, default=True
            是否使用缓存。如果True且配置已缓存，直接返回缓存的配置
        resolve_paths : bool, default=True
            是否解析相对路径为绝对路径
        resolve_env_vars : bool, default=True
            是否替换环境变量
        
        Returns
        -------
        dict
            配置字典
        
        Raises
        ------
        ImportError
            如果PyYAML未安装
        FileNotFoundError
            如果配置文件不存在
        yaml.YAMLError
            如果YAML文件格式错误
        """
        if yaml is None:
            raise ImportError(
                "PyYAML is required to load configuration files. "
                "Install it with: pip install pyyaml"
            )
        
        config_path = Path(config_path).resolve()
        
        # 检查缓存
        cache_key = str(config_path)
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key].copy()
        
        # 检查文件存在
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # 加载YAML
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML file {config_path}: {e}")
        
        if config is None:
            config = {}
        
        # 记录配置文件的基础路径（用于解析相对路径）
        config_base_path = config_path.parent
        self._config_base_paths[cache_key] = config_base_path
        
        # 处理环境变量替换
        if resolve_env_vars:
            config = self._resolve_env_vars(config)
        
        # 处理路径解析
        if resolve_paths:
            config = self._resolve_paths(config, config_base_path)
        
        # 缓存配置
        if use_cache:
            self._cache[cache_key] = config.copy()
        
        return config
    
    def _resolve_env_vars(self, obj: Any) -> Any:
        """递归替换配置中的环境变量
        
        支持 ${VAR_NAME} 语法
        如果环境变量不存在，保持原样
        """
        if isinstance(obj, str):
            # 查找所有环境变量引用
            def replacer(match):
                var_name = match.group(1)
                return os.environ.get(var_name, match.group(0))
            
            return self._env_var_pattern.sub(replacer, obj)
        
        elif isinstance(obj, dict):
            return {k: self._resolve_env_vars(v) for k, v in obj.items()}
        
        elif isinstance(obj, list):
            return [self._resolve_env_vars(item) for item in obj]
        
        else:
            return obj
    
    def _resolve_paths(
        self,
        obj: Any,
        base_path: Path,
        path_keys: Optional[set] = None
    ) -> Any:
        """递归解析配置中的相对路径
        
        Parameters
        ----------
        obj : Any
            配置对象（可以是dict、list或其他类型）
        base_path : Path
            配置文件所在目录（相对路径的基础路径）
        path_keys : set, optional
            需要解析为路径的键名集合。如果为None，使用默认集合。
        """
        if path_keys is None:
            # 常见的路径相关键名
            path_keys = {
                'path', 'dir', 'directory', 'file', 'filepath', 'file_path',
                'output_dir', 'input_dir', 'results_directory',
                'precipitation', 'evaporation', 'discharge_observations',
                'figures_directory', 'reports_directory',
                'pour_points_path', 'dem_path', 'output_path'
            }
        
        if isinstance(obj, dict):
            resolved = {}
            for k, v in obj.items():
                # 检查是否需要解析路径
                key_lower = k.lower().replace('_', '')
                should_resolve = any(pk.lower().replace('_', '') in key_lower for pk in path_keys)
                
                if should_resolve and isinstance(v, str) and v:
                    # 解析为路径
                    try:
                        path = Path(v)
                        if not path.is_absolute():
                            path = (base_path / path).resolve()
                        resolved[k] = str(path)
                    except (ValueError, OSError):
                        # 如果不是有效路径，保持原样
                        resolved[k] = v
                else:
                    # 递归处理
                    resolved[k] = self._resolve_paths(v, base_path, path_keys)
            
            return resolved
        
        elif isinstance(obj, list):
            return [self._resolve_paths(item, base_path, path_keys) for item in obj]
        
        else:
            return obj
    
    def get_cached_config(self, config_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """获取缓存的配置（如果存在）
        
        Parameters
        ----------
        config_path : str or Path
            配置文件路径
        
        Returns
        -------
        dict or None
            缓存的配置字典，如果不存在则返回None
        """
        config_path = Path(config_path).resolve()
        cache_key = str(config_path)
        return self._cache.get(cache_key, None)
    
    def clear_cache(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """清除配置缓存
        
        Parameters
        ----------
        config_path : str or Path, optional
            要清除的配置文件路径。如果为None，清除所有缓存。
        """
        if config_path is None:
            # 清除所有缓存
            self._cache.clear()
            self._config_base_paths.clear()
        else:
            # 清除指定配置的缓存
            config_path = Path(config_path).resolve()
            cache_key = str(config_path)
            self._cache.pop(cache_key, None)
            self._config_base_paths.pop(cache_key, None)
    
    def reload_config(
        self,
        config_path: Union[str, Path],
        **kwargs
    ) -> Dict[str, Any]:
        """重新加载配置（清除缓存后重新加载）
        
        Parameters
        ----------
        config_path : str or Path
            配置文件路径
        **kwargs
            传递给 load_config() 的其他参数
        
        Returns
        -------
        dict
            重新加载的配置字典
        """
        self.clear_cache(config_path)
        return self.load_config(config_path, **kwargs)
    
    def validate_config(
        self,
        config: Dict[str, Any],
        required_keys: Optional[list] = None,
        schema: Optional[Dict[str, Any]] = None
    ) -> tuple[bool, list[str]]:
        """验证配置
        
        Parameters
        ----------
        config : dict
            要验证的配置字典
        required_keys : list, optional
            必需的键列表
        schema : dict, optional
            配置schema（简单的类型验证）
            格式: {"key": type} 或 {"key": {"type": type, "required": bool}}
        
        Returns
        -------
        is_valid : bool
            配置是否有效
        errors : list of str
            错误消息列表
        """
        errors = []
        
        # 检查必需键
        if required_keys:
            for key in required_keys:
                if key not in config:
                    errors.append(f"Missing required key: {key}")
        
        # 检查schema
        if schema:
            for key, spec in schema.items():
                # 解析spec
                if isinstance(spec, type):
                    expected_type = spec
                    required = False
                elif isinstance(spec, dict):
                    expected_type = spec.get("type", object)
                    required = spec.get("required", False)
                else:
                    continue
                
                # 检查是否存在
                if key not in config:
                    if required:
                        errors.append(f"Missing required key: {key}")
                    continue
                
                # 检查类型
                value = config[key]
                if not isinstance(value, expected_type):
                    errors.append(
                        f"Invalid type for key '{key}': "
                        f"expected {expected_type.__name__}, got {type(value).__name__}"
                    )
        
        return len(errors) == 0, errors
    
    def get_config_value(
        self,
        config_path: Union[str, Path],
        key_path: str,
        default: Any = None
    ) -> Any:
        """获取配置中的特定值（支持嵌套路径）
        
        Parameters
        ----------
        config_path : str or Path
            配置文件路径
        key_path : str
            键路径，使用点号分隔，例如 "directories.base_results"
        default : Any, optional
            如果键不存在，返回的默认值
        
        Returns
        -------
        Any
            配置值
        
        Examples
        --------
        >>> config_mgr = ConfigManager()
        >>> base_dir = config_mgr.get_config_value(
        ...     "config/workflow_config.yaml",
        ...     "directories.base_results"
        ... )
        """
        config = self.load_config(config_path)
        
        # 分割键路径
        keys = key_path.split('.')
        
        # 逐层访问
        current = config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def update_config_value(
        self,
        config_path: Union[str, Path],
        key_path: str,
        value: Any
    ) -> None:
        """更新缓存中的配置值（不修改文件）
        
        Parameters
        ----------
        config_path : str or Path
            配置文件路径
        key_path : str
            键路径，使用点号分隔
        value : Any
            新值
        
        Notes
        -----
        此方法只更新内存中的缓存，不会修改配置文件
        """
        config_path = Path(config_path).resolve()
        cache_key = str(config_path)
        
        # 确保配置已加载
        if cache_key not in self._cache:
            self.load_config(config_path)
        
        # 分割键路径
        keys = key_path.split('.')
        
        # 逐层访问并更新
        current = self._cache[cache_key]
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # 设置最终值
        current[keys[-1]] = value
    
    @classmethod
    def reset_instance(cls) -> None:
        """重置单例实例（主要用于测试）"""
        with cls._lock:
            cls._instance = None
