"""工作流配置管理"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:
    yaml = None


class WorkflowConfig:
    """工作流配置管理器"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """初始化配置管理器
        
        Args:
            config_dir: 配置文件目录，默认为 config/workflows/
        """
        self.config_dir = config_dir or Path("config/workflows")
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def load(self, workflow_id: str) -> Dict[str, Any]:
        """加载工作流配置
        
        Args:
            workflow_id: 工作流ID
        
        Returns:
            配置字典
        """
        config_file = self.config_dir / f"{workflow_id}.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(f"工作流配置不存在: {config_file}")
        
        if yaml is None:
            raise RuntimeError("需要安装PyYAML来加载配置")
        
        return yaml.safe_load(config_file.read_text(encoding='utf-8'))
    
    def save(self, workflow_id: str, config: Dict[str, Any]) -> None:
        """保存工作流配置
        
        Args:
            workflow_id: 工作流ID
            config: 配置字典
        """
        if yaml is None:
            raise RuntimeError("需要安装PyYAML来保存配置")
        
        config_file = self.config_dir / f"{workflow_id}.yaml"
        config_file.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding='utf-8'
        )
    
    def list_workflows(self) -> list[str]:
        """列出所有可用的工作流"""
        return [
            f.stem for f in self.config_dir.glob("*.yaml")
        ]
