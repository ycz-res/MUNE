"""
配置管理模块
直接读取config.yaml文件，提供简洁的访问方法
"""

import yaml
import os
from typing import Dict, Any, Optional


class Config:
    """配置类，支持点号访问和get方法"""
    
    def __init__(self, config_file: str = 'config.yaml'):
        self.config_file = config_file
        self._data = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"配置文件不存在: {self.config_file}")
        
        with open(self.config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def __getitem__(self, key: str) -> Any:
        """支持点号访问，如 config['model.name']"""
        if '.' in key:
            keys = key.split('.')
            value = self._data
            for k in keys:
                value = value[k]
            return value
        else:
            return self._data[key]
    
        """支持点号设置，如 config['model.name'] = 'new_model'"""
        if '.' in key:
            keys = key.split('.')
            current = self._data
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
        else:
            self._data[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持点号访问"""
        try:
            return self[key]
        except KeyError:
            return default
    
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为普通字典"""
        return self._data.copy()


# 全局配置实例
config = Config()


def get_config() -> Config:
    """获取全局配置实例"""
    return config
