"""MCP服务器认证和权限管理"""

import os
import jwt
import hmac
import hashlib
import time
from enum import Enum
from functools import wraps
from typing import Dict, Optional
from fastapi import Security, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

logger = logging.getLogger(__name__)

# 安全配置
JWT_SECRET = os.environ.get('JWT_SECRET', 'your-secret-key-change-in-production')
JWT_ALGORITHM = 'HS256'
JWT_PUBLIC_KEY = os.environ.get('JWT_PUBLIC_KEY', '')

security = HTTPBearer()


class Permission(Enum):
    """权限定义"""
    # 项目权限
    PROJECT_CREATE = "project:create"
    PROJECT_READ = "project:read"
    PROJECT_UPDATE = "project:update"
    PROJECT_DELETE = "project:delete"
    
    # 数据权限
    DATA_UPLOAD = "data:upload"
    DATA_DOWNLOAD = "data:download"
    DATA_DELETE = "data:delete"
    
    # 计算权限
    SIMULATION_RUN = "simulation:run"
    CALIBRATION_RUN = "calibration:run"
    
    # 结果权限
    RESULT_VIEW = "result:view"
    RESULT_EXPORT = "result:export"
    
    # 管理权限
    USER_MANAGE = "user:manage"
    SYSTEM_CONFIG = "system:config"


class Role(Enum):
    """用户角色"""
    ADMIN = "admin"              # 管理员
    MODELER = "modeler"          # 建模师（高级用户）
    ANALYST = "analyst"          # 分析师（普通用户）
    VIEWER = "viewer"            # 查看者（只读）


# 角色-权限映射
ROLE_PERMISSIONS = {
    Role.ADMIN: [p for p in Permission],  # 所有权限
    Role.MODELER: [
        Permission.PROJECT_CREATE,
        Permission.PROJECT_READ,
        Permission.PROJECT_UPDATE,
        Permission.DATA_UPLOAD,
        Permission.DATA_DOWNLOAD,
        Permission.SIMULATION_RUN,
        Permission.CALIBRATION_RUN,
        Permission.RESULT_VIEW,
        Permission.RESULT_EXPORT,
    ],
    Role.ANALYST: [
        Permission.PROJECT_READ,
        Permission.DATA_UPLOAD,
        Permission.SIMULATION_RUN,
        Permission.RESULT_VIEW,
    ],
    Role.VIEWER: [
        Permission.PROJECT_READ,
        Permission.RESULT_VIEW,
    ],
}


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> Dict:
    """
    验证JWT Token
    
    Args:
        credentials: HTTP认证凭证
    
    Returns:
        Dict: 解析后的token载荷（包含user_id、role等）
    
    Raises:
        HTTPException: Token无效或过期
    """
    token = credentials.credentials
    
    try:
        # 解析token
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM]
        )
        
        # 验证必要字段
        if 'user_id' not in payload:
            raise HTTPException(status_code=401, detail='Token缺少user_id字段')
        
        # 验证过期时间
        if 'exp' in payload and payload['exp'] < time.time():
            raise HTTPException(status_code=401, detail='Token已过期')
        
        logger.info(f"用户认证成功: {payload.get('user_id')}")
        return payload
    
    except jwt.ExpiredSignatureError:
        logger.warning("Token已过期")
        raise HTTPException(status_code=401, detail='Token已过期')
    except jwt.InvalidTokenError as e:
        logger.warning(f"Token无效: {e}")
        raise HTTPException(status_code=401, detail='Token无效')
    except Exception as e:
        logger.error(f"Token验证失败: {e}")
        raise HTTPException(status_code=401, detail=f'Token验证失败: {str(e)}')


def verify_request_signature(request: Request) -> bool:
    """
    验证请求签名（防重放攻击）
    
    Args:
        request: FastAPI请求对象
    
    Returns:
        bool: 签名是否有效
    """
    signature = request.headers.get('X-Signature')
    timestamp = request.headers.get('X-Timestamp')
    
    if not signature or not timestamp:
        return False
    
    try:
        # 防重放攻击：检查时间戳（5分钟有效期）
        if abs(time.time() - float(timestamp)) > 300:
            logger.warning("请求时间戳过期")
            return False
        
        # 计算期望的签名
        secret_key = os.environ.get('REQUEST_SECRET_KEY', 'change-me')
        expected = hmac.new(
            secret_key.encode(),
            f'{timestamp}'.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # 比较签名
        return hmac.compare_digest(signature, expected)
    
    except Exception as e:
        logger.error(f"签名验证失败: {e}")
        return False


def require_permission(permission: Permission):
    """
    权限检查装饰器
    
    Args:
        permission: 所需权限
    
    Returns:
        装饰器函数
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 从kwargs中获取认证信息（由verify_token提供）
            auth_payload = kwargs.get('auth_payload')
            
            if not auth_payload:
                raise HTTPException(status_code=401, detail='未提供认证信息')
            
            # 获取用户角色
            role_str = auth_payload.get('role', 'viewer')
            try:
                role = Role(role_str)
            except ValueError:
                role = Role.VIEWER
            
            # 检查权限
            allowed_permissions = ROLE_PERMISSIONS.get(role, [])
            if permission not in allowed_permissions:
                logger.warning(
                    f"用户 {auth_payload.get('user_id')} "
                    f"尝试访问未授权的功能: {permission.value}"
                )
                raise HTTPException(
                    status_code=403,
                    detail=f'权限不足: 需要 {permission.value}'
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def generate_token(
    user_id: str,
    role: Role = Role.ANALYST,
    expires_in: int = 7 * 24 * 3600  # 默认7天
) -> str:
    """
    生成JWT Token
    
    Args:
        user_id: 用户ID
        role: 用户角色
        expires_in: 过期时间（秒）
    
    Returns:
        str: JWT Token
    """
    payload = {
        'user_id': user_id,
        'role': role.value,
        'iat': time.time(),
        'exp': time.time() + expires_in,
        'issuer': 'hydrosis-mcp-server'
    }
    
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token


def check_ip_whitelist(request: Request) -> bool:
    """
    检查IP白名单
    
    Args:
        request: FastAPI请求对象
    
    Returns:
        bool: IP是否在白名单中
    """
    client_ip = request.client.host
    
    # 从环境变量读取白名单
    whitelist_str = os.environ.get('IP_WHITELIST', '')
    if not whitelist_str:
        # 如果没有配置白名单，默认允许所有IP
        return True
    
    whitelist = [ip.strip() for ip in whitelist_str.split(',')]
    
    # 检查IP是否在白名单中
    if client_ip in whitelist:
        return True
    
    # 支持CIDR表示法的简单检查（可以使用ipaddress模块实现更复杂的逻辑）
    for allowed_ip in whitelist:
        if allowed_ip.endswith('/24'):
            # 简单的C类网段检查
            network_prefix = allowed_ip.replace('/24', '')
            if client_ip.startswith(network_prefix.rsplit('.', 1)[0]):
                return True
    
    logger.warning(f"IP不在白名单中: {client_ip}")
    return False


class AuthMiddleware:
    """认证中间件"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request = Request(scope, receive)
            
            # 跳过健康检查端点
            if request.url.path in ["/health", "/ready", "/"]:
                return await self.app(scope, receive, send)
            
            # IP白名单检查
            if not check_ip_whitelist(request):
                from fastapi.responses import JSONResponse
                response = JSONResponse(
                    status_code=403,
                    content={"error": "IP地址不在白名单中"}
                )
                return await response(scope, receive, send)
        
        return await self.app(scope, receive, send)
