"""
HTTP工具类 - 通用的HTTP客户端

支持：
- 异步请求
- 自动重试
- 超时控制
- 错误处理
"""

import asyncio
import json
from typing import Dict, Any, Optional
from urllib import request, error
from urllib.parse import urljoin
import time


class MCPHttpClient:
    """通用MCP HTTP客户端"""
    
    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        初始化HTTP客户端
        
        Args:
            base_url: 基础URL（如 http://localhost:8080）
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    async def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        GET请求
        
        Args:
            endpoint: API端点（如 /tools）
            params: 查询参数
            
        Returns:
            响应JSON
        """
        url = urljoin(self.base_url, endpoint.lstrip('/'))
        
        if params:
            from urllib.parse import urlencode
            url = f"{url}?{urlencode(params)}"
        
        return await self._request('GET', url)
    
    async def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        POST请求
        
        Args:
            endpoint: API端点
            data: 表单数据
            json_data: JSON数据
            
        Returns:
            响应JSON
        """
        url = urljoin(self.base_url, endpoint.lstrip('/'))
        
        body = None
        headers = {}
        
        if json_data is not None:
            body = json.dumps(json_data).encode('utf-8')
            headers['Content-Type'] = 'application/json'
        elif data is not None:
            from urllib.parse import urlencode
            body = urlencode(data).encode('utf-8')
            headers['Content-Type'] = 'application/x-www-form-urlencoded'
        
        return await self._request('POST', url, body=body, headers=headers)
    
    async def _request(
        self,
        method: str,
        url: str,
        body: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        执行HTTP请求，支持重试
        
        Args:
            method: HTTP方法
            url: 完整URL
            body: 请求体
            headers: 请求头
            
        Returns:
            响应JSON
            
        Raises:
            ConnectionError: 连接失败
            TimeoutError: 请求超时
            ValueError: 响应格式错误
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # 使用asyncio.to_thread在线程中执行阻塞操作
                response_data = await asyncio.to_thread(
                    self._sync_request,
                    method,
                    url,
                    body,
                    headers or {}
                )
                return response_data
            
            except error.URLError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    raise ConnectionError(f"无法连接到 {url}: {e}") from e
            
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    raise
        
        # 不应该到达这里
        raise RuntimeError(f"请求失败: {last_error}")
    
    def _sync_request(
        self,
        method: str,
        url: str,
        body: Optional[bytes],
        headers: Dict[str, str]
    ) -> Dict[str, Any]:
        """同步请求（在线程中执行）"""
        req = request.Request(
            url,
            data=body,
            headers=headers,
            method=method
        )
        
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                response_body = response.read().decode('utf-8')
                
                # 解析JSON
                try:
                    return json.loads(response_body)
                except json.JSONDecodeError as e:
                    raise ValueError(f"响应不是有效的JSON: {response_body[:200]}") from e
        
        except error.HTTPError as e:
            # HTTP错误（4xx, 5xx）
            error_body = e.read().decode('utf-8')
            try:
                error_data = json.loads(error_body)
                raise ValueError(f"HTTP {e.code}: {error_data.get('detail', error_body)}")
            except json.JSONDecodeError:
                raise ValueError(f"HTTP {e.code}: {error_body}")
    
    async def health_check(self) -> bool:
        """
        健康检查
        
        Returns:
            是否健康
        """
        try:
            response = await self.get('/health')
            return response.get('status') == 'healthy'
        except Exception:
            return False


if __name__ == "__main__":
    # 测试代码
    async def test():
        client = MCPHttpClient("http://localhost:8080")
        
        print("测试健康检查...")
        is_healthy = await client.health_check()
        print(f"服务器健康: {is_healthy}")
        
        if is_healthy:
            print("\n测试GET请求...")
            tools = await client.get("/mcp/tools")
            print(f"获取到 {len(tools.get('tools', []))} 个工具")
    
    # asyncio.run(test())
    print("HTTP工具类已就绪")
