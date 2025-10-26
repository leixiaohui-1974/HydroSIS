"""
MCP客户端使用示例

演示如何使用Python调用HydroSIS MCP服务器
"""

import requests
import time
import json
from typing import Dict, Any, List


class HydroSISMCPClient:
    """HydroSIS MCP客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8080", token: str = None):
        """
        初始化客户端
        
        Args:
            base_url: MCP服务器地址
            token: JWT认证令牌（可选）
        """
        self.base_url = base_url.rstrip('/')
        self.token = token
        self.session = requests.Session()
        
        if token:
            self.session.headers.update({
                "Authorization": f"Bearer {token}"
            })
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有可用工具"""
        response = self.session.get(f"{self.base_url}/mcp/tools")
        response.raise_for_status()
        return response.json()['tools']
    
    def get_tool_schema(self, tool_name: str) -> Dict[str, Any]:
        """获取工具的JSON Schema"""
        response = self.session.get(f"{self.base_url}/mcp/tools/{tool_name}/schema")
        response.raise_for_status()
        return response.json()
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        同步调用工具
        
        Args:
            tool_name: 工具名称
            arguments: 工具参数
        
        Returns:
            工具执行结果
        """
        response = self.session.post(
            f"{self.base_url}/mcp/tools/{tool_name}",
            json=arguments
        )
        response.raise_for_status()
        result = response.json()
        
        # 解析结果
        if result.get('isError'):
            error_msg = result['content'][0]['text']
            raise Exception(f"工具执行失败: {error_msg}")
        
        # 返回文本内容
        text_content = result['content'][0]['text']
        return json.loads(text_content)
    
    def submit_task(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        user_id: str,
        callback_url: str = None
    ) -> str:
        """
        提交异步任务
        
        Args:
            tool_name: 工具名称
            arguments: 工具参数
            user_id: 用户ID
            callback_url: 回调URL（可选）
        
        Returns:
            任务ID
        """
        response = self.session.post(
            f"{self.base_url}/tasks/submit",
            json={
                "tool_name": tool_name,
                "arguments": arguments,
                "user_id": user_id,
                "callback_url": callback_url
            }
        )
        response.raise_for_status()
        return response.json()['task_id']
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """获取任务状态"""
        response = self.session.get(f"{self.base_url}/tasks/{task_id}")
        response.raise_for_status()
        return response.json()
    
    def wait_for_task(
        self,
        task_id: str,
        timeout: int = 3600,
        poll_interval: int = 2
    ) -> Dict[str, Any]:
        """
        等待任务完成
        
        Args:
            task_id: 任务ID
            timeout: 超时时间（秒）
            poll_interval: 轮询间隔（秒）
        
        Returns:
            任务最终状态
        """
        start_time = time.time()
        
        while True:
            status = self.get_task_status(task_id)
            
            # 任务已完成
            if status['status'] in ['completed', 'failed', 'cancelled']:
                return status
            
            # 超时检查
            if time.time() - start_time > timeout:
                raise TimeoutError(f"任务超时: {task_id}")
            
            # 打印进度
            progress = status.get('progress', 0)
            print(f"任务进度: {progress:.1f}%", end='\r')
            
            time.sleep(poll_interval)
    
    def list_user_tasks(self, user_id: str, status: str = None) -> List[Dict[str, Any]]:
        """列出用户的任务"""
        url = f"{self.base_url}/tasks/user/{user_id}"
        if status:
            url += f"?status={status}"
        
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()['tasks']


def example_1_create_project():
    """示例1: 创建项目"""
    print("\n" + "="*60)
    print("示例1: 创建水文模拟项目")
    print("="*60)
    
    client = HydroSISMCPClient()
    
    # 健康检查
    health = client.health_check()
    print(f"服务器状态: {health['status']}")
    print(f"可用工具数: {health['tools_count']}")
    
    # 创建项目
    result = client.call_tool("create_project", {
        "user_id": "demo_user",
        "project_name": "长江上游流域模拟",
        "description": "基于HBV模型的日尺度径流模拟",
        "template": "advanced"
    })
    
    print(f"\n项目创建成功!")
    print(f"项目ID: {result['project_id']}")
    print(f"项目路径: {result['path']}")
    
    return result['project_id']


def example_2_list_tools():
    """示例2: 列出所有工具"""
    print("\n" + "="*60)
    print("示例2: 列出所有可用工具")
    print("="*60)
    
    client = HydroSISMCPClient()
    
    tools = client.list_tools()
    
    # 按分类组织工具
    categories = {}
    for tool in tools:
        category = tool.get('category', 'general')
        if category not in categories:
            categories[category] = []
        categories[category].append(tool['name'])
    
    print(f"\n共有 {len(tools)} 个工具:\n")
    for category, tool_names in categories.items():
        print(f"📁 {category}:")
        for name in tool_names:
            print(f"   - {name}")
        print()


def example_3_configure_model():
    """示例3: 配置模型"""
    print("\n" + "="*60)
    print("示例3: 配置产流模型")
    print("="*60)
    
    client = HydroSISMCPClient()
    
    # 假设已有项目ID
    project_id = "demo-project-id"
    
    # 配置HBV产流模型
    result = client.call_tool("configure_runoff_model", {
        "project_id": project_id,
        "model_type": "hbv",
        "parameters": {
            "fc": 200.0,      # 田间持水量
            "beta": 2.0,      # 土壤非线性系数
            "lp": 0.7,        # 蒸散发阈值
            "pwp": 100.0      # 凋萎点
        }
    })
    
    print(f"\n模型配置成功!")
    print(f"模型类型: {result['model_type']}")
    print(f"参数: {json.dumps(result['parameters'], indent=2, ensure_ascii=False)}")


def example_4_async_simulation():
    """示例4: 异步模拟任务"""
    print("\n" + "="*60)
    print("示例4: 提交异步模拟任务")
    print("="*60)
    
    client = HydroSISMCPClient()
    
    project_id = "demo-project-id"
    
    # 提交模拟任务
    print("提交模拟任务...")
    task_id = client.submit_task(
        tool_name="run_simulation",
        arguments={
            "project_id": project_id,
            "start_date": "2020-01-01",
            "end_date": "2020-12-31",
            "generate_report": True
        },
        user_id="demo_user"
    )
    
    print(f"任务ID: {task_id}")
    print("\n等待任务完成...")
    
    # 等待任务完成
    try:
        final_status = client.wait_for_task(task_id, timeout=600)
        
        if final_status['status'] == 'completed':
            print("\n✅ 模拟完成!")
            result = final_status.get('result', {})
            print(f"运行ID: {result.get('run_id')}")
            print(f"执行情景数: {result.get('scenarios_executed')}")
        else:
            print(f"\n❌ 任务失败: {final_status.get('error')}")
    
    except TimeoutError:
        print("\n⏱️ 任务超时")


def example_5_calibration():
    """示例5: 参数校准"""
    print("\n" + "="*60)
    print("示例5: 模型参数校准")
    print("="*60)
    
    client = HydroSISMCPClient()
    
    project_id = "demo-project-id"
    
    # 准备观测数据（示例）
    observed_data = {
        "outlet": [10.5, 12.3, 15.6, 18.2, 20.1]  # 流量观测值
    }
    
    # 提交校准任务
    print("提交校准任务...")
    task_id = client.submit_task(
        tool_name="calibrate_parameters",
        arguments={
            "project_id": project_id,
            "observed_data": observed_data,
            "optimization_metric": "nse",
            "max_iterations": 100
        },
        user_id="demo_user"
    )
    
    print(f"任务ID: {task_id}")
    
    # 轮询状态（简化版）
    for i in range(5):
        status = client.get_task_status(task_id)
        print(f"状态: {status['status']}, 进度: {status['progress']:.1f}%")
        time.sleep(1)


def example_6_workflow():
    """示例6: 完整工作流"""
    print("\n" + "="*60)
    print("示例6: 完整水文建模工作流")
    print("="*60)
    
    client = HydroSISMCPClient()
    user_id = "demo_user"
    
    # 步骤1: 创建项目
    print("\n📝 步骤1: 创建项目")
    project = client.call_tool("create_project", {
        "user_id": user_id,
        "project_name": "完整工作流示例",
        "template": "advanced"
    })
    project_id = project['project_id']
    print(f"✅ 项目ID: {project_id}")
    
    # 步骤2: 流域划分
    print("\n🗺️ 步骤2: 流域划分")
    delineation = client.call_tool("delineate_watershed", {
        "project_id": project_id,
        "dem_path": "/data/dem/example.tif",
        "pour_points": [
            {"id": "outlet", "lon": 120.5, "lat": 30.5}
        ]
    })
    print(f"✅ 子流域数: {delineation['subbasin_count']}")
    
    # 步骤3: 配置模型
    print("\n⚙️ 步骤3: 配置模型")
    config = client.call_tool("configure_runoff_model", {
        "project_id": project_id,
        "model_type": "scs",
        "parameters": {"cn": 75}
    })
    print(f"✅ 模型类型: {config['model_type']}")
    
    # 步骤4: 上传数据
    print("\n📤 步骤4: 上传驱动数据")
    upload = client.call_tool("upload_forcing_data", {
        "project_id": project_id,
        "data_type": "precipitation",
        "data": {
            "zone1": [10.5, 12.3, 8.9, 15.2, 20.1]
        },
        "timestamps": [
            "2020-01-01", "2020-01-02", "2020-01-03",
            "2020-01-04", "2020-01-05"
        ]
    })
    print(f"✅ 数据上传完成")
    
    # 步骤5: 运行模拟（异步）
    print("\n🚀 步骤5: 运行模拟")
    task_id = client.submit_task(
        tool_name="run_simulation",
        arguments={
            "project_id": project_id,
            "generate_report": True
        },
        user_id=user_id
    )
    print(f"✅ 任务已提交: {task_id}")
    
    print("\n✨ 工作流创建完成!")


def main():
    """主函数"""
    print("="*60)
    print("HydroSIS MCP客户端使用示例")
    print("="*60)
    
    # 运行示例
    try:
        example_1_create_project()
        example_2_list_tools()
        # example_3_configure_model()
        # example_4_async_simulation()
        # example_5_calibration()
        # example_6_workflow()
        
    except requests.exceptions.ConnectionError:
        print("\n❌ 错误: 无法连接到MCP服务器")
        print("请确保服务器正在运行: python -m mcp_server.main")
    except Exception as e:
        print(f"\n❌ 错误: {e}")


if __name__ == "__main__":
    main()
