-- HydroSIS MCP服务器数据库初始化脚本

-- 创建数据库（如果不存在）
CREATE DATABASE IF NOT EXISTS hydrosis;

\c hydrosis;

-- 用户表
CREATE TABLE IF NOT EXISTS users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    display_name VARCHAR(100),
    avatar_url TEXT,
    role VARCHAR(20) DEFAULT 'analyst',
    storage_quota BIGINT DEFAULT 10737418240, -- 10GB
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- 项目表
CREATE TABLE IF NOT EXISTS projects (
    project_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    model_config JSONB NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_deleted BOOLEAN DEFAULT FALSE
);

-- 任务表
CREATE TABLE IF NOT EXISTS tasks (
    task_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
    project_id UUID REFERENCES projects(project_id) ON DELETE CASCADE,
    tool_name VARCHAR(100) NOT NULL,
    arguments JSONB NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    progress FLOAT DEFAULT 0,
    result JSONB,
    error_message TEXT,
    callback_url TEXT,
    metadata JSONB,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 运行记录表
CREATE TABLE IF NOT EXISTS simulation_runs (
    run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(project_id) ON DELETE CASCADE,
    task_id UUID REFERENCES tasks(task_id) ON DELETE SET NULL,
    scenario_ids TEXT[],
    start_date DATE,
    end_date DATE,
    status VARCHAR(20) NOT NULL,
    metrics JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- 校准记录表
CREATE TABLE IF NOT EXISTS calibration_runs (
    calibration_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(project_id) ON DELETE CASCADE,
    task_id UUID REFERENCES tasks(task_id) ON DELETE SET NULL,
    optimization_metric VARCHAR(20),
    best_parameters JSONB,
    best_score FLOAT,
    iterations INTEGER,
    convergence_history JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- 会话表
CREATE TABLE IF NOT EXISTS sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
    token_hash VARCHAR(255) NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- API调用日志表
CREATE TABLE IF NOT EXISTS api_logs (
    log_id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(user_id) ON DELETE SET NULL,
    endpoint VARCHAR(200),
    method VARCHAR(10),
    status_code INTEGER,
    response_time_ms FLOAT,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_projects_user ON projects(user_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_projects_created ON projects(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tasks_user_status ON tasks(user_id, status);
CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tasks_project ON tasks(project_id);
CREATE INDEX IF NOT EXISTS idx_simulation_runs_project ON simulation_runs(project_id);
CREATE INDEX IF NOT EXISTS idx_calibration_runs_project ON calibration_runs(project_id);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_api_logs_user ON api_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_api_logs_created ON api_logs(created_at DESC);

-- 创建更新时间戳触发器
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_projects_updated_at BEFORE UPDATE ON projects
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- 插入示例用户（仅用于开发测试）
INSERT INTO users (username, email, password_hash, display_name, role)
VALUES 
    ('demo_user', 'demo@hydrosis.com', 'hashed_password', '演示用户', 'modeler'),
    ('admin', 'admin@hydrosis.com', 'hashed_password', '管理员', 'admin')
ON CONFLICT (username) DO NOTHING;

-- 授权
GRANT ALL PRIVILEGES ON DATABASE hydrosis TO hydrosis;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO hydrosis;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO hydrosis;

-- 完成
SELECT 'Database initialized successfully!' AS status;
