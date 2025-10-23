# HydroSIS 编码系统说明

## 概述

HydroSIS 使用分层编码系统对流域、参数分区和子流域进行标识。本文档说明了编码的逻辑和规则。

## 参数分区编码（Zone ID）

### 编码原则

参数分区（Parameter Zone）使用顺序数字编号，从上游到下游递增：
- **Zone 1**: 最上游的分区（深度最大，无上游汇入）
- **Zone 2, 3, ...**:  中游分区
- **Zone n**: 最下游的分区（深度最小，流域出口）

### 编码计算

```python
# 1. 计算每个分区的深度（depth）
#    - 出口分区（无下游）: depth = 0
#    - 直接流向出口的分区: depth = 1
#    - 依此类推，上游分区深度递增

# 2. 按深度降序排列（深度大的排前面）
sorted_zones = sorted(zones, key=lambda z: depth[z], reverse=True)

# 3. 分配序号：深度最大（最上游）的为 Zone 1
for idx, zone in enumerate(sorted_zones, start=1):
    zone_id = idx  # 1, 2, 3, ...
```

### 示例

```
流域结构：
    Zone 1 (上游)
        ↓
    Zone 2 (中游)
        ↓
    Zone 3 (下游/出口)

深度计算：
    Zone 1: depth = 2
    Zone 2: depth = 1
    Zone 3: depth = 0

最终编号：
    Zone 1 (深度2) → zone_id = 1
    Zone 2 (深度1) → zone_id = 2
    Zone 3 (深度0) → zone_id = 3
```

## 子流域编码（Subzone ID）

### 编码原则

子流域（Subbasin/Subzone）编码采用**分层编码**方式：
```
subzone_id = zone_id × 100 + subzone_index
```

其中：
- `zone_id`: 所属参数分区的编号（1, 2, 3, ...）
- `subzone_index`: 该分区内的子流域序号（1, 2, 3, ...）

### 编码范围

- **Zone 1 的子流域**: 101, 102, 103, 104, ...
- **Zone 2 的子流域**: 201, 202, 203, 204, ...
- **Zone 3 的子流域**: 301, 302, 303, 304, ...
- **Zone n 的子流域**: n01, n02, n03, n04, ...

### 优势

1. **层次清晰**: 通过编号可直接识别子流域所属的参数分区
   - 编号 205 → Zone 2 的第 5 个子流域
   - 编号 103 → Zone 1 的第 3 个子流域

2. **容易扩展**: 每个分区最多支持 99 个子流域（足够大多数应用）

3. **便于管理**: 参数可以按分区批量配置

### 示例

```
流域划分：
Zone 1 (上游)
  ├─ Subzone 101
  ├─ Subzone 102
  └─ Subzone 103
      ↓
Zone 2 (中游)
  ├─ Subzone 201
  ├─ Subzone 202
  └─ Subzone 203
      ↓
Zone 3 (下游)
  ├─ Subzone 301
  └─ Subzone 302
```

## 河道编码（Channel/Segment ID）

### 编码原则

河道段（Channel Segment）的编号与其对应的子流域编号**相同**：
```
segment_id = subzone_id
```

### 理由

每个子流域有且仅有一条主河道段，因此：
- Subzone 101 的河道 → Segment 101
- Subzone 205 的河道 → Segment 205

### 河道属性

每个河道段包含以下属性：
- `segment_id`: 河道编号（= subzone_id）
- `zone_id`: 所属参数分区
- `subzone_id`: 所属子流域
- `downstream_id`: 下游河道编号
- `upstream_ids`: 上游河道编号列表
- `length_m`: 河道长度（米）
- `slope`: 河道坡度
- `drop_m`: 高程降落（米）

## 下游关系编码

### 子流域下游关系

- 同一分区内的子流域：`downstream_subzone_id` 指向同分区内的下游子流域
  - 例如：102 → 101

- 跨分区的下游关系：`downstream_subzone_id` 指向下一个分区的子流域
  - 例如：103 (Zone 1的出口) → 201 (Zone 2的入口)

### 分区下游关系

- 分区的 `downstream_id` 指向下游分区的 zone_id
  - 例如：Zone 1 → downstream_id = "2"
  - 例如：Zone 2 → downstream_id = "3"
  - 例如：Zone 3 → downstream_id = None (出口)

## 编码示例：完整流域

```
Upper Truckee River 流域示例：

参数分区：
  Zone 1 (最上游): zone_id = "1", downstream_id = "2"
  Zone 2 (中游):   zone_id = "2", downstream_id = "3"
  Zone 3 (下游):   zone_id = "3", downstream_id = None

子流域：
  Zone 1:
    101 → downstream: 102
    102 → downstream: 201 (跨分区)

  Zone 2:
    201 → downstream: 202
    202 → downstream: 203
    203 → downstream: 301 (跨分区)

  Zone 3:
    301 → downstream: 302
    302 → downstream: None (出口)

河道段：
  101, 102 (Zone 1)
  201, 202, 203 (Zone 2)
  301, 302 (Zone 3)
```

## 编码验证

### 有效的编码
- Zone ID: `"1"`, `"2"`, `"10"`, `"99"` (正整数字符串)
- Subzone ID: `"101"`, `"205"`, `"1503"` (zone_id × 100 + index)

### 无效的编码
- Zone ID: `"0"`, `"-1"`, `"A1"` (非正整数)
- Subzone ID: `"1"`, `"50"`, `"100"` (不符合编码规则)

## 代码参考

主要实现位置：
- 分区编码: `hydrosis/parameters/partition_builder.py:474-491`
- 子流域编码: `hydrosis/parameters/partition_builder.py:462-469`
- 深度计算: `hydrosis/parameters/partition_grid.py:89-106`

## 更新记录

- **2025-10-23**: 重构编码系统，确保分区和子流域编号一致性
  - 修复：将子流域生成移到分区重编号之后
  - 修复：确保 zone_id 从上游(1)到下游(n)递增
  - 优化：添加详细注释说明编码逻辑
