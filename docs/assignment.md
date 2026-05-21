# 作业说明

## 作业描述

本次作业需要基于已有的 2D 网格导航仿真器，完成 **代价地图 / 全局规划 / 局部控制** 三大模块的核心算法，使机器人能够在地图中自主导航到目标点。

## 如何启动

可以使用 uv 创建环境：

```bash
uv venv -p 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

启动：

```bash
# Level 1：纯静态地图
uv run python navigation/main.py --level 1 --record level1.mp4
# Level 2：加入随机移动的动态障碍物 + Lidar 感知
uv run python navigation/main.py --level 2 --record level2.mp4
```

- `--record` 参数会把仿真过程录成视频，方便调参和最终提交。

控制方式：

- 左键单击地图：临时设置目标点，切换为 `debug` 模式，不计时
- 右键单击地图：重置机器人和动态障碍物，回到 `timed` 模式，并重新计时
- ESC / 关闭窗口：退出

## 需要修改的目录与文件

需要修改的部分都以 `TODO` 注释标记，集中在 [`navigation/nav/`](../navigation/nav/) 目录下的四个函数。

- [`compute_costmap`](../navigation/nav/costmap.py)：全局代价地图
- [`update_local_costmap`](../navigation/nav/costmap.py)：局部代价地图
- [`global_plan`](../navigation/nav/planner.py)：全局规划
- [`local_plan`](../navigation/nav/controller.py)：局部控制

你可以以任意方式实现这四个部分，无论是调库还是自己写算法，只要满足接口要求即可。如果调库的话，需要自己配置环境并更新 `requirements.txt`。可以调整 `main.py` 里调用这四个函数的传入参数或逻辑，但没有必要、也尽量不要这样做。但你可能需要了解 `main.py` 的实现细节。不要修改其他文件，如果确认是其他文件的问题，请提 issue 说明。

下面的内容与文件中的英文注释基本相同，你可以直接参考文件中的注释来实现，也可以忽略这些注释，使用更好的方法。

### 1. `compute_costmap`

**做什么**：把 0/1 的二值静态地图变成一个 `uint8` 的代价地图。障碍物本身是一个很大的值，障碍物附近一定半径内的格子有一个随距离衰减的代价，更远的地方代价是 0。

**为什么需要**：机器人存在体积，只用 0/1 地图找出的路径会贴着墙走；带膨胀的代价地图会把路径推开一些，更安全。

**思路提示**：

- 可以计算每个自由单元格到最近障碍物的欧氏距离，（`scipy.ndimage.distance_transform_edt` 可以一次性完成此操作），然后将距离映射为代价。代价平滑地衰减至某个 `inflation_radius`。超出该半径，代价变为 0。
- 衰减的形状（线性、指数等）和膨胀半径的大小是可调参数。选择一个能够明显引导路径远离墙壁，但又不会使狭窄通道无法通行的值。膨胀半径太大也会导致机器人绕远路，浪费时间。

### 2. `update_local_costmap`

**做什么**：在 `compute_costmap` 的基础上，把 `Lidar` 当前帧扫到的命中点也加进代价地图，构成一张"静态膨胀 + 动态膨胀"的合成图。

**为什么需要**：静态地图只看得到墙；动态障碍物的位置每帧都在变，要靠 `Lidar` 实时探测。

**思路提示**：

- 将每次射线命中点 `(angle_i, distance[i])` 转换为世界坐标点 `(x + d*cos(a), y + d*sin(a))`，将该点所在单元标记为障碍物并进行代价计算。与静态代价地图取最大值合成最终的局部代价地图。
- 跳过没命中的射线（`distance[i] >= lidar_range`）。
- 可以跳过落在静态障碍物上的命中点。

### 3. `global_plan`

**做什么**：在代价地图上跑路径搜索，返回起点到终点的世界坐标路径点列表。

**思路提示：**

- 可以使用 8 连通性（上下左右四条线 + 4 条对角线）进行路径搜索。相邻单元格之间的步长成本应为 `dist + cell_cost`，其中 `dist` 对于直线移动为 $1$，对于对角线移动为 $\sqrt{2}$。
- 可以使用最短路径算法（如 Dijkstra）或者启发式路径搜索算法（如 A*）来搜索。如果使用启发式路径搜索算法，可以用八分位距离或欧几里得距离作为启发式函数。

### 4. `local_plan`

**做什么**：根据规划的路径生成一个本帧的速度指令 `(cmd_vx, cmd_vy)`。

**为什么需要**：规划的路径可能是弯曲的折线，机器人存在速度、加速度等物理限制，需要在局部进行速度和方向的调整，来跟随规划的路径。

**思路提示**：

- 简单的追踪：从最近的路径点向前扫描，直到累计距离超过前瞻半径（一个可调常数，例如 1.5-2.5 个网格单位）。速度方向为 `look_ahead - current_pose`，速度大小为 `max_speed`（如果剩余路径长度较短，则使用较小的速度值，以便更轻松地到达目标）。
- 可以使用更复杂的局部控制（如 Dynamic Window Approach），也可以尝试使用基于模型的控制方法（如 MPPI）。

## 可能需要知道的细节

### 整体代码架构

```
navigation/
├── main.py                # 仿真器入口，负责初始化和主循环调度
├── configs/
│   └── cfg.yaml           # 任务配置（障碍物参数、机器人参数、任务目标等）
├── core/
│   ├── recorder.py        # 视频记录与回放
│   ├── renderer.py        # 基于 Pygame 的视觉渲染引擎
│   ├── robot.py           # 机器人模型（记录位置和物理限制）
│   ├── sensor.py          # 传感器仿真（Lidar 射线扫描）
│   ├── task.py            # 任务逻辑（计时、更新、重置、状态管理）
│   └── world.py           # 物理环境（地图管理、碰撞检测、Robot 包装）
└── nav/
    ├── controller.py      # 局部控制
    ├── costmap.py         # 代价地图
    └── planner.py         # 全局规划
```

你的代码中只能拿到静态地图、`Lidar` 数据和机器人物理参数与当前位置，不能直接访问仿真器的内部状态（如动态障碍物的真实位置）。

最后提交作业时需要复原 [`configs/cfg.yaml`](../navigation/configs/cfg.yaml) 中的默认配置，以保证评测环境的一致性。你可以根据这个配置中的参数进行一些针对性的算法优化。

### 任务终止条件

[`core/task.py`](../navigation/core/task.py) 的 `Task` 类维护两个模式：

- **`timed` 模式**：起始点和目标点都在任务配置中，计时器从 0 开始递增。
- **`debug` 模式**：目标点是你左键点的位置，计时器冻结，不参与评测。

`timed` 模式下，**只有同时满足以下两个条件，计时器才会停止**：

1. 机器人到目标的距离小于阈值
2. 最近几帧的位置都相差很小（即机器人真的停下来了，而不是高速冲过终点）

这意味着如果你的 `local_plan` 在到达终点后还来回抖，或者机器人冲过终点继续往外走，**计时器都不会停**。完整的"到达"状态在侧栏会显示为 `reached!`，仅满足第一个条件时显示 `arriving...`。

### 运动学

运动学主要实现在 [`core/world.py`](../navigation/core/world.py) 的 `MapWorld` 类中，每一帧 `world.update(cmd_vx, cmd_vy, dt)` 会先更新动态障碍物，再更新机器人。

> 需要注意的是，机器人会和所有障碍物发生碰撞，但是动态障碍物只和静态地图碰撞，和机器人不会发生碰撞

#### 机器人

机器人是一个**全向圆盘**，由 [`core/robot.py`](../navigation/core/robot.py) 的 `Robot` 类描述，状态量是位置 `(x, y)` 和速度 `(vx, vy)`，物理参数有半径 `radius`、最大速度 `max_speed`、最大加速度 `max_accel`。

1. **速度饱和**：输入的速度指令 `(cmd_vx, cmd_vy)` 如果超过 `max_speed`，会被缩放到 `max_speed`

```python
spd = math.hypot(cmd_vx, cmd_vy)
if spd > max_speed:
    cmd_vx *= max_speed / spd
    cmd_vy *= max_speed / spd
```

2. **加速度限幅**：机器人不能瞬间从当前速度跳到目标速度，本帧速度增量会被裁剪到 `max_accel * dt`

```python
dvx = cmd_vx - vx
dvy = cmd_vy - vy
dv = math.hypot(dvx, dvy)
max_dv = max_accel * dt
if dv > max_dv and dv > 1e-9:
    dvx *= max_dv / dv
    dvy *= max_dv / dv
vx += dvx
vy += dvy
```

3. **位置更新 + 轴对齐滑动碰撞**

```python
nx = x + vx * dt
ny = y + vy * dt
```

如果 `(nx, ny)` 不和任何障碍物相撞，就接受这个新位置。否则尝试**轴对齐滑动**：

- 只挪 `x`：`(nx, y)` 可行，则 `x = nx`，同时把 `vy` 置零
- 只挪 `y`：`(x, ny)` 可行，则 `y = ny`，同时把 `vx` 置零
- 两个都不行：`vx = vy = 0`，机器人原地停下

#### 动态障碍物

动态障碍物被绑定在一个矩形区域内做匀速运动，位置更新和碰撞逻辑与机器人基本相同，碰到界或者静态墙时把该轴速度反向。

- 动态障碍物的数量、大小、矩形区域范围等参数在任务配置中
- 动态障碍物的初始速度方向是随机的，速度大小在任务配置的区间内均匀采样

## Checkpoints

如果你在完成作业过程中遇到困难，可以逐步完成以下 Checkpoints：

- [ ] 完成 `compute_costmap`，并在仿真器中显示预期的代价地图
- [ ] 完成 `global_plan`，并在仿真器中显示预期的规划的路径
- [ ] 完成 `local_plan`，并在仿真器中看到机器人能够沿着规划的路径移动，能够在 Level 1 中成功到达目标点
- [ ] 完成 `update_local_costmap`，并在仿真器中显示预期的局部代价地图，能够在 Level 2 中成功到达目标点
- [ ] 在 Level 1 中进行优化，使得机器人能够更好的跟随规划的路径
- [ ] 在 Level 2 中进行优化，使得机器人能够更好地避开动态障碍物