import taichi as ti

# 初始化 Taichi，使用 GPU 加速运算
ti.init(arch=ti.gpu)

# 物理与网格参数
N = 20             # 布料网格分辨率 N x N
mass = 1.0         # 质点质量
dt = 5e-4          # 时间步长 (半隐式稳定上限约 0.01，当前远低于此)
k_s = 10000.0      # 弹簧劲度系数
k_d = 1.0          # 阻尼系数
gravity = ti.Vector([0.0, -9.8, 0.0])
max_velocity = 50.0  # 速度上限，防止数值爆炸
# 弹簧最大数量：结构 + 剪切 + 弯曲
max_springs = N * N * 8

# 定义 Taichi 数据场
x = ti.Vector.field(3, dtype=float, shape=N * N)       # 位置
v = ti.Vector.field(3, dtype=float, shape=N * N)       # 速度
f = ti.Vector.field(3, dtype=float, shape=N * N)       # 受力
is_fixed = ti.field(dtype=int, shape=N * N)            # 是否为固定点

# 隐式欧拉专用的预测缓存场
x_next = ti.Vector.field(3, dtype=float, shape=N * N)
v_next = ti.Vector.field(3, dtype=float, shape=N * N)
f_next = ti.Vector.field(3, dtype=float, shape=N * N)  # 隐式欧拉专用力场

# 弹簧数据场
spring_indices = ti.field(dtype=int, shape=max_springs * 2) # 用于渲染画线
spring_pairs = ti.Vector.field(2, dtype=int, shape=max_springs)
spring_lengths = ti.field(dtype=float, shape=max_springs)
num_springs = ti.field(dtype=int, shape=())

# 球体碰撞参数
sphere_center = ti.Vector.field(3, dtype=float, shape=1)
sphere_radius = ti.field(dtype=float, shape=())

# ============ 初始化 (拆分为多个 kernel 保证 GPU 同步) ============

@ti.kernel
def init_positions():
    """初始化质点位置与固定状态"""
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        # 将布料放置在三维空间的 XY 平面上
        x[idx] = ti.Vector([i * 0.05 - 0.5, 0.8, j * 0.05 - 0.5])
        v[idx] = ti.Vector([0.0, 0.0, 0.0])
        f[idx] = ti.Vector([0.0, 0.0, 0.0])
        # 固定第一排的两个角点
        if j == 0 and (i == 0 or i == N - 1):
            is_fixed[idx] = 1
        else:
            is_fixed[idx] = 0

@ti.kernel
def init_springs():
    """初始化弹簧：结构弹簧 + 剪切弹簧 + 弯曲弹簧"""
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        # ----- 结构弹簧 -----
        # 右侧相邻点
        if i < N - 1:
            idx_right = (i + 1) * N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_right])
            spring_lengths[c] = (x[idx] - x[idx_right]).norm()
        # 下方相邻点
        if j < N - 1:
            idx_down = i * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_down])
            spring_lengths[c] = (x[idx] - x[idx_down]).norm()

        # ----- 剪切弹簧（对角线） -----
        # 右下对角线
        if i < N - 1 and j < N - 1:
            idx_diag = (i + 1) * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_diag])
            spring_lengths[c] = (x[idx] - x[idx_diag]).norm()
        # 左下对角线（对应右上方向）
        if i > 0 and j < N - 1:
            idx_anti = (i - 1) * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_anti])
            spring_lengths[c] = (x[idx] - x[idx_anti]).norm()

        # ----- 弯曲弹簧（间隔一个质点的邻接点） -----
        # 右侧隔点
        if i < N - 2:
            idx_bend_right = (i + 2) * N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_bend_right])
            spring_lengths[c] = (x[idx] - x[idx_bend_right]).norm()
        # 下方隔点
        if j < N - 2:
            idx_bend_down = i * N + (j + 2)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_bend_down])
            spring_lengths[c] = (x[idx] - x[idx_bend_down]).norm()

@ti.kernel
def init_spring_indices():
    """同步渲染索引"""
    for i in range(num_springs[None]):
        spring_indices[i * 2] = spring_pairs[i][0]
        spring_indices[i * 2 + 1] = spring_pairs[i][1]

def init_cloth():
    """从 Python 层按顺序调用各初始化 kernel，确保 GPU 同步，并放置球体"""
    num_springs[None] = 0  # 重置弹簧计数
    init_positions()
    init_springs()
    init_spring_indices()
    # 放置一个球体在布料下方中央偏左的位置
    sphere_center[0] = ti.Vector([0.0, 0.3, 0.0])
    sphere_radius[None] = 0.15

# ============ 合并的力计算函数 (ti.func 内联到 kernel 中) ============

@ti.func
def compute_forces_on(pos: ti.template(), vel: ti.template(), force: ti.template()):
    """计算所有力 (重力 + 阻尼 + 弹簧力)"""
    # 第一阶段：清空受力，施加重力与阻尼
    for i in range(N * N):
        force[i] = gravity * mass - k_d * vel[i]
    # 第二阶段：累加弹簧力 (使用 atomic_add 保证多线程安全)
    for i in range(num_springs[None]):
        idx_a = spring_pairs[i][0]
        idx_b = spring_pairs[i][1]
        pos_a = pos[idx_a]
        pos_b = pos[idx_b]
        d = pos_a - pos_b
        dist = d.norm()
        if dist > 1e-6:
            d_normalized = d / dist
            f_spring = -k_s * (dist - spring_lengths[i]) * d_normalized
            ti.atomic_add(force[idx_a], f_spring)
            ti.atomic_add(force[idx_b], -f_spring)

@ti.func
def clamp_velocity(vel: ti.template(), idx: int):
    """速度钳制，防止数值爆炸"""
    vel_norm = vel[idx].norm()
    if vel_norm > max_velocity:
        vel[idx] = vel[idx] / vel_norm * max_velocity

# ============ 球体碰撞解决 ============
@ti.func
def resolve_sphere_collision(pos: ti.template(), vel: ti.template(), idx: int):
    """处理质点与球体的碰撞：穿透时推出，并调整速度法向分量"""
    center = sphere_center[0]
    r = sphere_radius[None]
    dir_vec = pos[idx] - center
    dist = dir_vec.norm()
    if dist < r and dist > 1e-6:
        # 推出到球体表面
        pos[idx] = center + dir_vec / dist * r
        # 如果速度指向球心，则反弹（去除法向分量）
        v_n = vel[idx].dot(dir_vec / dist)
        if v_n < 0:
            vel[idx] -= v_n * (dir_vec / dist)

# ============ 合并的积分 kernel (每步仅 1 次 kernel 启动) ============

@ti.kernel
def step_explicit():
    """显式欧拉 (Explicit Euler)"""
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            x[i] += v[i] * dt
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
            resolve_sphere_collision(x, v, i)

@ti.kernel
def step_semi_implicit():
    """半隐式欧拉 (Semi-Implicit Euler)"""
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
            x[i] += v[i] * dt
            resolve_sphere_collision(x, v, i)

@ti.kernel
def step_implicit_iter():
    """隐式欧拉 (Implicit Euler) - 使用定点迭代法，内部也进行碰撞处理"""
    # 1. 复制当前状态到预测场
    for i in range(N * N):
        v_next[i] = v[i]
        x_next[i] = x[i]
    # 2. 定点迭代求解未来状态 (ti.static 在编译期展开)
    for _ in ti.static(range(3)):
        compute_forces_on(x_next, v_next, f_next)
        for i in range(N * N):
            if is_fixed[i] == 0:
                v_next[i] = v[i] + (f_next[i] / mass) * dt
                clamp_velocity(v_next, i)
                x_next[i] = x[i] + v_next[i] * dt
                # 每次迭代都解决碰撞，帮助收敛到合理的无穿透状态
                resolve_sphere_collision(x_next, v_next, i)
    # 3. 将收敛后的状态写回
    for i in range(N * N):
        v[i] = v_next[i]
        x[i] = x_next[i]

# ============ 主函数 ============

def main():
    init_cloth()

    # 建立 GGUI 窗口
    window = ti.ui.Window("Games101 - Mass Spring System (Enhanced)", (800, 800))
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.0, 0.5, 2.0)
    camera.lookat(0.0, 0.0, 0.0)

    current_method = 1  # 0: 显式, 1: 半隐式, 2: 隐式
    paused = False

    while window.running:
        # =========== GUI 控制面板 ===========
        window.GUI.begin("Control Panel", 0.02, 0.02, 0.38, 0.36)

        window.GUI.text("Integration Method:")

        prefix_0 = "[*] " if current_method == 0 else "[ ] "
        prefix_1 = "[*] " if current_method == 1 else "[ ] "
        prefix_2 = "[*] " if current_method == 2 else "[ ] "

        if window.GUI.button(prefix_0 + "Explicit Euler (Explosive)"):
            current_method = 0
            init_cloth()
        if window.GUI.button(prefix_1 + "Semi-Implicit Euler (Stable)"):
            current_method = 1
            init_cloth()
        if window.GUI.button(prefix_2 + "Implicit Euler (Damped)"):
            current_method = 2
            init_cloth()

        window.GUI.text("")

        pause_label = "Resume Simulation" if paused else "Pause Simulation"
        if window.GUI.button(pause_label):
            paused = not paused

        if window.GUI.button("Reset Cloth"):
            init_cloth()

        window.GUI.end()
        # ====================================

        if not paused:
            for _ in range(40):
                if current_method == 0:
                    step_explicit()
                elif current_method == 1:
                    step_semi_implicit()
                elif current_method == 2:
                    step_implicit_iter()

        # 渲染场景
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        # 布料质点与弹簧线框
        scene.particles(x, radius=0.015, color=(0.2, 0.6, 1.0))
        scene.lines(x, indices=spring_indices, width=1.5, color=(0.8, 0.8, 0.8))
        # 绘制球体（一个大粒子）
        scene.particles(sphere_center, radius=sphere_radius[None], color=(1.0, 0.5, 0.0))

        canvas.scene(scene)
        window.show()

if __name__ == '__main__':
    main()