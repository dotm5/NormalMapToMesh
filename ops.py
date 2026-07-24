# -*- coding: utf-8 -*-
# NormalMapToMesh 操作层 —— bpy 侧: 直算前端(免烘焙)、Multires 管理、
# SIMPLE 细分求值、numpy 位移、multires_reshape 写回。
#
# v6 不变式: 一切场量只由低模 + 法线贴图决定, 细分只是采样密度——任何级别
# 都输出同一张曲面(基面 = 低模的确定函数), 位移管线不读细分网格自身属性。
#   1. 直算前端: mikktspace 切线帧(calc_tangents, 与渲染器同源)+逐三角形解析
#      ∂P/∂u,∂P/∂v 光栅化到 UV 网格; 材质法线链用 numpy 节点求值器直接算,
#      切线空间法线图 + 切线帧 → 高度梯度。不支持的节点/网格回退 EMIT 三图烘焙。
#   2. 镜像 Neumann 泊松积分 → 物理高度场(平贴严格 0) → 开放边界距离衰减。
#   3. Multires 建层只为数据结构(隐藏态跑 subdivide; reshape 完整覆写 MDISPS);
#      重建时层数匹配则整段跳过。
#   4. 细分基面 = 岛界折痕锁定的 Catmull-Clark 极限曲面(Subsurf 求值副本,
#      use_limit_surface: 任何级别都是同一极限曲面的嵌套采样): 低模粗曲率由
#      细分平滑(法线图只携带高频细节), 而所有 UV 岛边界边 crease=1 + 岛界顶点
#      vertex crease=1——折痕链逐级取中点且端点钉死, 边界折线被精确锁在原位
#      (CC 默认把边界链平滑成 B 样条曲线, 即"边缘软化"/卡片缝隙的根源)。
#      UV 保持线性插值(uv_smooth=NONE), 采样对位不随细分漂移。
#   5. 位移是光滑场作用于光滑曲面, 全链无离散值直读: 方向 = 极限曲面自身的
#      光滑法线场(极限采样网格的平滑顶点法线, O(顶点距²) 收敛——平坦低模的
#      Phong 插值场在每条基面边有 C0 折痕, 会把线框浮雕进曲面, 已废除);
#      高度 = 固定物理场 × 级别匹配重建滤波(线性网格要呈现平滑曲面, 内容
#      波长须 ≳6×顶点距; 如同曲线之于控制点——少点=同一条曲线的光滑粗版)
#      × 分辨率无关的源噪声地板(8bit/BC 压缩坡度噪声积分成固定尺度凹凸)。
#      采样高度(B 样条 C2) → 逐岛去趋势/缝合 → 边界硬锁 → 位移 → reshape 写回。
#
# 重复点"应用"= 从基面重建(幂等), 改倍数即所见即所得(全缓存命中, 只剩写回)。

import threading
import time

import bpy
import numpy as np
from bpy_extras.io_utils import ImportHelper

from . import core

MAX_LEVELS = 9
BAKE_MARGIN = 16
BAKE_SAMPLES = 1   # EMIT 烘焙无噪声, 1 采样足够(回退路径)

# 运行期缓存(只认网格/材质"身份", 不追踪材质节点内容变化)
_grad_cache = {}     # 前端结果: (gx, gy, wmap)
_island_cache = {}
_joint_cache = {}    # 联合求解结果: 未乘 disp_scale 的细分顶点标量位移
_active_joint_job = None


class _JointSolveRequest(RuntimeError):
    """Detached NumPy inputs for a modal background solve."""

    def __init__(self, cache_key, positional, keyword):
        super().__init__("联合优化等待后台求解")
        self.cache_key = cache_key
        self.positional = positional
        self.keyword = keyword


def _cache_put(cache, key, val, cap=2):
    cache.pop(key, None)
    while len(cache) >= cap:
        cache.pop(next(iter(cache)))
    cache[key] = val


def clear_caches():
    global _active_joint_job
    if _active_joint_job is not None:
        cancel_event = getattr(_active_joint_job, "_cancel_event", None)
        if cancel_event is not None:
            cancel_event.set()
    _grad_cache.clear()
    _island_cache.clear()
    _joint_cache.clear()


# ---------------------------------------------------------------------------
# 数据读取(全 foreach_get, 零 Python 逐元素循环)
# ---------------------------------------------------------------------------

def _read_loop_uvs(me):
    """兼容 3.5+ 的 layer.uv 与旧 layer.data 两种访问路径。"""
    layer = me.uv_layers.active
    if layer is None:
        raise RuntimeError("网格没有活动 UV 层")
    n = len(me.loops)
    buf = np.empty(n * 2, np.float32)
    try:
        layer.uv.foreach_get("vector", buf)
    except (AttributeError, TypeError):
        layer.data.foreach_get("uv", buf)
    return buf.reshape(-1, 2)


def _read_loop_verts(me):
    n = len(me.loops)
    buf = np.empty(n, np.int32)
    me.loops.foreach_get("vertex_index", buf)
    return buf


def _read_vert_cos(me):
    n = len(me.vertices)
    buf = np.empty(n * 3, np.float32)
    me.vertices.foreach_get("co", buf)
    return buf.reshape(-1, 3)


def _optional_float_attribute(me, name, domain):
    """Read an optional float attribute, treating absence as an empty array."""
    attribute = me.attributes.get(name)
    if (attribute is None or attribute.domain != domain
            or attribute.data_type != 'FLOAT'):
        return np.empty(0, np.float32)
    values = np.empty(len(attribute.data), np.float32)
    attribute.data.foreach_get("value", values)
    return values


def _joint_solution_cache_key(me, loop_uv, loop_vert, loop_total,
                              gx, gy, wmap, level,
                              position_weight, irls_iters):
    """Exact cache key for an unscaled joint-optimization solution.

    Unlike the interactive gradient cache, this key hashes complete content:
    base geometry/topology, UVs, user creases, and the post-filter gradient
    fields.  Reusing a million-vertex solution after a subtle edit would be far
    worse than spending a fraction of a second hashing these arrays.
    """
    loop_edge = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("edge_index", loop_edge)
    edge_vert = np.empty(len(me.edges) * 2, np.int32)
    me.edges.foreach_get("vertices", edge_vert)
    geometry_digest = core.content_digest(
        _read_vert_cos(me),
        loop_vert,
        loop_edge,
        edge_vert,
        loop_total,
        loop_uv,
        _optional_float_attribute(me, "crease_edge", 'EDGE'),
        _optional_float_attribute(me, "crease_vert", 'POINT'),
    )
    gradient_digest = core.content_digest(gx, gy, wmap)
    return (
        "joint-solution-v1",
        geometry_digest,
        gradient_digest,
        int(level),
        float(position_weight),
        int(irls_iters),
        400,
        1e-5,
    )


def _uv_fill(me, loop_uv):
    """UV 占用率(自动级别用): 三角化 UV 面积之和, 裁到 [0.02, 1]。"""
    me.calc_loop_triangles()
    t = len(me.loop_triangles)
    tl = np.empty(t * 3, np.int32)
    me.loop_triangles.foreach_get("loops", tl)
    uvt = loop_uv[tl].reshape(-1, 3, 2)
    e1 = uvt[:, 1] - uvt[:, 0]
    e2 = uvt[:, 2] - uvt[:, 0]
    uv_area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).sum()
    return float(min(max(uv_area, 0.02), 1.0))


# ---------------------------------------------------------------------------
# 法线来源判定
# ---------------------------------------------------------------------------

def _tree_has_normal_map(nt, seen=None):
    if nt is None:
        return False
    if seen is None:
        seen = set()
    if nt.name_full in seen:
        return False
    seen.add(nt.name_full)
    for n in nt.nodes:
        if n.type == 'NORMAL_MAP':
            return True
        if n.type == 'GROUP' and _tree_has_normal_map(n.node_tree, seen):
            return True
    return False


def _has_material_normal_chain(obj):
    return any(s.material is not None and _tree_has_normal_map(s.material.node_tree)
               for s in obj.material_slots)


def _resolve_source(obj, s):
    """返回 'MATERIAL' 或 'IMAGE'。AUTO 优先材质自带法线链(最忠实, 含通道重建网络)。"""
    if s.source == 'MATERIAL':
        if not _has_material_normal_chain(obj):
            raise RuntimeError("物体材质里没有 Normal Map 节点, 无法按材质求值; 请改用贴图模式")
        return 'MATERIAL'
    if s.source == 'IMAGE':
        if s.image is None:
            raise RuntimeError("贴图模式需要先选择法线贴图")
        return 'IMAGE'
    if _has_material_normal_chain(obj):
        return 'MATERIAL'
    if s.image is not None:
        return 'IMAGE'
    raise RuntimeError("物体材质没有 Normal Map 节点, 也没有选择贴图——两者需有其一")


def _upstream_image_sizes(start_socket, seen=None):
    """从某 socket 沿输入连线向上游 BFS, 收集经过的 Image Texture 节点分辨率。"""
    if seen is None:
        seen = set()
    sizes = []
    stack = [start_socket]
    while stack:
        sock = stack.pop()
        if not sock.is_linked:
            continue
        node = sock.links[0].from_node
        key = (node.id_data.name_full, node.name)
        if key in seen:
            continue
        seen.add(key)
        if node.type == 'TEX_IMAGE' and node.image is not None:
            w, h = node.image.size
            if w > 0 and h > 0:
                sizes.append(max(w, h))
        # 节点组内部不展开(求值器本身也不支持 GROUP, 命中即整体回退烘焙路径;
        # 分辨率探测保守跳过, 不影响正确性, 只影响自动选到的工作分辨率)
        for inp in node.inputs:
            stack.append(inp)
    return sizes


def _native_resolution(obj, source, image):
    """工作分辨率 = 实际接入的法线贴图原生分辨率, 不再由用户猜数字。

    高于源贴图分辨率的网格只是把已有像素插值放大, 不产生任何新细节还多耗算力;
    低于源分辨率则白白丢弃作者烘焙进贴图的信息。两者都没有意义, 直接对齐现实。
    MATERIAL 来源沿每个材质 Normal Map 节点的 Color 输入网络回溯, 取所有材质槽
    命中的最大分辨率(多材质共享同一张工作网格); 找不到则回退 2048。
    """
    if source == 'IMAGE':
        w, h = image.size
        return max(w, h, 64)
    sizes = []
    for slot in obj.material_slots:
        mat = slot.material
        if mat is None or mat.node_tree is None:
            continue
        for n in mat.node_tree.nodes:
            if n.type == 'NORMAL_MAP':
                sizes.extend(_upstream_image_sizes(n.inputs['Color']))
    if not sizes:
        print("[NormalMapToMesh] 警告: 未在材质法线链中找到贴图, 工作分辨率回退 2048")
        return 2048
    return max(sizes)


# ---------------------------------------------------------------------------
# 直算前端: numpy 材质法线链求值器
# ---------------------------------------------------------------------------

class _NodeEvalUnsupported(Exception):
    """材质网络含求值器不支持的节点/接法 → 整体回退 Cycles 烘焙路径。"""


def _read_image_grid(img, size):
    """图像重采样到 (size, size, 4): 与烘焙语义一致——网格 texel 中心做双线性。
    分辨率恰好相同时为逐位直读。"""
    if img is None:
        raise _NodeEvalUnsupported("图像节点没有图像")
    w, h = img.size
    if w == 0 or h == 0:
        raise RuntimeError(f"贴图 '{img.name}' 没有像素数据(文件缺失?)")
    ch = img.channels
    buf = np.empty(w * h * ch, np.float32)
    img.pixels.foreach_get(buf)
    px = buf.reshape(h, w, ch)
    if ch < 4:
        rgba = np.ones((h, w, 4), np.float32)
        rgba[..., :ch] = px
        px = rgba
    if (w, h) == (size, size):
        return px[..., :4]
    uu = ((np.arange(size) + 0.5) / size).astype(np.float32)
    grid_u = np.broadcast_to(uu[None, :], (size, size)).ravel()
    grid_v = np.broadcast_to(uu[:, None], (size, size)).ravel()
    out = core.sample_bilinear_wrap(px[..., :4], grid_u, grid_v)
    return out.reshape(size, size, 4).astype(np.float32)


def _as_scalar(x, size):
    if isinstance(x, np.ndarray):
        if x.ndim == 3:
            # 颜色隐转标量: 取平均(Blender 隐转是亮度, 法线链里几乎不出现——保守拒绝)
            raise _NodeEvalUnsupported("颜色→标量隐式转换")
        return x
    return float(x)


def _as_vec3(x, size):
    if isinstance(x, np.ndarray):
        if x.ndim == 2:
            return np.repeat(x[..., None], 3, axis=2)
        return x[..., :3]
    if isinstance(x, (int, float)):
        return np.full((size, size, 3), float(x), np.float32)
    v = np.asarray(x, np.float32)[:3]
    return np.broadcast_to(v, (size, size, 3)).copy()


def _eval_socket(socket, size, memo):
    """递归求值输出 socket → float / (S,S) / (S,S,3)。不支持 → _NodeEvalUnsupported。"""
    key = (socket.node.name, socket.identifier)
    if key in memo:
        return memo[key]
    node = socket.node
    nt = node.type

    def inp(i):
        sk = node.inputs[i]
        if sk.is_linked:
            return _eval_socket(sk.links[0].from_socket, size, memo)
        dv = sk.default_value
        try:
            return float(dv)
        except TypeError:
            return tuple(dv)[:3]

    if nt == 'REROUTE':
        val = _eval_socket(node.inputs[0].links[0].from_socket, size, memo) \
            if node.inputs[0].is_linked else 0.0
    elif nt == 'TEX_IMAGE':
        if node.inputs['Vector'].is_linked:
            raise _NodeEvalUnsupported("图像节点带自定义 Vector 输入")
        rgba = _read_image_grid(node.image, size)
        if socket.name == 'Alpha':
            val = rgba[..., 3].copy()
        else:
            val = rgba[..., :3].copy()
    elif nt in ('SEPARATE_COLOR', 'SEPRGB', 'SEPARATE_XYZ', 'SEPXYZ'):
        if nt == 'SEPARATE_COLOR' and getattr(node, 'mode', 'RGB') != 'RGB':
            raise _NodeEvalUnsupported(f"Separate Color 模式 {node.mode}")
        vec = _as_vec3(inp(0), size)
        idx = {'Red': 0, 'Green': 1, 'Blue': 2, 'X': 0, 'Y': 1, 'Z': 2}[socket.name]
        val = vec[..., idx].copy()
    elif nt in ('COMBINE_COLOR', 'COMBRGB', 'COMBINE_XYZ', 'COMBXYZ'):
        if nt == 'COMBINE_COLOR' and getattr(node, 'mode', 'RGB') != 'RGB':
            raise _NodeEvalUnsupported(f"Combine Color 模式 {node.mode}")
        parts = [_as_scalar(inp(i), size) for i in range(3)]
        if all(isinstance(p, float) for p in parts):
            val = tuple(parts)
        else:
            parts = [p if isinstance(p, np.ndarray)
                     else np.full((size, size), p, np.float32) for p in parts]
            val = np.stack(parts, axis=-1).astype(np.float32)
    elif nt == 'MATH':
        op = node.operation
        a = _as_scalar(inp(0), size)
        b = _as_scalar(inp(1), size) if len(node.inputs) > 1 else 0.0
        if op == 'ADD':
            val = a + b
        elif op == 'SUBTRACT':
            val = a - b
        elif op == 'MULTIPLY':
            val = a * b
        elif op == 'DIVIDE':
            val = a / np.maximum(np.abs(b), 1e-20) * np.sign(b) if isinstance(b, np.ndarray) \
                else (a / b if b != 0.0 else a * 0.0)
        elif op == 'MULTIPLY_ADD':
            val = a * b + _as_scalar(inp(2), size)
        elif op == 'POWER':
            val = np.power(np.maximum(a, 0.0), b) if isinstance(a, np.ndarray) else a ** b
        elif op == 'SQRT':
            val = np.sqrt(np.maximum(a, 0.0))
        elif op == 'ABSOLUTE':
            val = np.abs(a)
        elif op == 'MINIMUM':
            val = np.minimum(a, b)
        elif op == 'MAXIMUM':
            val = np.maximum(a, b)
        elif op == 'FLOOR':
            val = np.floor(a)
        elif op == 'ROUND':
            val = np.round(a)
        elif op == 'FRACT':
            val = a - np.floor(a)
        else:
            raise _NodeEvalUnsupported(f"Math 运算 {op}")
        if node.use_clamp:
            val = np.clip(val, 0.0, 1.0)
    elif nt == 'VECT_MATH':
        op = node.operation
        a = _as_vec3(inp(0), size)
        b = _as_vec3(inp(1), size) if len(node.inputs) > 1 else None
        if op == 'ADD':
            val = a + b
        elif op == 'SUBTRACT':
            val = a - b
        elif op == 'MULTIPLY':
            val = a * b
        elif op == 'DIVIDE':
            val = a / np.where(np.abs(b) < 1e-20, 1.0, b)
        elif op == 'MULTIPLY_ADD':
            val = a * b + _as_vec3(inp(2), size)
        elif op == 'SCALE':
            sc = node.inputs['Scale']
            scv = _eval_socket(sc.links[0].from_socket, size, memo) if sc.is_linked \
                else float(sc.default_value)
            val = a * (scv[..., None] if isinstance(scv, np.ndarray) else scv)
        elif op == 'NORMALIZE':
            ln = np.linalg.norm(a, axis=-1, keepdims=True)
            val = a / np.maximum(ln, 1e-20)
        elif op == 'DOT_PRODUCT':
            val = np.einsum('...i,...i->...', a, b).astype(np.float32)
        elif op == 'CROSS_PRODUCT':
            val = np.cross(a, b).astype(np.float32)
        elif op == 'LENGTH':
            val = np.linalg.norm(a, axis=-1).astype(np.float32)
        else:
            raise _NodeEvalUnsupported(f"Vector Math 运算 {op}")
        if isinstance(val, np.ndarray) and socket.name == 'Value' and val.ndim == 3:
            raise _NodeEvalUnsupported(f"Vector Math {op} 的 Value 输出")
    elif nt == 'VALUE':
        val = float(node.outputs[0].default_value)
    elif nt == 'RGB':
        val = tuple(node.outputs[0].default_value)[:3]
    elif nt == 'GAMMA':
        a = _as_vec3(inp(0), size)
        g = _as_scalar(inp(1), size)
        val = np.power(np.maximum(a, 0.0), g)
    elif nt == 'INVERT':
        fac = _as_scalar(inp(0), size)
        col = _as_vec3(inp(1), size)
        val = col + (1.0 - 2.0 * col) * (fac[..., None] if isinstance(fac, np.ndarray) else fac)
    else:
        raise _NodeEvalUnsupported(f"节点类型 {nt}")
    memo[key] = val
    return val


def _eval_material_tangent_map(mat, size):
    """numpy 求值材质法线链 → (S,S,3) 切线空间法线(已解码, 含 Strength)。

    取第一个 Normal Map 节点的 Color 输入上游网络求值, t = 2c−1;
    Strength ≠ 1 时 t' = (0,0,1)(1−s) + t·s (Normal Map 节点的线性混合语义)。
    无 Normal Map 节点 → None(平坦)。
    """
    nt_tree = mat.node_tree
    nmaps = [n for n in nt_tree.nodes if n.type == 'NORMAL_MAP']
    if not nmaps:
        return None
    if len(nmaps) > 1:
        print(f"[NormalMapToMesh] 警告: 材质 '{mat.name}' 有 {len(nmaps)} 个 Normal Map, 取第一个")
    nmap = nmaps[0]
    if nmap.space != 'TANGENT':
        raise _NodeEvalUnsupported(f"Normal Map 空间 {nmap.space}")
    if nmap.inputs['Strength'].is_linked:
        raise _NodeEvalUnsupported("Normal Map Strength 被连线")
    strength = float(nmap.inputs['Strength'].default_value)
    csock = nmap.inputs['Color']
    if not csock.is_linked:
        col = np.broadcast_to(np.array([0.5, 0.5, 1.0], np.float32), (size, size, 3)).copy()
    else:
        col = _as_vec3(_eval_socket(csock.links[0].from_socket, size, {}), size)
    t = col.astype(np.float32) * 2.0 - 1.0
    if strength != 1.0:
        flat = np.array([0.0, 0.0, 1.0], np.float32)
        t = flat * (1.0 - strength) + t * strength
    return t


# ---------------------------------------------------------------------------
# 直算前端: 切线帧光栅化 + 梯度
# ---------------------------------------------------------------------------

def _mesh_fingerprint(me, loop_uv):
    fp = hash(loop_uv[:: max(1, loop_uv.shape[0] // 4096)].tobytes())
    return (me.name_full, len(me.vertices), len(me.polygons), len(me.loops), fp)


def _gradients_direct(obj, me, source, image, size, loop_uv, loop_vert,
                      deadzone=0.0, slope_limit=0.0):
    """免烘焙直算: mikktspace 切线帧标量 + 逐三角形解析 ∂P → UV 高度梯度。

    shader 等价求值纪律——几何属性不进"赢家覆盖"的共享网格:
    重叠 UV 卡片(正/背面、镜像复用、图集多层)会让逐 texel 覆盖形成
    逐三角形补丁的属性马赛克, 正/背面梯度互为相反数, 积分后成严重锯齿。
    因此: ①梯度帧标量只由**正 UV 绕向**三角形贡献且逐 texel **平均**
    (孤儿镜像岛用负绕向做二次补洞); ②位移方向不进网格, 由消费端取
    各 loop 自己网格的平滑角法线。返回 (gx, gy, wmap)。
    """
    uv_name = me.uv_layers.active.name
    try:
        me.calc_tangents(uvmap=uv_name)
    except RuntimeError as e:
        raise _NodeEvalUnsupported(f"calc_tangents 失败(网格含五边以上面?): {e}")
    n_l = len(me.loops)
    tan = np.empty(n_l * 3, np.float32)
    me.loops.foreach_get("tangent", tan)
    tan = tan.reshape(-1, 3)
    sign = np.empty(n_l, np.float32)
    me.loops.foreach_get("bitangent_sign", sign)
    nrm = np.empty(n_l * 3, np.float32)
    me.corner_normals.foreach_get("vector", nrm)
    nrm = nrm.reshape(-1, 3)
    me.free_tangents()

    pos = _read_vert_cos(me)
    me.calc_loop_triangles()
    t_count = len(me.loop_triangles)
    tl = np.empty(t_count * 3, np.int32)
    me.loop_triangles.foreach_get("loops", tl)
    tl = tl.reshape(-1, 3)
    tp = np.empty(t_count, np.int32)
    me.loop_triangles.foreach_get("polygon_index", tp)
    pmat = np.empty(len(me.polygons), np.int32)
    me.polygons.foreach_get("material_index", pmat)

    tri_uv = loop_uv[tl.ravel()].reshape(-1, 3, 2)
    tri_pos = pos[loop_vert[tl.ravel()]].reshape(-1, 3, 3)
    d1 = (tri_uv[:, 1] - tri_uv[:, 0]).astype(np.float64)
    d2 = (tri_uv[:, 2] - tri_uv[:, 0]).astype(np.float64)
    e1 = (tri_pos[:, 1] - tri_pos[:, 0]).astype(np.float64)
    e2 = (tri_pos[:, 2] - tri_pos[:, 0]).astype(np.float64)
    det = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    valid = np.abs(det) > 1e-16
    det_safe = np.where(valid, det, 1.0)
    pu = (e1 * d2[:, 1, None] - e2 * d1[:, 1, None]) / det_safe[:, None]
    pv = (e2 * d1[:, 0, None] - e1 * d2[:, 0, None]) / det_safe[:, None]

    # 逐角切线帧标量: au=T·Pu, bu=B·Pu, av=T·Pv, bv=B·Pv (B = sign·N×T)
    tc = tan[tl.ravel()].reshape(-1, 3, 3).astype(np.float64)
    nc = nrm[tl.ravel()].reshape(-1, 3, 3).astype(np.float64)
    sc = sign[tl.ravel()].reshape(-1, 3).astype(np.float64)
    bc = np.cross(nc, tc) * sc[:, :, None]
    attrs = np.empty((t_count, 3, 4), np.float32)
    attrs[:, :, 0] = np.einsum('tcj,tj->tc', tc, pu)
    attrs[:, :, 1] = np.einsum('tcj,tj->tc', bc, pu)
    attrs[:, :, 2] = np.einsum('tcj,tj->tc', tc, pv)
    attrs[:, :, 3] = np.einsum('tcj,tj->tc', bc, pv)

    # 正绕向为主贡献(逐 texel 平均), 负绕向只补正绕向没覆盖的洞
    pos_sel = valid & (det > 0)
    neg_sel = valid & (det < 0)
    sum_p, cnt_p = core.rasterize_tris(tri_uv[pos_sel], attrs[pos_sel], size,
                                       accumulate=True)
    frame = np.zeros((size, size, 4), np.float32)
    covered_p = cnt_p > 0
    frame[covered_p] = sum_p[covered_p] / cnt_p[covered_p][:, None]
    mask0 = covered_p
    if neg_sel.any():
        sum_n, cnt_n = core.rasterize_tris(tri_uv[neg_sel], attrs[neg_sel], size,
                                           accumulate=True)
        fill = (~covered_p) & (cnt_n > 0)
        if fill.any():
            frame[fill] = sum_n[fill] / cnt_n[fill][:, None]
            mask0 = covered_p | fill

    # 切线空间法线图
    if source == 'IMAGE':
        t_map = _read_image_grid(image, size)[..., :3] * 2.0 - 1.0
    else:
        flat = np.broadcast_to(np.array([0.0, 0.0, 1.0], np.float32), (size, size, 3))
        mat_maps = []
        for slot in obj.material_slots:
            m = slot.material
            t = _eval_material_tangent_map(m, size) if m is not None else None
            mat_maps.append(flat if t is None else t)
        if not mat_maps:
            mat_maps = [flat]
        if len(mat_maps) == 1:
            t_map = np.ascontiguousarray(mat_maps[0])
        else:
            # 多材质槽: 逐 texel 材质号(覆盖式光栅化)选择对应贴图链结果
            mat_attr = np.broadcast_to(
                pmat[tp].astype(np.float32)[:, None, None], (t_count, 3, 1)).copy()
            mgrid, _ = core.rasterize_tris(tri_uv[valid], mat_attr[valid], size)
            mi = np.clip(np.round(mgrid[..., 0]).astype(np.int64), 0, len(mat_maps) - 1)
            t_map = np.empty((size, size, 3), np.float32)
            for i, m in enumerate(mat_maps):
                sel = mi == i
                t_map[sel] = m[sel]

    # 梯度只取真实 UV 覆盖区: 外扩 margin 的复制内容会虚增积分能量
    gx, gy, _ = core.gradients_from_frame_scalars(
        t_map, frame[..., 0], frame[..., 1], frame[..., 2], frame[..., 3], mask0,
        deadzone=deadzone, slope_limit=slope_limit)

    # 采样有效域: 掩码外扩(岛边界 B 样条采样不吃到无效 texel)
    wmap = core.dilate_mask(mask0, BAKE_MARGIN).astype(np.float32)
    return gx, gy, wmap


def _gradients_cached(context, obj, me, source, image, size, loop_uv, loop_vert,
                      force_bake, deadzone, slope_limit):
    mats = tuple(s.material.name_full if s.material else '' for s in obj.material_slots)
    key = (_mesh_fingerprint(me, loop_uv), mats, source,
           image.name_full if image is not None else '', int(size), bool(force_bake),
           round(float(deadzone), 6), round(float(slope_limit), 6))
    got = _grad_cache.get(key)
    if got is not None:
        return got

    result = None
    if not force_bake:
        try:
            result = _gradients_direct(obj, me, source, image, size, loop_uv, loop_vert,
                                       deadzone=deadzone, slope_limit=slope_limit)
            print("[NormalMapToMesh] 前端: 直算(免烘焙)")
        except _NodeEvalUnsupported as e:
            print(f"[NormalMapToMesh] 直算不支持({e}), 回退 Cycles 烘焙")
    if result is None:
        rgb_detail, rgb_base, pos_map = _bake_triple(context, obj, source, image, size, loop_uv)
        gx, gy, wmap = core.height_gradients(rgb_detail, rgb_base, pos_map,
                                             deadzone=deadzone, slope_limit=slope_limit)
        result = (gx, gy, wmap)
    _cache_put(_grad_cache, key, result)
    return result


# ---------------------------------------------------------------------------
# 回退路径: Cycles EMIT 三图烘焙 (v3 原样保留)
# ---------------------------------------------------------------------------

def _build_encoder(nt, src_socket, vector_type='NORMAL', encode=True):
    """向量(世界空间) → Vector Transform 转物体空间 → (可选 ×0.5+0.5 编码) → Emission。"""
    vt = nt.nodes.new('ShaderNodeVectorTransform')
    vt.vector_type = vector_type
    vt.convert_from = 'WORLD'
    vt.convert_to = 'OBJECT'
    nt.links.new(vt.inputs['Vector'], src_socket)
    nodes = [vt]
    out_socket = vt.outputs['Vector']
    if encode:
        vm = nt.nodes.new('ShaderNodeVectorMath')
        vm.operation = 'MULTIPLY_ADD'
        vm.inputs[1].default_value = (0.5, 0.5, 0.5)
        vm.inputs[2].default_value = (0.5, 0.5, 0.5)
        nt.links.new(vm.inputs[0], out_socket)
        out_socket = vm.outputs['Vector']
        nodes.append(vm)
    em = nt.nodes.new('ShaderNodeEmission')
    nt.links.new(em.inputs['Color'], out_socket)
    nodes.append(em)
    return nodes, em


def _bake_once(context, obj, kind, source, image, bake_size):
    """单次物体空间 EMIT 烘焙 → (H, W, 3) float32。(回退路径)"""
    me = obj.data
    bake_img = None
    tmp_mat = None
    saved_slots = None
    slot_appended = False
    inserted = []
    grafts = []
    try:
        bake_img = bpy.data.images.new(f"NMTM_bake_{kind}", width=bake_size,
                                       height=bake_size, float_buffer=True)
        bake_img.colorspace_settings.name = 'Non-Color'

        if kind != 'DETAIL' or source == 'IMAGE':
            tmp_mat = bpy.data.materials.new("NMTM_bake_mat")
            nt = tmp_mat.node_tree
            nt.nodes.clear()
            out = nt.nodes.new('ShaderNodeOutputMaterial')
            if kind == 'DETAIL':
                timg = nt.nodes.new('ShaderNodeTexImage')
                timg.image = image
                nmap = nt.nodes.new('ShaderNodeNormalMap')
                nt.links.new(nmap.inputs['Color'], timg.outputs['Color'])
                src_socket, vtype, enc = nmap.outputs['Normal'], 'NORMAL', True
            elif kind == 'BASELINE':
                geo = nt.nodes.new('ShaderNodeNewGeometry')
                src_socket, vtype, enc = geo.outputs['Normal'], 'NORMAL', True
            else:   # POSITION
                geo = nt.nodes.new('ShaderNodeNewGeometry')
                src_socket, vtype, enc = geo.outputs['Position'], 'POINT', False
            _, em = _build_encoder(nt, src_socket, vtype, enc)
            nt.links.new(out.inputs['Surface'], em.outputs['Emission'])
            for n in nt.nodes:
                n.select = False
            target = nt.nodes.new('ShaderNodeTexImage')
            target.image = bake_img
            target.select = True
            nt.nodes.active = target
            if obj.material_slots:
                saved_slots = [s.material for s in obj.material_slots]
                for i in range(len(obj.material_slots)):
                    obj.material_slots[i].material = tmp_mat
            else:
                me.materials.append(tmp_mat)
                slot_appended = True
        else:
            done = set()
            for slot in obj.material_slots:
                mat = slot.material
                if mat is None or mat.name in done:
                    continue
                done.add(mat.name)
                nt = mat.node_tree
                out_node = nt.get_output_node('CYCLES')
                if out_node is None:
                    continue
                surf = out_node.inputs['Surface']
                orig_from = None
                if surf.is_linked:
                    lk = surf.links[0]
                    orig_from = (lk.from_node.name, lk.from_socket.name)
                nmap_names = [n.name for n in nt.nodes if n.type == 'NORMAL_MAP']
                if len(nmap_names) > 1:
                    print(f"[NormalMapToMesh] 警告: 材质 '{mat.name}' 有 "
                          f"{len(nmap_names)} 个 Normal Map 节点, 取第一个")
                new_nodes = []
                if nmap_names:
                    src_socket = nt.nodes[nmap_names[0]].outputs['Normal']
                else:
                    geo = nt.nodes.new('ShaderNodeNewGeometry')
                    new_nodes.append(geo)
                    src_socket = geo.outputs['Normal']
                enc_nodes, em = _build_encoder(nt, src_socket)
                new_nodes.extend(enc_nodes)
                nt.links.new(out_node.inputs['Surface'], em.outputs['Emission'])
                grafts.append((mat.name, [n.name for n in new_nodes],
                               out_node.name, orig_from))

                prev_active = nt.nodes.active.name if nt.nodes.active else ''
                # bpy 集合迭代出的是新包装对象, `is` 比较恒假——先全清再对持有的原始引用赋值
                for n in nt.nodes:
                    n.select = False
                target = nt.nodes.new('ShaderNodeTexImage')
                target.image = bake_img
                target.location = (0, 600)
                target.select = True
                nt.nodes.active = target
                inserted.append((mat.name, target.name, prev_active))

        bpy.ops.object.bake(type='EMIT')

        buf = np.empty(bake_size * bake_size * 4, np.float32)
        bake_img.pixels.foreach_get(buf)
        rgb = buf.reshape(bake_size, bake_size, 4)[..., :3].astype(np.float32, copy=True)
        if float(rgb.std()) < 1e-5:
            raise RuntimeError(f"{kind} 烘焙结果是纯色, 物体空间法线烘焙未生效(检查材质与 UV)")
        return rgb
    finally:
        for mat_name, node_names, out_name, orig_from in grafts:
            mat = bpy.data.materials.get(mat_name)
            if mat is None:
                continue
            nt = mat.node_tree
            for nn in node_names:
                n = nt.nodes.get(nn)
                if n is not None:
                    nt.nodes.remove(n)
            if orig_from is not None:
                out_node = nt.nodes.get(out_name)
                from_node = nt.nodes.get(orig_from[0])
                if out_node is not None and from_node is not None:
                    try:
                        nt.links.new(out_node.inputs['Surface'],
                                     from_node.outputs[orig_from[1]])
                    except Exception:
                        pass
        for mat_name, node_name, prev_active in inserted:
            mat = bpy.data.materials.get(mat_name)
            if mat is None:
                continue
            nt = mat.node_tree
            n = nt.nodes.get(node_name)
            if n is not None:
                nt.nodes.remove(n)
            if prev_active:
                pa = nt.nodes.get(prev_active)
                if pa is not None:
                    nt.nodes.active = pa
        if saved_slots is not None:
            for i, m in enumerate(saved_slots):
                obj.material_slots[i].material = m
        if slot_appended:
            me.materials.pop(index=len(me.materials) - 1)
        if tmp_mat is not None:
            bpy.data.materials.remove(tmp_mat)
        if bake_img is not None:
            bpy.data.images.remove(bake_img)


def _bake_triple(context, obj, source, image, bake_size, loop_uv):
    """三次同参数物体空间 EMIT 烘焙(n1, n0, P)。(回退路径, 无缓存——由上层缓存)"""
    if source == 'MATERIAL':
        for slot in obj.material_slots:
            if slot.material is not None and slot.material.library is not None:
                raise RuntimeError(
                    f"材质 '{slot.material.name}' 来自链接库, 无法插入烘焙节点; 请先 Make Local")
    if source == 'IMAGE':
        try:
            if image.source == 'FILE' and image.colorspace_settings.name != 'Non-Color':
                image.colorspace_settings.name = 'Non-Color'
        except Exception:
            pass

    scene = context.scene
    saved_scene = (scene.render.engine, scene.cycles.device, scene.cycles.samples,
                   scene.cycles.bake_type, scene.render.bake.use_selected_to_active,
                   scene.render.bake.margin)
    saved_hide = [(o.name, o.hide_render) for o in bpy.data.objects]
    saved_show_render = [(m.name, m.show_render) for m in obj.modifiers]
    try:
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'CPU'
        scene.cycles.samples = BAKE_SAMPLES
        scene.cycles.bake_type = 'EMIT'
        scene.render.bake.use_selected_to_active = False
        scene.render.bake.margin = BAKE_MARGIN

        for o in bpy.data.objects:
            o.hide_render = True
        obj.hide_render = False
        for o in list(context.selected_objects):
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        for m in obj.modifiers:
            m.show_render = False

        rgb_detail = _bake_once(context, obj, 'DETAIL', source, image, bake_size)
        rgb_base = _bake_once(context, obj, 'BASELINE', source, None, bake_size)
        pos_map = _bake_once(context, obj, 'POSITION', source, None, bake_size)
    finally:
        for name, vis in saved_hide:
            o = bpy.data.objects.get(name)
            if o is not None:
                o.hide_render = vis
        for name, vis in saved_show_render:
            m = obj.modifiers.get(name)
            if m is not None:
                m.show_render = vis
        (scene.render.engine, scene.cycles.device, scene.cycles.samples,
         scene.cycles.bake_type, scene.render.bake.use_selected_to_active,
         scene.render.bake.margin) = saved_scene

    return rgb_detail, rgb_base, pos_map


# ---------------------------------------------------------------------------
# 岛界折痕锁定的 Catmull-Clark 极限曲面求值(拓扑与 multires 逐位一致, 实测)
# ---------------------------------------------------------------------------

def _island_border_edges(me, loop_uv, loop_vert, labels, loop_total,
                         include_uv_seams=True):
    """细分折痕边集合 + 其端点集合。

    边界 = 开放边/非流形边、两侧面属不同岛的边、两侧 UV 不连续的接缝边
    (量化口径与 face_islands 一致)。联合优化模式传 ``include_uv_seams=False``：
    只锁真实开放/非流形边，普通 UV 缝继续共享同一张 Catmull-Clark 曲面。
    返回 (edge_bool[E], vert_bool[V])。
    """
    e_count = len(me.edges)
    l_count = len(me.loops)
    le = np.empty(l_count, np.int32)
    me.loops.foreach_get("edge_index", le)
    ls = np.empty(len(me.polygons), np.int64)
    me.polygons.foreach_get("loop_start", ls)
    lt = loop_total.astype(np.int64)
    nxt = np.arange(l_count, dtype=np.int64) + 1
    nxt[ls + lt - 1] = ls
    poly_of_loop = np.repeat(np.arange(len(me.polygons), dtype=np.int64), lt)

    counts = np.bincount(le, minlength=e_count)
    border = counts != 2                       # 开放边/非流形边一律锁
    two_edges = np.flatnonzero(counts == 2)
    if include_uv_seams and two_edges.size:
        order = np.argsort(le, kind='stable')
        first = np.searchsorted(le[order], two_edges, side='left')
        l1 = order[first].astype(np.int64)
        l2 = order[first + 1].astype(np.int64)
        diff_isl = labels[poly_of_loop[l1]] != labels[poly_of_loop[l2]]
        quv = np.round(loop_uv.astype(np.float64) * 65536.0).astype(np.int64)
        lv = loop_vert.astype(np.int64)
        opp = lv[l2] != lv[l1]                 # 对向绕行(流形正常态)
        c2a = np.where(opp, nxt[l2], l2)       # 对侧面上与 l1 同顶点的角
        c2b = np.where(opp, l2, nxt[l2])
        seam = ((quv[l1] != quv[c2a]).any(axis=1)
                | (quv[nxt[l1]] != quv[c2b]).any(axis=1))
        border[two_edges[diff_isl | seam]] = True

    ev = np.empty(e_count * 2, np.int32)
    me.edges.foreach_get("vertices", ev)
    vert_pin = np.zeros(len(me.vertices), bool)
    vert_pin[ev.reshape(-1, 2)[border].ravel()] = True
    return border, vert_pin


def _subsurf_eval_mesh(context, obj, level, border_edges, border_verts):
    """网格副本 + 折痕锁定 CC Subsurf 极限求值 → 新 Mesh(调用方负责删除)。

    无算子、无选择/撤销依赖, 也天然不受 Mesh 里已有 MDISPS 影响(Subsurf 忽略之)。
    岛界边 crease=1 + 岛界顶点 vertex crease=1: 折痕链逐级取线性中点、原顶点
    钉死——边界折线精确保持原位(CC 默认把边界链平滑成 B 样条曲线 = 边缘软化);
    内部收敛到 C2 极限曲面, use_limit_surface 使任何级别都是同一曲面的嵌套采样。
    uv_smooth=NONE 保证 UV 纯线性插值(shader 重心插值同构, 采样对位不漂)。
    与用户已有折痕取 max 合并, 原网格不动。
    """
    me2 = obj.data.copy()
    ec = me2.attributes.get("crease_edge")
    if ec is None or ec.domain != 'EDGE':
        ec = me2.attributes.new("crease_edge", 'FLOAT', 'EDGE')
        cur_e = np.zeros(len(me2.edges), np.float32)
    else:
        cur_e = np.empty(len(me2.edges), np.float32)
        ec.data.foreach_get("value", cur_e)
    ec.data.foreach_set("value", np.maximum(cur_e, border_edges.astype(np.float32)))
    vc = me2.attributes.get("crease_vert")
    if vc is None or vc.domain != 'POINT':
        vc = me2.attributes.new("crease_vert", 'FLOAT', 'POINT')
        cur_v = np.zeros(len(me2.vertices), np.float32)
    else:
        cur_v = np.empty(len(me2.vertices), np.float32)
        vc.data.foreach_get("value", cur_v)
    vc.data.foreach_set("value", np.maximum(cur_v, border_verts.astype(np.float32)))

    tmp_o = bpy.data.objects.new("NMTM_subd_tmp", me2)
    context.scene.collection.objects.link(tmp_o)
    try:
        mod = tmp_o.modifiers.new("NMTM_subd", 'SUBSURF')
        mod.subdivision_type = 'CATMULL_CLARK'
        mod.levels = level
        mod.render_levels = level
        mod.quality = 4                  # multires 默认
        mod.use_limit_surface = True
        mod.use_creases = True
        mod.uv_smooth = 'NONE'
        mod.boundary_smooth = 'ALL'
        dg = context.evaluated_depsgraph_get()
        out = bpy.data.meshes.new_from_object(tmp_o.evaluated_get(dg),
                                              preserve_all_data_layers=True, depsgraph=dg)
    finally:
        bpy.data.objects.remove(tmp_o, do_unlink=True)
        try:
            bpy.data.meshes.remove(me2)
        except Exception:
            pass
    return out


def _open_edge_segments(me, loop_uv):
    """低模开放边(单面边) → UV 线段端点对 (E,2,2), 供边缘衰减场。"""
    ecount = len(me.edges)
    if ecount == 0 or len(me.loops) == 0:
        return np.zeros((0, 2, 2), np.float32)
    le = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("edge_index", le)
    open_edge = np.bincount(le, minlength=ecount) == 1
    if not open_edge.any():
        return np.zeros((0, 2, 2), np.float32)
    ls = np.empty(len(me.polygons), np.int64)
    me.polygons.foreach_get("loop_start", ls)
    lt = np.empty(len(me.polygons), np.int64)
    me.polygons.foreach_get("loop_total", lt)
    nxt = np.arange(len(me.loops), dtype=np.int64) + 1
    nxt[ls + lt - 1] = ls                      # 面内环回: 末角的下一角是首角
    sel = np.flatnonzero(open_edge[le])
    return np.stack([loop_uv[sel], loop_uv[nxt[sel]]], axis=1).astype(np.float32)


def _get_island_labels(me, loop_vert, loop_uv, loop_total):
    """基面 → UV 岛标签(按网格内容指纹缓存)。"""
    fp = hash(loop_uv[:: max(1, loop_uv.shape[0] // 4096)].tobytes())
    key = (me.name_full, len(me.polygons), len(me.loops), fp)
    got = _island_cache.get(key)
    if got is None:
        poly_of_loop = np.repeat(np.arange(len(me.polygons), dtype=np.int64), loop_total)
        got = core.face_islands(loop_vert, loop_uv, poly_of_loop, len(me.polygons))
        _cache_put(_island_cache, key, got)
    return got


# ---------------------------------------------------------------------------
# 主构建
# ---------------------------------------------------------------------------

def _find_multires(obj):
    for m in obj.modifiers:
        if m.type == 'MULTIRES':
            return m
    return None


def _auto_level(corner_count, texel_count, fill, quad_budget):
    """最小 L 使 四边形数 = corners*4^(L-1) ≥ 有效texel数; 再按预算回退。"""
    needed = texel_count * fill
    level = 1
    while corner_count * (4 ** (level - 1)) < needed and level < MAX_LEVELS:
        level += 1
    while level > 1 and corner_count * (4 ** (level - 1)) > quad_budget:
        level -= 1
    return level


def build(context, obj, s, report, *, defer_joint=False,
          expected_joint_cache_key=None):
    """核心构建。s = 场景设置 PropertyGroup。异常直接抛出, 由 Operator 兜底。"""
    t0 = time.perf_counter()
    me = obj.data

    if context.view_layer.objects.active is not obj:
        context.view_layer.objects.active = obj
    if obj.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    if me.uv_layers.active is None:
        raise RuntimeError("网格没有 UV 层, 法线贴图无从对应")

    # ---- 把无关物体临时摘出视图层求值: 每次 bpy.ops 都触发整场景深度图刷新,
    #      重型场景(如整只角色的骨骼变形网格)会被每个算子白算一遍 ----
    hidden_objs = []
    for o in context.view_layer.objects:
        if o.name != obj.name and not o.hide_get():
            try:
                o.hide_set(True)
                hidden_objs.append(o.name)
            except Exception:
                pass
    try:
        _build_inner(
            context, obj, s, report, t0,
            defer_joint=defer_joint,
            expected_joint_cache_key=expected_joint_cache_key,
        )
    finally:
        vl_objects = context.view_layer.objects
        for name in hidden_objs:
            o = vl_objects.get(name)
            if o is not None:
                try:
                    o.hide_set(False)
                except Exception:
                    pass


def _build_inner(context, obj, s, report, t0, *,
                 defer_joint=False, expected_joint_cache_key=None):
    me = obj.data
    reconstruction_mode = getattr(s, "reconstruction_mode", 'POISSON')
    joint_mode = reconstruction_mode == 'JOINT'
    if expected_joint_cache_key is not None and not joint_mode:
        raise RuntimeError(
            "重建求解器在后台联合求解期间已改变；旧结果未应用，"
            "请按当前设置重新点击“应用 / 更新”")
    joint_position_weight = float(getattr(s, "joint_position_weight", 0.1))
    joint_irls_iters = int(getattr(s, "joint_irls_iters", 3))

    source = _resolve_source(obj, s)
    loop_uv = _read_loop_uvs(me)
    loop_vert = _read_loop_verts(me)
    # 工作分辨率 = 实际接入贴图的原生分辨率, 不再由用户猜数字(见 _native_resolution)
    bake_size = _native_resolution(obj, source, s.image if source == 'IMAGE' else None)
    print(f"[NormalMapToMesh] 工作分辨率(源贴图原生) = {bake_size}px")
    gx, gy, wmap = _gradients_cached(
        context, obj, me, source, s.image if source == 'IMAGE' else None,
        bake_size, loop_uv, loop_vert, bool(s.force_bake),
        s.deadzone_lsb / 127.5, s.slope_limit)

    if not (wmap > 0).any():
        raise RuntimeError("梯度全部无效(UV 未覆盖/法线异常), 高度重建失败")
    t_front = time.perf_counter()

    fill = _uv_fill(me, loop_uv)

    # 岛标签 + 岛界折痕集合(基面属性, 细分求值与岛处理共用)
    loop_total = np.empty(len(me.polygons), np.int32)
    me.polygons.foreach_get("loop_total", loop_total)
    labels, n_islands = _get_island_labels(me, loop_vert, loop_uv, loop_total)
    border_edges, border_verts = _island_border_edges(me, loop_uv, loop_vert,
                                                     labels, loop_total,
                                                     include_uv_seams=not joint_mode)

    # ---- Multires 兼容性预检 ----
    # 真正建层延后到求解成功之后：后台求解可取消，取消时不应先破坏已有细节。
    mod = _find_multires(obj)
    owned = bool(obj.get("nmtm_owned"))
    if mod is not None and mod.total_levels > 0 and not owned:
        raise RuntimeError(
            "物体已有带层级的 Multires(非本工具创建)。为防细节丢失请先应用或移除它。")

    if s.auto_levels:
        level = _auto_level(len(me.loops), bake_size * bake_size, fill, s.quad_budget)
    else:
        level = min(s.levels, MAX_LEVELS)
        while level > 1 and len(me.loops) * (4 ** (level - 1)) > s.quad_budget:
            level -= 1
    level = max(1, level)
    quads = len(me.loops) * (4 ** (level - 1))

    # ---- 两级带限 + 求解器输入 ----
    # ① 级别匹配重建滤波(采样理论): 顶点间线性插值要呈现平滑曲面, 内容波长须
    #    ≳6×顶点距——σ=1.5×texel/顶点距。旧 0.6 系数把内容钉在每级 ~3.75 采样/
    #    波长, 呈现恒为多边形折面感且各级观感相同("升级别不变光滑"的根源)。
    #    高度物理场本身固定, 各级别是它经该滤波的光滑投影(曲线之于控制点)。
    # ② 源噪声地板(分辨率无关, 与级别无关): 8bit 量化/BC 压缩块的坡度噪声经
    #    积分放大成固定尺度凹凸(把贴图噪声过拟合成几何), 恒定 σ 压掉。
    texels_per_edge = float(np.sqrt(bake_size * bake_size * fill / max(quads, 1)))
    sigma_sampling = 1.5 * texels_per_edge
    sigma_px = float(np.sqrt(float(s.detail_smooth_px) ** 2 + sigma_sampling ** 2))
    if joint_mode:
        # 联合优化不先生成全图高度：只对两个法线梯度分量做自由边界低通，随后
        # 在真实细分拓扑上组装边位移观测。低模位置以 d=0 屏蔽项进入同一优化。
        smooth_grad = core.smooth_fields_neumann(
            np.stack([gx, gy], axis=-1), sigma_px / bake_size)
        gx_work = smooth_grad[..., 0]
        gy_work = smooth_grad[..., 1]
        field = None
    else:
        # 传统兼容后端：Neumann(镜像)积分为 UV 高度场。
        field = core.integrate_height(gx, gy, smooth_sigma=sigma_px / bake_size)
        gx_work = gy_work = None

    # 开放边界(卡片边缘)衰减: 高度场沿 UV 距离场 smoothstep 归零。场量属于
    # 低模 UV 域, 与细分级别无关; 边界顶点另有硬锁零位移兜底
    fall_stat = ""
    if int(s.edge_falloff_px) > 0:
        seg = _open_edge_segments(me, loop_uv)
        if seg.shape[0]:
            falloff = core.edge_falloff_field(seg, bake_size,
                                              int(s.edge_falloff_px))
            if joint_mode:
                gx_work *= falloff
                gy_work *= falloff
            else:
                field *= falloff
            fall_stat = f" | 边缘衰减 {int(s.edge_falloff_px)}px/{seg.shape[0]:,}段"
    joint_cache_key = None
    if joint_mode:
        joint_cache_key = _joint_solution_cache_key(
            me, loop_uv, loop_vert, loop_total,
            gx_work, gy_work, wmap, level,
            joint_position_weight, joint_irls_iters)
        if (expected_joint_cache_key is not None
                and joint_cache_key != expected_joint_cache_key):
            raise RuntimeError(
                "模型、贴图或联合参数在后台求解期间已改变；"
                "旧结果未应用，请重新点击“应用 / 更新”")
        grad_mag = np.hypot(gx_work[wmap > 0], gy_work[wmap > 0])
        field_stat = f"梯度幅值 p95 {np.percentile(grad_mag, 95):.4g}"
        solver_label = "位置+法线联合优化"
    else:
        field_stat = (f"高度场 p95 "
                      f"{np.percentile(np.abs(field[wmap > 0]), 95) * 1000:.2f}‰")
        solver_label = "UV Neumann/Poisson"
    print(f"[NormalMapToMesh] 求解器 {solver_label} | 梯度有效率 {wmap.mean():.1%} | "
          f"{field_stat} | 重建滤波 σ {sigma_sampling:.2f}px "
          f"⊕ 噪声地板 {float(s.detail_smooth_px):.1f}px{fall_stat}")

    # ---- 细分基面: 岛界折痕锁定 CC 极限曲面(Subsurf 求值副本, 拓扑与 multires
    #      逐位一致) ----
    # 临时关掉其它修改器, 保证 reshape 空间纯净(骨架变形不得混入目标面)
    # 注意: bpy RNA 包装对象不能用 `is` 比较(每次访问都是新包装), 按类型过滤
    saved_vis = [(m, m.show_viewport) for m in obj.modifiers if m.type != 'MULTIRES']
    for m, _ in saved_vis:
        m.show_viewport = False
    tmp_obj = None
    tmp_me = None
    try:
        if mod is not None:
            mod.show_viewport = False
        try:
            tmp_me = _subsurf_eval_mesh(context, obj, level, border_edges, border_verts)
        finally:
            if mod is not None:
                mod.show_viewport = True
        t_eval = time.perf_counter()

        vcount = len(tmp_me.vertices)
        expected_loops = len(me.loops) * (4 ** (level - 1)) * 4
        if len(tmp_me.loops) != expected_loops:
            raise RuntimeError(
                f"Subsurf 细分拓扑异常: {len(tmp_me.loops):,} vs 预期 {expected_loops:,}")
        lv2 = _read_loop_verts(tmp_me)
        uv2 = _read_loop_uvs(tmp_me)

        # 边界硬锁: 开放边界顶点(卡片边缘)位移严格归零——边缘偏移会把原本
        # 贴合的卡片边撕出缝隙。联合优化把它们直接从未知量消去；传统后端在
        # 采样后硬锁。普通 UV 缝不在此集合中。
        ecount = len(tmp_me.edges)
        ev = np.empty(ecount * 2, np.int32)
        tmp_me.edges.foreach_get("vertices", ev)
        ev = ev.reshape(-1, 2)
        le = np.empty(len(tmp_me.loops), np.int32)
        tmp_me.loops.foreach_get("edge_index", le)
        edge_face_count = np.bincount(le, minlength=ecount)
        boundary_verts = np.unique(ev[edge_face_count[:ecount] != 2].ravel())
        pin_mask = np.zeros(vcount, bool)
        pin_mask[boundary_verts] = True

        # 位移方向 = 极限曲面自身的光滑法线场(极限采样网格的平滑顶点法线,
        # O(顶点距²) 收敛): 位移是向量场作用于光滑曲面。
        vn = np.empty(vcount * 3, np.float32)
        tmp_me.vertex_normals.foreach_get("vector", vn)
        n0_vert = vn.reshape(-1, 3)
        n0_vert = n0_vert / np.maximum(np.linalg.norm(n0_vert, axis=1), 1e-12)[:, None]
        co = _read_vert_cos(tmp_me)

        if joint_mode:
            cached_joint = _joint_cache.get(joint_cache_key)
            cache_hit = (
                cached_joint is not None
                and cached_joint[0].shape == (vcount,)
            )
            if cache_hit:
                h_vert, joint_stats = cached_joint
                joint_stats = dict(joint_stats)
                _cache_put(_joint_cache, joint_cache_key, cached_joint)
            else:
                # 法线梯度只作为每个细分面内边的位移差观测；顶点未知量按真实
                # 网格共享，UV 缝/多岛不再产生独立高度常量，也无需 PLANE 去趋势。
                poly_count2 = len(tmp_me.polygons)
                loop_start2 = np.empty(poly_count2, np.int64)
                loop_total2 = np.empty(poly_count2, np.int64)
                tmp_me.polygons.foreach_get("loop_start", loop_start2)
                tmp_me.polygons.foreach_get("loop_total", loop_total2)
                edge_i, edge_j, target_delta, edge_weight = \
                    core.gradient_constraints_from_loops(
                        lv2, uv2, loop_start2, loop_total2,
                        gx_work, gy_work, wmap, loop_edge=le)
                if edge_i.size == 0:
                    raise RuntimeError("联合优化没有有效的法线梯度边约束")
                edge_vec = co[edge_j] - co[edge_i]
                extent = np.ptp(co.astype(np.float64), axis=0)
                reference_sq = max(float(np.dot(extent, extent)), 1e-20)
                position_measure = (
                    np.einsum("ij,ij->i", edge_vec, edge_vec) / reference_sq
                ).astype(np.float32)
                solve_positional = (
                    edge_i, edge_j, target_delta, vcount)
                solve_keyword = {
                    "base_weight": edge_weight,
                    "position_weight": joint_position_weight,
                    "position_measure": position_measure,
                    "pinned": pin_mask,
                    "irls_iters": joint_irls_iters,
                    "max_iter": 400,
                    "tolerance": 1e-5,
                }
                if defer_joint:
                    raise _JointSolveRequest(
                        joint_cache_key, solve_positional, solve_keyword)
                h_vert, joint_stats = core.solve_joint_position_normal(
                    *solve_positional, **solve_keyword)
                _cache_put(
                    _joint_cache,
                    joint_cache_key,
                    (h_vert.copy(), dict(joint_stats)),
                )
            if not joint_stats["converged"]:
                report({'WARNING'},
                       f"联合优化 PCG 未在 400 次内完全收敛；已保留当前残差 "
                       f"RMS={joint_stats['residual_rms']:.3g}")
            solve_stat = (
                f"联合约束 {joint_stats['edge_count']:,} | "
                f"{'解缓存命中' if cache_hit else '新求解'} | "
                f"PCG {joint_stats['pcg_iterations']} | "
                f"IRLS {joint_stats['irls_updates']} | "
                f"残差 p95 {joint_stats['residual_p95']:.3g} | "
                f"降权 {joint_stats['downweighted_fraction']:.1%}")
        else:
            # 传统后端：逐 loop 采样 UV 高度，逐岛去趋势并缝合。
            samp = np.stack([field, wmap], axis=-1)
            s2 = core.sample_bspline_wrap(samp, uv2[:, 0], uv2[:, 1])
            h_loop = s2[:, 0].astype(np.float32)
            w_loop = (s2[:, 1] > 0.5).astype(np.float32)
            per_face = loop_total.astype(np.int64) * (4 ** (level - 1))
            island_of_loop2 = np.repeat(np.repeat(labels, per_face), 4)
            if island_of_loop2.shape[0] != h_loop.shape[0]:
                raise RuntimeError(
                    f"细分拓扑映射失配: {island_of_loop2.shape[0]:,} "
                    f"vs {h_loop.shape[0]:,}")
            h_loop = core.detrend_per_island(
                h_loop, uv2, island_of_loop2, n_islands, 'PLANE')
            h_loop = core.stitch_islands(
                h_loop, lv2, island_of_loop2, n_islands)
            h_loop *= w_loop
            h_vert = core.average_loops_to_verts(h_loop, lv2, vcount)
            solve_stat = f"UV Poisson | {n_islands} 岛"
        t_np1 = time.perf_counter()

        # ---- 建层(只为 Multires 数据结构; reshape 会完整覆写 MDISPS) ----
        # 求解成功后才创建/重建，确保后台取消不会损坏已有结果。层数已匹配则
        # 整段跳过；隐藏态细分面只作为 reshape 容器。
        if mod is None:
            mod = obj.modifiers.new("NormalMapToMesh", 'MULTIRES')
        if obj.modifiers.find(mod.name) != 0:
            bpy.ops.object.modifier_move_to_index(modifier=mod.name, index=0)
        if not (owned and mod.total_levels == level):
            mod.show_viewport = False
            try:
                if mod.total_levels > 0:
                    mod.levels = 0
                    mod.sculpt_levels = 0
                    bpy.ops.object.multires_higher_levels_delete(
                        modifier=mod.name)
                for _ in range(level):
                    bpy.ops.object.multires_subdivide(
                        modifier=mod.name, mode='CATMULL_CLARK')
            finally:
                mod.show_viewport = True
        mod.levels = level
        mod.sculpt_levels = level
        mod.render_levels = level
        if hasattr(mod, "uv_smooth"):
            mod.uv_smooth = 'NONE'
        t_subdiv = time.perf_counter()

        dvec = n0_vert * (h_vert * np.float32(s.disp_scale))[:, None]
        if boundary_verts.size:
            dvec[boundary_verts] = 0.0
        t_np2 = time.perf_counter()

        co += dvec
        tmp_me.vertices.foreach_set("co", co.ravel())
        tmp_me.update()
        mag = np.linalg.norm(dvec, axis=1)
        disp_stat = (f"{solve_stat} | 边界锁定 {boundary_verts.size:,} 顶点 | "
                     f"位移幅值 p50 {np.percentile(mag, 50) * 1000:.2f} / "
                     f"p95 {np.percentile(mag, 95) * 1000:.2f} / "
                     f"max {mag.max() * 1000:.2f} (千分之一物体单位)")
        t_displace = time.perf_counter()
        print(f"[NormalMapToMesh] 位移明细: 准备/基面 {t_eval - t_front:.1f}s"
              f" + 约束/求解 {t_np1 - t_eval:.1f}s"
              f" + 建层 {t_subdiv - t_np1:.1f}s"
              f" + 锁边 {t_np2 - t_subdiv:.1f}s"
              f" + 写坐标 {t_displace - t_np2:.1f}s")

        # ---- reshape 写回 Multires 位移层 ----
        tmp_obj = bpy.data.objects.new("NMTM_reshape_tmp", tmp_me)
        context.scene.collection.objects.link(tmp_obj)
        tmp_obj.matrix_world = obj.matrix_world.copy()
        for o in list(context.selected_objects):
            o.select_set(False)
        tmp_obj.select_set(True)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        bpy.ops.object.multires_reshape(modifier=mod.name)
    finally:
        if tmp_obj is not None:
            bpy.data.objects.remove(tmp_obj, do_unlink=True)
        if tmp_me is not None:
            try:
                bpy.data.meshes.remove(tmp_me)
            except Exception:
                pass
        for m, vis in saved_vis:
            m.show_viewport = vis

    # 高模细节按平滑着色观感正确
    obj.select_set(True)
    bpy.ops.object.shade_smooth()

    obj["nmtm_owned"] = 1
    obj["nmtm_level"] = level
    obj["nmtm_source"] = ("材质法线链" if source == 'MATERIAL'
                          else (s.image.name if s.image is not None else '?'))
    obj["nmtm_scale"] = float(s.disp_scale)
    obj["nmtm_solver"] = solver_label
    if joint_mode:
        obj["nmtm_joint_position_weight"] = joint_position_weight
        obj["nmtm_joint_irls_iters"] = joint_irls_iters
    else:
        for key in ("nmtm_joint_position_weight", "nmtm_joint_irls_iters"):
            if key in obj:
                del obj[key]

    t_end = time.perf_counter()
    msg = (f"{'材质' if source == 'MATERIAL' else '贴图'}求值 {bake_size}px | 级别 {level} | "
           f"{quads:,} 四边形 | {disp_stat} | "
           f"前端 {t_front - t0:.1f}s + 准备/求解 {t_np1 - t_front:.1f}s + "
           f"建层 {t_subdiv - t_np1:.1f}s + "
           f"位移 {t_displace - t_subdiv:.1f}s + 写回 {t_end - t_displace:.1f}s "
           f"= {t_end - t0:.1f}s")
    print(f"[NormalMapToMesh] {obj.name}: {msg}")
    report({'INFO'}, msg)


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

def _poll_mesh(context):
    obj = context.active_object
    return obj is not None and obj.type == 'MESH' and not obj.library


class NMTM_OT_build(bpy.types.Operator):
    """按面板设置构建/更新 Multires 细节(重复执行 = 从基面重建, 可反复调倍数)"""
    bl_idname = "nmtm.build"
    bl_label = "应用 / 更新"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _poll_mesh(context) and _active_joint_job is None

    def _start_worker(self, context, request):
        global _active_joint_job
        self._request = request
        self._object_name = context.active_object.name_full
        self._scene_name = context.scene.name_full
        self._cancel_event = threading.Event()
        self._cancel_requested = False
        self._progress_done = 0
        self._progress_total = (
            (max(0, int(request.keyword.get("irls_iters", 0))) + 1)
            * max(1, int(request.keyword.get("max_iter", 1)))
        )
        self._worker_state = {"done": False}
        self._area = context.area
        self._timer = context.window_manager.event_timer_add(
            0.1, window=context.window)
        context.window_manager.progress_begin(0, self._progress_total)
        context.window_manager.modal_handler_add(self)
        _active_joint_job = self

        def progress(done, total):
            self._progress_done = int(done)
            self._progress_total = int(total)

        def worker():
            keyword = dict(request.keyword)
            keyword["cancel_check"] = self._cancel_event.is_set
            keyword["progress_callback"] = progress
            try:
                self._worker_state["result"] = \
                    core.solve_joint_position_normal(
                        *request.positional, **keyword)
            except Exception as error:
                self._worker_state["error"] = error
            finally:
                self._worker_state["done"] = True

        self._thread = threading.Thread(
            target=worker,
            name="NormalMapToMesh-JointSolve",
            daemon=True,
        )
        self._thread.start()
        self.report(
            {'INFO'}, "联合优化已在后台求解；Blender 可继续操作，按 Esc 取消")
        return {'RUNNING_MODAL'}

    def _finish_modal_ui(self, context):
        global _active_joint_job
        timer = getattr(self, "_timer", None)
        if timer is not None:
            try:
                context.window_manager.event_timer_remove(timer)
            except Exception:
                pass
            self._timer = None
        try:
            context.window_manager.progress_end()
        except Exception:
            pass
        area = getattr(self, "_area", None)
        if area is not None:
            try:
                area.header_text_set(None)
            except Exception:
                pass
        if _active_joint_job is self:
            _active_joint_job = None

    def invoke(self, context, _event):
        # 后台/脚本执行保持同步、可复现；交互式联合模式把纯 NumPy PCG 放到
        # 工作线程。所有 bpy 读取、Multires 建层与 reshape 仍在主线程。
        if (getattr(context.scene.nmtm, "reconstruction_mode", 'POISSON')
                != 'JOINT' or context.window is None):
            return self.execute(context)
        try:
            build(
                context,
                context.active_object,
                context.scene.nmtm,
                self.report,
                defer_joint=True,
            )
        except _JointSolveRequest as request:
            return self._start_worker(context, request)
        except Exception as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        return {'FINISHED'}

    def modal(self, context, event):
        if event.type == 'ESC' and not self._cancel_requested:
            self._cancel_requested = True
            self._cancel_event.set()
            self.report({'INFO'}, "正在取消联合优化…")
            return {'RUNNING_MODAL'}

        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        done = min(self._progress_done, self._progress_total)
        try:
            context.window_manager.progress_update(done)
        except Exception:
            pass
        area = getattr(self, "_area", None)
        if area is not None:
            try:
                percent = 100.0 * done / max(self._progress_total, 1)
                area.header_text_set(
                    f"NormalMapToMesh 联合优化 {percent:.0f}%"
                    "（Esc 取消）")
            except Exception:
                pass

        if not self._worker_state.get("done", False):
            return {'RUNNING_MODAL'}

        self._finish_modal_ui(context)
        try:
            self._thread.join(timeout=0.0)
        except Exception:
            pass

        error = self._worker_state.get("error")
        if self._cancel_requested or isinstance(
                error, core.JointSolveCancelled):
            self._request = None
            self._worker_state = None
            self.report({'INFO'}, "联合优化已取消，模型未修改")
            return {'CANCELLED'}
        if error is not None:
            self._request = None
            self._worker_state = None
            self.report({'ERROR'}, f"联合优化失败: {error}")
            return {'CANCELLED'}

        result, stats = self._worker_state["result"]
        _cache_put(
            _joint_cache,
            self._request.cache_key,
            (result.copy(), dict(stats)),
        )
        obj = bpy.data.objects.get(self._object_name)
        if (obj is None or context.scene.name_full != self._scene_name
                or context.view_layer.objects.get(obj.name) is None):
            self._request = None
            self._worker_state = None
            self.report(
                {'ERROR'}, "后台求解完成，但原对象/场景已切换；结果未应用")
            return {'CANCELLED'}
        try:
            build(
                context,
                obj,
                context.scene.nmtm,
                self.report,
                expected_joint_cache_key=self._request.cache_key,
            )
        except Exception as build_error:
            self.report({'ERROR'}, str(build_error))
            return {'CANCELLED'}
        finally:
            self._request = None
            self._worker_state = None
        return {'FINISHED'}

    def cancel(self, context):
        cancel_event = getattr(self, "_cancel_event", None)
        if cancel_event is not None:
            self._cancel_requested = True
            cancel_event.set()
        self._finish_modal_ui(context)

    def execute(self, context):
        s = context.scene.nmtm
        obj = context.active_object
        try:
            build(context, obj, s, self.report)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        return {'FINISHED'}


class NMTM_OT_load_build(bpy.types.Operator, ImportHelper):
    """选择法线贴图文件, 加载后立即按贴图模式一键构建"""
    bl_idname = "nmtm.load_build"
    bl_label = "加载法线并一键构建"
    bl_options = {'REGISTER', 'UNDO'}

    filter_glob: bpy.props.StringProperty(
        default="*.png;*.jpg;*.jpeg;*.tga;*.tif;*.tiff;*.exr;*.bmp;*.webp;*.dds",
        options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return _poll_mesh(context) and _active_joint_job is None

    def execute(self, context):
        s = context.scene.nmtm
        try:
            img = bpy.data.images.load(self.filepath, check_existing=True)
        except Exception as e:
            self.report({'ERROR'}, f"加载贴图失败: {e}")
            return {'CANCELLED'}
        s.image = img
        s.source = 'IMAGE'
        if context.window is not None and s.reconstruction_mode == 'JOINT':
            return bpy.ops.nmtm.build('INVOKE_DEFAULT')
        return bpy.ops.nmtm.build()


class NMTM_OT_remove(bpy.types.Operator):
    """移除本工具生成的 Multires 细节与修改器, 恢复低模"""
    bl_idname = "nmtm.remove"
    bl_label = "移除细节"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (_active_joint_job is None
                and _poll_mesh(context) and bool(obj.get("nmtm_owned"))
                and _find_multires(obj) is not None)

    def execute(self, context):
        obj = context.active_object
        if context.view_layer.objects.active is not obj:
            context.view_layer.objects.active = obj
        if obj.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        mod = _find_multires(obj)
        if mod is not None:
            if mod.total_levels > 0:
                mod.levels = 0
                mod.sculpt_levels = 0
                bpy.ops.object.multires_higher_levels_delete(modifier=mod.name)
            bpy.ops.object.modifier_remove(modifier=mod.name)
        for k in ("nmtm_owned", "nmtm_level", "nmtm_image", "nmtm_source",
                  "nmtm_strength", "nmtm_scale", "nmtm_solver",
                  "nmtm_joint_position_weight", "nmtm_joint_irls_iters"):
            if k in obj.keys():
                del obj[k]
        self.report({'INFO'}, "已恢复低模")
        return {'FINISHED'}


CLASSES = (NMTM_OT_build, NMTM_OT_load_build, NMTM_OT_remove)
