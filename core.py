# -*- coding: utf-8 -*-
# NormalMapToMesh 数学内核 —— 纯 numpy, 不依赖 bpy, 可用 `python core.py` 独立自测。
#
# v6 不变式: 一切场量只由低模 + 法线贴图决定, 细分只是对同一连续曲面的采样密度
# ——任何细分方式/级别都必须输出同一张曲面(shader 逐像素求值的几何化)。
# 直算前端(ops 侧)光栅化 mikktspace 切线帧标量 → 逐 texel 高度梯度
#   dh/du = −(au·fx + bu·fy), fx = tx/tz (au=T·∂P/∂u 等为逐角解析标量),
# 镜像扩展 Neumann 最小二乘积分出物理高度 → 逐岛去趋势 → 岛间缝合 →
# 沿低模平滑角法线的线性插值场位移(与 shader 逐像素插值同构)。
# 烘焙路径(Cycles EMIT 三图: n1/n0/P, 同一套梯度代数)仅作不支持节点的回退。
#
# 历史教训:
# - v1 直接按"UV 轴=切线轴"解读贴图梯度做全局积分——真实资产逐岛切线帧任意旋转
#   +镜像混合手性, 前提性失败。
# - v2 差分位移(n1−n0)×scale——幅值与细节波长无关, 高频发丝沟壑位移超过顶点间距,
#   表面撕碎; 坡度必须先积分成高度(高频自动得到小高度)才物理正确。
# - v3~v5 周期 FFT 积分 + CC/PN 弯曲基面: wrap-around 让岛间隔着图集边界互相
#   泄漏低频; 弯曲基面/细分网格离散法线引入细分方式依赖——shader 从不弯曲基面,
#   位移场也从不该读细分网格自身的任何数据。

import hashlib

import numpy as np


class JointSolveCancelled(RuntimeError):
    """Raised when a caller cancels a long-running joint solve."""


def content_digest(*arrays, digest_size=16):
    """Return a deterministic digest over numeric array metadata and contents.

    The joint-solution cache must invalidate on any geometry, topology, UV, or
    gradient change.  ``hash()`` and sampled fingerprints are insufficient for
    that purpose, so cache keys use this complete byte-level digest instead.
    """
    digest = hashlib.blake2b(digest_size=int(digest_size))
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        if array.dtype.hasobject:
            raise TypeError("content_digest only supports non-object arrays")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray([array.ndim], dtype=np.int64).tobytes())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(array).cast("B"))
    return digest.digest()

# ---------------------------------------------------------------------------
# 采样与解码
# ---------------------------------------------------------------------------


def sample_bilinear_wrap(field, u, v):
    """按 UV 双线性采样; field 形状 (H, W) 或 (H, W, C); u/v 任意实数, 平铺 wrap。"""
    h, w = field.shape[:2]
    x = u * np.float32(w) - np.float32(0.5)
    y = v * np.float32(h) - np.float32(0.5)
    x0f = np.floor(x)
    y0f = np.floor(y)
    tx = (x - x0f).astype(np.float32)
    ty = (y - y0f).astype(np.float32)
    x0 = x0f.astype(np.int64) % w
    y0 = y0f.astype(np.int64) % h
    x1 = x0 + 1
    x1[x1 == w] = 0
    y1 = y0 + 1
    y1[y1 == h] = 0
    if field.ndim == 3:
        tx = tx[:, None]
        ty = ty[:, None]
    a = field[y0, x0]
    b = field[y0, x1]
    c = field[y1, x0]
    d = field[y1, x1]
    return (a * (1.0 - tx) + b * tx) * (1.0 - ty) + (c * (1.0 - tx) + d * tx) * ty


def sample_bspline_wrap(field, u, v):
    """三次均匀 B 样条采样(16 タップ, wrap); field (H,W) 或 (H,W,C)。

    位移曲面继承采样核的连续性: 双线性是 C0(texel 边界处导数跳变, 素模视图
    下呈颗粒/折面感), 三次 B 样条是 C2——这才是"渲染级光滑"的几何等价物。
    B 样条精确再现常量与线性场(细节形状不漂移), 高频略柔(≈附加 σ~0.5px,
    正好吃掉残余混叠)。
    """
    h, w = field.shape[:2]
    x = u * np.float32(w) - np.float32(0.5)
    y = v * np.float32(h) - np.float32(0.5)
    x0 = np.floor(x)
    y0 = np.floor(y)
    fx = (x - x0).astype(np.float32)
    fy = (y - y0).astype(np.float32)
    ix = x0.astype(np.int64)
    iy = y0.astype(np.int64)

    def bs_w(f):
        f2 = f * f
        f3 = f2 * f
        return ((1.0 - f) ** 3 / 6.0,
                (3.0 * f3 - 6.0 * f2 + 4.0) / 6.0,
                (-3.0 * f3 + 3.0 * f2 + 3.0 * f + 1.0) / 6.0,
                f3 / 6.0)

    wx = bs_w(fx)
    wy = bs_w(fy)
    multi = field.ndim == 3
    out = None
    for j in range(4):
        yj = (iy + (j - 1)) % h
        wyj = wy[j][:, None] if multi else wy[j]
        for i in range(4):
            xi = (ix + (i - 1)) % w
            wxi = wx[i][:, None] if multi else wx[i]
            contrib = field[yj, xi] * (wyj * wxi)
            out = contrib if out is None else out + contrib
    return out


def decode_unit_normal(rgb):
    """[0,1] 编码法线 → 单位向量 + 有效权重。

    权重 = 解码后长度接近 1 才为 1(烘焙背景黑 → (-1,-1,-1) 长度 1.73 → 0),
    防未烘焙 texel 的垃圾值污染。返回 (n 单位化, w)。
    """
    n = rgb.astype(np.float32) * 2.0 - 1.0
    ln = np.sqrt(np.einsum('...i,...i->...', n, n))
    w = (np.abs(ln - 1.0) < 0.35).astype(np.float32)
    n /= np.maximum(ln, 1e-6)[..., None]
    return n, w


# ---------------------------------------------------------------------------
# 高度梯度装配(零约定猜测)
# ---------------------------------------------------------------------------

def height_gradients(rgb_detail, rgb_base, pos, min_cos=0.2,
                     deadzone=0.0, slope_limit=0.0):
    """三张烘焙图 → UV 域高度梯度 (gx=dh/du, gy=dh/dv, 物体单位) + 有效权重。

    g = n0 − n1/(n1·n0): 高度场表面梯度的 3D 形式(切平面向量, |g| = tanθ);
    ∂P/∂u、∂P/∂v 用位置图中心差分——镜像岛的 U 轴自动反向, 混合手性零处理。
    dh/du = g·∂P/∂u 直接携带每 texel 的真实世界尺度(逐岛密度差异自动正确)。
    岛间沟槽处两侧 margin 相遇会产生巨大 |∂P| 假差分, 用稳健中位数阈值剔除。
    deadzone/slope_limit 语义同 gradients_from_tangent_frames(按 |g|=tanθ 幅值近似)。
    """
    n1, w1 = decode_unit_normal(rgb_detail)
    n0, w0 = decode_unit_normal(rgb_base)
    dot = np.einsum('...i,...i->...', n1, n0)
    w = w1 * w0 * (dot > min_cos).astype(np.float32)
    g = n0 - n1 / np.maximum(dot, min_cos)[..., None]
    if deadzone > 0.0 or slope_limit > 0.0:
        gm = np.sqrt(np.einsum('...i,...i->...', g, g))
        if deadzone > 0.0:
            g = np.where((gm <= deadzone)[..., None], 0.0, g)
        if slope_limit > 0.0:
            scale = np.minimum(1.0, slope_limit / np.maximum(gm, 1e-12))
            g = g * scale[..., None]

    hgt, wid = dot.shape
    pu = (np.roll(pos, -1, axis=1) - np.roll(pos, 1, axis=1)) * (wid / 2.0)
    pv = (np.roll(pos, -1, axis=0) - np.roll(pos, 1, axis=0)) * (hgt / 2.0)
    # 中心差分要求两侧邻 texel 也有效
    wu = w * np.roll(w, -1, axis=1) * np.roll(w, 1, axis=1)
    wv = w * np.roll(w, -1, axis=0) * np.roll(w, 1, axis=0)

    lu = np.sqrt(np.einsum('...i,...i->...', pu, pu))
    lv = np.sqrt(np.einsum('...i,...i->...', pv, pv))
    vu = lu[wu > 0.0]
    vv = lv[wv > 0.0]
    if vu.size:
        wu = wu * (lu < 16.0 * max(float(np.median(vu)), 1e-12)).astype(np.float32)
    if vv.size:
        wv = wv * (lv < 16.0 * max(float(np.median(vv)), 1e-12)).astype(np.float32)

    gx = (np.einsum('...i,...i->...', g, pu) * wu).astype(np.float32)
    gy = (np.einsum('...i,...i->...', g, pv) * wv).astype(np.float32)
    return gx, gy, w


# ---------------------------------------------------------------------------
# 直算前端: UV 三角形光栅化 + 切线空间梯度(免烘焙)
# ---------------------------------------------------------------------------

def rasterize_tris(tri_uv, tri_attr, size, accumulate=False):
    """UV 三角形光栅化: 逐 texel 重心插值角属性。

    tri_uv (T,3,2) float32, tri_attr (T,3,C) float32。
    accumulate=False: 重叠 texel 后写覆盖 → (grid (S,S,C), mask (S,S))。
    accumulate=True: 重叠 texel 累加平均的分子/分母 → (sum (S,S,C), count (S,S))
    ——多张卡片共享同一贴图区域时, 覆盖会产生"赢家马赛克"(逐三角形补丁的
    属性跳变, 积分后成锯齿), 平均则得到平滑的折中场。
    texel 中心 ((x+0.5)/S, (y+0.5)/S); 按 bbox 尺寸分桶批量向量化; 退化三角形跳过。
    """
    n_ch = tri_attr.shape[2]
    grid = np.zeros((size, size, n_ch), np.float32)
    if accumulate:
        cnt = np.zeros(size * size, np.float64)
        acc = np.zeros((size * size, n_ch), np.float64)
    mask = np.zeros((size, size), bool)
    if tri_uv.shape[0] == 0:
        return (grid, mask) if not accumulate else (grid, np.zeros((size, size), np.float32))
    gflat = grid.reshape(-1, n_ch)
    mflat = mask.reshape(-1)

    uv = tri_uv.astype(np.float64)
    a, b, c = uv[:, 0], uv[:, 1], uv[:, 2]
    e1 = b - a
    e2 = c - a
    den = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    ok = np.abs(den) > 1e-14

    px_min = np.minimum(np.minimum(a, b), c) * size - 0.5
    px_max = np.maximum(np.maximum(a, b), c) * size - 0.5
    x0 = np.clip(np.ceil(px_min[:, 0]), 0, size - 1).astype(np.int64)
    y0 = np.clip(np.ceil(px_min[:, 1]), 0, size - 1).astype(np.int64)
    x1 = np.clip(np.floor(px_max[:, 0]), 0, size - 1).astype(np.int64)
    y1 = np.clip(np.floor(px_max[:, 1]), 0, size - 1).astype(np.int64)
    side = np.maximum(x1 - x0 + 1, y1 - y0 + 1)
    ok &= (x1 >= x0) & (y1 >= y0)

    order = np.argsort(side, kind='stable')
    eps = 1e-6
    for k in (4, 8, 16, 32, 64, 128, 256, 1 << 30):
        sel = order[ok[order] & (side[order] <= k)]
        if sel.size == 0:
            continue
        order = order[~np.isin(order, sel, assume_unique=True)]
        kk = int(min(k, size))
        # 控制单块内存: n*kk*kk ≤ ~4M texel
        chunk = max(1, (4_000_000 // (kk * kk)))
        for s0 in range(0, sel.size, chunk):
            t = sel[s0:s0 + chunk]
            n = t.size
            xs = x0[t][:, None] + np.arange(kk)[None, :]           # (n,kk)
            ys = y0[t][:, None] + np.arange(kk)[None, :]
            vx = xs <= x1[t][:, None]
            vy = ys <= y1[t][:, None]
            u = (xs.astype(np.float64) + 0.5) / size
            v = (ys.astype(np.float64) + 0.5) / size
            pu = u[:, None, :] - a[t, 0][:, None, None]            # (n,kk,kk)
            pv = v[:, :, None] - a[t, 1][:, None, None]
            d = den[t][:, None, None]
            l1 = (pu * e2[t, 1][:, None, None] - pv * e2[t, 0][:, None, None]) / d
            l2 = (e1[t, 0][:, None, None] * pv - e1[t, 1][:, None, None] * pu) / d
            l0 = 1.0 - l1 - l2
            inside = ((l0 >= -eps) & (l1 >= -eps) & (l2 >= -eps)
                      & vx[:, None, :] & vy[:, :, None])
            if not inside.any():
                continue
            attr = tri_attr[t]                                     # (n,3,C)
            vals = (l0[..., None] * attr[:, 0][:, None, None, :]
                    + l1[..., None] * attr[:, 1][:, None, None, :]
                    + l2[..., None] * attr[:, 2][:, None, None, :])
            gi = ys[:, :, None] * size + xs[:, None, :]            # (n,kk,kk)
            idx = gi[inside]
            if accumulate:
                vin = vals[inside]
                for c in range(n_ch):
                    acc[:, c] += np.bincount(idx, weights=vin[:, c], minlength=size * size)
                cnt += np.bincount(idx, minlength=size * size)
            else:
                gflat[idx] = vals[inside].astype(np.float32)
                mflat[idx] = True
        if order.size == 0:
            break
    if accumulate:
        grid = acc.astype(np.float32).reshape(size, size, n_ch)
        return grid, cnt.astype(np.float32).reshape(size, size)
    return grid, mask


def dilate_grid(grid, mask, iters):
    """有效区向外扩 iters 圈(4邻域均值填充), 作用等价烘焙 margin:
    岛边界的双线性采样不吃到无效 texel。返回 (grid, mask) 新数组。"""
    g = grid.copy()
    m = mask.copy()
    for _ in range(iters):
        if m.all():
            break
        nb_sum = np.zeros_like(g)
        nb_cnt = np.zeros(m.shape, np.float32)
        for sh, ax in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
            ms = np.roll(m, sh, axis=ax)
            gs = np.roll(g, sh, axis=ax)
            w = ms.astype(np.float32)
            nb_sum += gs * w[..., None]
            nb_cnt += w
        fill = (~m) & (nb_cnt > 0)
        g[fill] = nb_sum[fill] / nb_cnt[fill][:, None]
        m = m | fill
    return g, m


def gradients_from_frame_scalars(t_xyz, au, bu, av, bv, mask, min_cos=0.2,
                                 deadzone=0.0, slope_limit=0.0):
    """切线空间法线图 + 光栅化切线帧标量 → UV 高度梯度(免烘焙直算)。

    与烘焙路径同一代数(逐 texel): g = −(T·tx + B·ty)/max(tz, min_cos·|t|),
    dh/du = g·∂P/∂u = −(au·fx + bu·fy), 其中 au=T·∂P/∂u, bu=B·∂P/∂u,
    av=T·∂P/∂v, bv=B·∂P/∂v 为逐角解析标量(可跨重叠卡片逐 texel 平均——
    标量平均 = 各卡片高度函数的世界尺度折中, 平滑无马赛克)。
    |t| 异常(未覆盖/坏像素)或 cos ≤ min_cos 的 texel 权重归零。
    deadzone: |tx|/|ty| ≤ 该值(归一化)视为纯平——8bit 量化噪声经积分会放大
    成低频起伏/斑点, 死区让平坦区严格为平。
    slope_limit: 坡度幅值(tanθ)限幅, 压制噪声/压缩伪影的尖刺。
    """
    tx = t_xyz[..., 0]
    ty = t_xyz[..., 1]
    tz = t_xyz[..., 2]
    ln = np.sqrt(tx * tx + ty * ty + tz * tz)
    cosw = tz / np.maximum(ln, 1e-12)
    w = (mask & (ln > 0.25) & (ln < 2.25) & (cosw > min_cos)).astype(np.float32)

    denom = np.maximum(tz, np.float32(min_cos) * ln)
    denom = np.maximum(denom, 1e-12)
    fx = tx / denom
    fy = ty / denom
    if deadzone > 0.0:
        lm = np.maximum(ln, 1e-12)
        fx = np.where(np.abs(tx) <= deadzone * lm, 0.0, fx)
        fy = np.where(np.abs(ty) <= deadzone * lm, 0.0, fy)
    if slope_limit > 0.0:
        mag = np.hypot(fx, fy)
        scale = np.minimum(1.0, slope_limit / np.maximum(mag, 1e-12))
        fx = fx * scale
        fy = fy * scale
    gx = -(au * fx + bu * fy) * w
    gy = -(av * fx + bv * fy) * w
    return gx.astype(np.float32), gy.astype(np.float32), w


def dilate_mask(mask, iters):
    """布尔掩码 4 邻域膨胀 iters 圈(岛边界采样安全余量)。"""
    m = mask.copy()
    for _ in range(iters):
        if m.all():
            break
        m = (m | np.roll(m, 1, 0) | np.roll(m, -1, 0)
             | np.roll(m, 1, 1) | np.roll(m, -1, 1))
    return m


def edge_falloff_field(seg_uv, size, radius_px):
    """开放边界 UV 线段 → 距离衰减场(边界=0, ≥radius=1), 乘在高度场上。

    高度沿开放边(卡片边缘)平滑归零防撕缝。场量定义在 UV 域(低模属性),
    与细分方式/级别无关。逐线段按像素密度采样打点 → 4 邻域 BFS 传播整数
    距离 → smoothstep(C1) → 2 轮 Jacobi 平滑抹掉整数环带。np.roll 环绕
    与平铺采样语义一致。seg_uv (E,2,2)。
    """
    if seg_uv.shape[0] == 0 or radius_px <= 0:
        return np.ones((size, size), np.float32)
    a = seg_uv[:, 0].astype(np.float64) * size
    b = seg_uv[:, 1].astype(np.float64) * size
    ln = np.maximum(np.abs(b - a).max(axis=1), 1e-9)
    n = np.minimum(np.ceil(ln).astype(np.int64) + 1, 4 * size)
    total = int(n.sum())
    seg_id = np.repeat(np.arange(seg_uv.shape[0]), n)
    ofs = np.concatenate([[0], np.cumsum(n)[:-1]])
    t = (np.arange(total) - ofs[seg_id]) / np.maximum(n[seg_id] - 1, 1)
    p = a[seg_id] * (1.0 - t[:, None]) + b[seg_id] * t[:, None]
    xi = np.floor(p[:, 0]).astype(np.int64) % size
    yi = np.floor(p[:, 1]).astype(np.int64) % size
    r = int(radius_px)
    dist = np.full((size, size), r + 1, np.int16)
    dist[yi, xi] = 0
    for _ in range(r):
        m = np.minimum(dist, np.roll(dist, 1, 0) + 1)
        m = np.minimum(m, np.roll(dist, -1, 0) + 1)
        m = np.minimum(m, np.roll(dist, 1, 1) + 1)
        m = np.minimum(m, np.roll(dist, -1, 1) + 1)
        if np.array_equal(m, dist):
            break
        dist = m
    s = np.clip(dist.astype(np.float32) / np.float32(r + 1), 0.0, 1.0)
    s = s * s * (3.0 - 2.0 * s)
    for _ in range(2):
        s = 0.5 * s + 0.125 * (np.roll(s, 1, 0) + np.roll(s, -1, 0)
                               + np.roll(s, 1, 1) + np.roll(s, -1, 1))
    return s


# ---------------------------------------------------------------------------
# Neumann 最小二乘泊松积分(半样本镜像 ≡ DCT)
# ---------------------------------------------------------------------------

def integrate_height(gx, gy, smooth_sigma=0.0):
    """Neumann 最小二乘可积化: 给定梯度场求高度场 (平地锚定)。

    半样本镜像扩展到 2H×2W 后做周期 Frankot-Chellappa: gx 沿 x 反对称/沿 y
    对称, gy 相反——对称数据的周期最小二乘解自动继承偶对称, 取原象限即
    Neumann 边界解。消除周期 FFT 的 wrap-around: 斜坡/非周期内容精确还原,
    岛间不再经图集边界环绕互相泄漏低频(旧高通压泄漏的拐杖随之废除)。
    梯度为 dh/du(物体单位/UV), 返回物理高度。4K 图瞬时内存 ~1.5GB(complex128)。
    smooth_sigma: >0 时高斯低通(σ, UV单位)——级别匹配重建滤波 ⊕ 源噪声地板。
    """
    h, w = gx.shape
    gx2 = np.empty((2 * h, 2 * w), np.float32)
    gy2 = np.empty((2 * h, 2 * w), np.float32)
    gx2[:h, :w] = gx
    gx2[:h, w:] = -gx[:, ::-1]
    gx2[h:] = gx2[:h][::-1]
    gy2[:h, :w] = gy
    gy2[:h, w:] = gy[:, ::-1]
    gy2[h:] = -gy2[:h][::-1]
    wx = (2.0 * np.pi) * np.fft.rfftfreq(2 * w, d=1.0 / w)   # rad / UV单位
    wy = (2.0 * np.pi) * np.fft.fftfreq(2 * h, d=1.0 / h)
    gx_f = np.fft.rfft2(gx2)
    del gx2
    gy_f = np.fft.rfft2(gy2)
    del gy2
    denom = wx[None, :] ** 2 + wy[:, None] ** 2
    denom[0, 0] = 1.0
    hf = (wx[None, :] * gx_f + wy[:, None] * gy_f) * (-1j)
    del gx_f, gy_f
    hf /= denom
    hf[0, 0] = 0.0
    if smooth_sigma > 0.0:
        k_sq = wx[None, :] ** 2 + wy[:, None] ** 2
        hf *= np.exp(-0.5 * smooth_sigma * smooth_sigma * k_sq)
    out = np.fft.irfft2(hf, s=(2 * h, 2 * w))[:h, :w].astype(np.float32)
    # 基准面锚定: 平坦区(零梯度, 含未烘焙背景)应为 0 高度
    flat = (gx == 0.0) & (gy == 0.0)
    if flat.mean() > 0.01:
        out -= np.float32(out[flat].mean())
    else:
        out -= np.float32(np.median(out))
    return out


# ---------------------------------------------------------------------------
# 位置 + 法线联合优化（实验）
# ---------------------------------------------------------------------------


def smooth_fields_neumann(fields, smooth_sigma):
    """自由边界高斯低通；fields 为 (H,W) 或 (H,W,C)，sigma 使用 UV 单位。

    每个标量分量作半样本偶镜像后 FFT 滤波，等价于自由边界上的 DCT 高斯滤波。
    与 ``integrate_height`` 中梯度的奇/偶镜像不同：这里平滑的是已经装配好的标量
    梯度分量，偶镜像可保持常量坡度，不会在图集边缘人为压向零。
    """
    src = np.asarray(fields, dtype=np.float32)
    if smooth_sigma <= 0.0:
        return src.copy()
    squeeze = src.ndim == 2
    if squeeze:
        src = src[..., None]
    if src.ndim != 3:
        raise ValueError("fields 必须是 (H,W) 或 (H,W,C)")

    h, w, channels = src.shape
    wx = (2.0 * np.pi) * np.fft.rfftfreq(2 * w, d=1.0 / w)
    wy = (2.0 * np.pi) * np.fft.fftfreq(2 * h, d=1.0 / h)
    filt = np.exp(-0.5 * float(smooth_sigma) ** 2
                  * (wx[None, :] ** 2 + wy[:, None] ** 2))
    out = np.empty_like(src)
    # 逐通道处理，避免同时保留 C 份 complex128 频谱。
    for channel in range(channels):
        ext = np.empty((2 * h, 2 * w), np.float32)
        ext[:h, :w] = src[..., channel]
        ext[:h, w:] = src[:, ::-1, channel]
        ext[h:] = ext[:h][::-1]
        spec = np.fft.rfft2(ext)
        del ext
        spec *= filt
        out[..., channel] = np.fft.irfft2(spec, s=(2 * h, 2 * w))[:h, :w]
    return out[..., 0] if squeeze else out


def _merge_duplicate_edge_observations(vert_i, vert_j, target, weight,
                                       uv_i, uv_j, edge_id):
    """精确合并普通流形边的双面重复观测，UV 缝观测保持独立。

    细分面逐 loop 装配时，普通内部边会从相邻两面各出现一次。两条观测在统一
    到 ``min(vertex) -> max(vertex)`` 的方向后，若 UV 端点与目标值完全相同，
    它们在最小二乘和 Huber IRLS 中是同一残差的重复项，可把权重相加而不改变
    目标函数。UV 缝、非流形边和任何不完全相同的观测都保留原样。
    """
    vi = np.asarray(vert_i, dtype=np.int64).ravel()
    vj = np.asarray(vert_j, dtype=np.int64).ravel()
    gd = np.asarray(target, dtype=np.float32).ravel()
    bw = np.asarray(weight, dtype=np.float32).ravel()
    ui = np.asarray(uv_i, dtype=np.float32)
    uj = np.asarray(uv_j, dtype=np.float32)
    eid = np.asarray(edge_id, dtype=np.int64).ravel()
    if not (vi.size == vj.size == gd.size == bw.size == eid.size):
        raise ValueError("重复边观测数组长度不一致")
    if vi.size < 2:
        return vi, vj, gd, bw

    swap = vi > vj
    low = np.minimum(vi, vj)
    high = np.maximum(vi, vj)
    canonical_target = np.where(swap, -gd, gd).astype(np.float32, copy=False)
    valid_edge_id = eid >= 0
    if not np.any(valid_edge_id):
        return low, high, canonical_target, bw

    edge_count = int(eid[valid_edge_id].max()) + 1
    counts = np.bincount(eid[valid_edge_id], minlength=edge_count)
    pair_edges = np.flatnonzero(counts == 2)
    if pair_edges.size == 0:
        return low, high, canonical_target, bw

    # Blender 已提供 loop.edge_index；用 O(E) 的 first/last 归并，不对百万级
    # 约束做 O(E log E) 全量排序。int32 足以容纳 Blender 的 loop 数并减少峰值内存。
    positions = np.arange(vi.size, dtype=np.int32)
    first = np.full(edge_count, vi.size, dtype=np.int32)
    last = np.full(edge_count, -1, dtype=np.int32)
    np.minimum.at(first, eid[valid_edge_id], positions[valid_edge_id])
    np.maximum.at(last, eid[valid_edge_id], positions[valid_edge_id])
    pair_first = first[pair_edges].astype(np.int64, copy=False)
    pair_second = last[pair_edges].astype(np.int64, copy=False)

    first_low_uv = np.where(
        swap[pair_first, None], uj[pair_first], ui[pair_first])
    first_high_uv = np.where(
        swap[pair_first, None], ui[pair_first], uj[pair_first])
    second_low_uv = np.where(
        swap[pair_second, None], uj[pair_second], ui[pair_second])
    second_high_uv = np.where(
        swap[pair_second, None], ui[pair_second], uj[pair_second])
    same_uv = (
        np.all(first_low_uv == second_low_uv, axis=1)
        & np.all(first_high_uv == second_high_uv, axis=1)
    )
    same_target = (
        canonical_target[pair_first] == canonical_target[pair_second]
    )
    merge_first = pair_first[same_uv & same_target]
    merge_second = pair_second[same_uv & same_target]
    if merge_first.size == 0:
        return low, high, canonical_target, bw

    keep = np.ones(low.size, dtype=bool)
    bw[merge_first] += bw[merge_second]
    keep[merge_second] = False
    return (low[keep], high[keep], canonical_target[keep], bw[keep])


def gradient_constraints_from_loops(loop_vert, loop_uv, loop_start, loop_total,
                                    gx, gy, weight_map, min_weight=0.05,
                                    merge_duplicates=True, loop_edge=None):
    """把法线导出的 UV 梯度变成真实网格拓扑上的边位移观测。

    对每个面内有向边 ``i→j``，在 UV 中点采样 ``dh/du, dh/dv``，得到
    ``d_j-d_i ≈ gx·Δu + gy·Δv``。同一几何边在 UV 缝两侧会保留两条独立观测，
    但二者共享同一对顶点未知量；UV 因此只负责采样，不再定义几何连通性。
    """
    lv = np.asarray(loop_vert, dtype=np.int64)
    uv = np.asarray(loop_uv, dtype=np.float32)
    starts = np.asarray(loop_start, dtype=np.int64)
    totals = np.asarray(loop_total, dtype=np.int64)
    if lv.ndim != 1 or uv.shape != (lv.size, 2):
        raise ValueError("loop_vert/loop_uv 形状不匹配")
    if starts.shape != totals.shape:
        raise ValueError("loop_start/loop_total 形状不匹配")
    if lv.size == 0:
        empty_i = np.zeros(0, np.int64)
        empty_f = np.zeros(0, np.float32)
        return empty_i, empty_i.copy(), empty_f, empty_f.copy()

    nxt = np.arange(lv.size, dtype=np.int64) + 1
    last = starts + totals - 1
    nxt[last] = starts
    vi = lv
    vj = lv[nxt]
    duv = uv[nxt].astype(np.float64) - uv.astype(np.float64)
    mid = (uv[nxt].astype(np.float64) + uv.astype(np.float64)) * 0.5

    packed = np.stack([gx, gy, weight_map], axis=-1).astype(np.float32)
    obs = sample_bspline_wrap(packed, mid[:, 0].astype(np.float32),
                              mid[:, 1].astype(np.float32))
    target = obs[:, 0].astype(np.float64) * duv[:, 0] \
        + obs[:, 1].astype(np.float64) * duv[:, 1]
    weight = np.clip(obs[:, 2].astype(np.float64), 0.0, 1.0)
    valid = ((vi != vj) & np.isfinite(target) & np.isfinite(weight)
             & (weight >= float(min_weight)))
    vi_valid = vi[valid].astype(np.int64)
    vj_valid = vj[valid].astype(np.int64)
    target_valid = target[valid].astype(np.float32)
    weight_valid = weight[valid].astype(np.float32)
    if not merge_duplicates or loop_edge is None:
        return vi_valid, vj_valid, target_valid, weight_valid
    edge_id = np.asarray(loop_edge, dtype=np.int64).ravel()
    if edge_id.size != lv.size:
        raise ValueError("loop_edge 长度与 loop 数不一致")
    return _merge_duplicate_edge_observations(
        vi_valid, vj_valid, target_valid, weight_valid,
        uv[valid], uv[nxt][valid], edge_id[valid])


def solve_joint_position_normal(edge_i, edge_j, target_delta, vert_count,
                                base_weight=None, position_weight=0.1,
                                position_measure=None, pinned=None, prior=None,
                                irls_iters=3,
                                max_iter=400, tolerance=1e-5,
                                cancel_check=None, progress_callback=None):
    """位置先验 + 法线梯度约束的矩阵自由标量位移优化。

    求解::

        Σ_e w_e ρ((d_j-d_i)-g_e) + λ Σ_i (d_i-d_prior_i)^2

    ``g_e`` 来自法线贴图梯度沿真实网格边 UV 方向的线积分。固定 IRLS 权重时，
    系统是带位置屏蔽项的加权图拉普拉斯；使用纯 NumPy PCG，不依赖 SciPy。
    ``position_measure`` 可为每条边提供归一化后的局部面积尺度（通常取细分边长
    平方/物体包围盒对角线平方）；位置项使用顶点相邻边尺度的加权均值，因此增加
    细分级别不会让同一个 ``position_weight`` 越来越强。未提供时保留无量纲图权重。
    ``pinned`` 顶点从未知量中消去（通常为真实开放边界），不会把 UV 缝当边界。
    ``cancel_check`` 与 ``progress_callback`` 只接触 Python/NumPy 数据，可安全由
    Blender 的后台工作线程使用；回调不得访问 bpy。
    """
    outer_count = max(0, int(irls_iters)) + 1
    inner_budget = max(1, int(max_iter))
    progress_total = outer_count * inner_budget

    def check_cancelled():
        if cancel_check is not None and cancel_check():
            raise JointSolveCancelled("联合优化已取消")

    def report_progress(done):
        if progress_callback is not None:
            progress_callback(int(done), int(progress_total))

    check_cancelled()
    n_vert = int(vert_count)
    if n_vert < 0:
        raise ValueError("vert_count 不能为负")
    ei = np.asarray(edge_i, dtype=np.int64).ravel()
    ej = np.asarray(edge_j, dtype=np.int64).ravel()
    gd = np.asarray(target_delta, dtype=np.float64).ravel()
    if not (ei.size == ej.size == gd.size):
        raise ValueError("边约束数组长度不一致")
    if base_weight is None:
        bw = np.ones(ei.size, np.float64)
    else:
        bw = np.asarray(base_weight, dtype=np.float64).ravel()
        if bw.size != ei.size:
            raise ValueError("base_weight 长度不一致")
    if position_measure is None:
        pm = None
    else:
        pm = np.asarray(position_measure, dtype=np.float64).ravel()
        if pm.size != ei.size:
            raise ValueError("position_measure 长度不一致")
    if pinned is None:
        pin = np.zeros(n_vert, bool)
    else:
        pin = np.asarray(pinned, dtype=bool).ravel()
        if pin.size != n_vert:
            raise ValueError("pinned 长度不一致")
    if prior is None:
        prior_all = np.zeros(n_vert, np.float64)
    else:
        prior_all = np.asarray(prior, dtype=np.float64).ravel()
        if prior_all.size != n_vert:
            raise ValueError("prior 长度不一致")

    valid = ((ei >= 0) & (ei < n_vert) & (ej >= 0) & (ej < n_vert)
             & (ei != ej) & np.isfinite(gd) & np.isfinite(bw) & (bw > 0.0))
    if pm is not None:
        valid &= np.isfinite(pm) & (pm > 0.0)
    ei, ej, gd, bw = ei[valid], ej[valid], gd[valid], bw[valid]
    if pm is not None:
        pm = pm[valid]
    free_verts = np.flatnonzero(~pin)
    if free_verts.size == 0:
        report_progress(progress_total)
        return prior_all.astype(np.float32), {
            "edge_count": int(ei.size), "free_count": 0, "pcg_iterations": 0,
            "irls_updates": 0, "converged": True, "residual_rms": 0.0,
            "residual_p95": 0.0, "downweighted_fraction": 0.0,
        }
    free_map = np.full(n_vert, -1, np.int64)
    free_map[free_verts] = np.arange(free_verts.size, dtype=np.int64)
    fi = free_map[ei]
    fj = free_map[ej]
    keep = (fi >= 0) | (fj >= 0)
    fi, fj, gd, bw = fi[keep], fj[keep], gd[keep], bw[keep]
    if pm is not None:
        pm = pm[keep]
    if gd.size == 0:
        report_progress(progress_total)
        return prior_all.astype(np.float32), {
            "edge_count": 0, "free_count": int(free_verts.size), "pcg_iterations": 0,
            "irls_updates": 0, "converged": True, "residual_rms": 0.0,
            "residual_p95": 0.0, "downweighted_fraction": 0.0,
        }

    has_i = fi >= 0
    has_j = fj >= 0
    n_free = free_verts.size
    # 固定端点可能有非零 prior。把它对边差的常量贡献移到观测右端，使通用 API
    # 与当前常用的零边界固定都严格成立。
    pinned_difference = np.zeros(gd.size, np.float64)
    pin_i = ~has_i
    pin_j = ~has_j
    if np.any(pin_i):
        pinned_difference[pin_i] -= prior_all[ei[keep][pin_i]]
    if np.any(pin_j):
        pinned_difference[pin_j] += prior_all[ej[keep][pin_j]]
    effective_target = gd - pinned_difference
    # 用一个固定为零的哑元表示 pinned 端点。相比每次 SpMV 都布尔筛选并复制
    # 约 200 万条边的 values，此布局只做连续 gather/bincount，显著降低 PCG
    # 内层的临时数组与内存带宽。
    dummy = n_free
    fi_safe = np.where(has_i, fi, dummy)
    fj_safe = np.where(has_j, fj, dummy)
    extended = np.empty(n_free + 1, np.float64)

    def difference(x):
        extended[:-1] = x
        extended[-1] = 0.0
        out = extended[fj_safe]
        out -= extended[fi_safe]
        return out

    def transpose(values):
        out = np.bincount(fj_safe, weights=values, minlength=n_free + 1)
        out -= np.bincount(fi_safe, weights=values, minlength=n_free + 1)
        return out[:-1]

    degree0_all = np.bincount(fi_safe, weights=bw, minlength=n_free + 1)
    degree0_all += np.bincount(fj_safe, weights=bw, minlength=n_free + 1)
    degree0 = degree0_all[:-1]
    positive_degree = degree0[degree0 > 0.0]
    degree_scale = float(np.median(positive_degree)) if positive_degree.size else 1.0
    if pm is None:
        position_mass = np.full(n_free, degree_scale, np.float64)
    else:
        mass_weight = bw * pm
        mass_sum_all = np.bincount(
            fi_safe, weights=mass_weight, minlength=n_free + 1)
        mass_sum_all += np.bincount(
            fj_safe, weights=mass_weight, minlength=n_free + 1)
        mass_sum = mass_sum_all[:-1]
        position_mass = mass_sum / np.maximum(degree0, 1e-30)
        positive_mass = position_mass[position_mass > 0.0]
        fallback_mass = float(np.median(positive_mass)) if positive_mass.size else 1.0
        position_mass = np.where(position_mass > 0.0, position_mass, fallback_mass)
    screen = max(float(position_weight), 0.0) * position_mass
    numerical_screen = max(degree_scale, 1.0) * 1e-12
    total_screen = screen + numerical_screen
    prior_free = prior_all[free_verts]

    x = prior_free.copy()
    robust = np.ones(gd.size, np.float64)
    total_pcg = 0
    converged = False
    updates_done = 0
    final_residual = difference(x) - effective_target

    for outer in range(outer_count):
        check_cancelled()
        weights = bw * robust
        degree_all = np.bincount(
            fi_safe, weights=weights, minlength=n_free + 1)
        degree_all += np.bincount(
            fj_safe, weights=weights, minlength=n_free + 1)
        degree = degree_all[:-1]
        diagonal = np.maximum(degree + total_screen, 1e-20)
        rhs = transpose(weights * effective_target) + total_screen * prior_free

        def apply(value):
            return transpose(weights * difference(value)) + total_screen * value

        residual = rhs - apply(x)
        rhs_norm = max(float(np.linalg.norm(rhs)), 1.0)
        z = residual / diagonal
        direction = z.copy()
        rz = float(np.dot(residual, z))
        converged = float(np.linalg.norm(residual)) <= float(tolerance) * rhs_norm
        pcg_used = 0
        for iteration in range(inner_budget):
            if iteration % 8 == 0:
                check_cancelled()
                report_progress(outer * inner_budget + iteration)
            if converged or rz <= 0.0:
                break
            ad = apply(direction)
            denom = float(np.dot(direction, ad))
            if not np.isfinite(denom) or denom <= 1e-30:
                break
            alpha = rz / denom
            x += alpha * direction
            residual -= alpha * ad
            pcg_used = iteration + 1
            if float(np.linalg.norm(residual)) <= float(tolerance) * rhs_norm:
                converged = True
                break
            z = residual / diagonal
            rz_new = float(np.dot(residual, z))
            if not np.isfinite(rz_new) or rz_new <= 0.0:
                break
            direction = z + (rz_new / rz) * direction
            rz = rz_new
        total_pcg += pcg_used
        report_progress((outer + 1) * inner_budget)
        final_residual = difference(x) - effective_target

        if outer >= outer_count - 1:
            break
        abs_residual = np.abs(final_residual)
        sigma = 1.4826 * float(np.median(abs_residual))
        reference = max(float(np.median(np.abs(effective_target))), 1e-12)
        if not np.isfinite(sigma) or sigma <= reference * 1e-8:
            break
        delta = 1.345 * sigma
        new_robust = np.minimum(1.0, delta / np.maximum(abs_residual, 1e-30))
        updates_done += 1
        if np.max(np.abs(new_robust - robust)) < 1e-3:
            robust = new_robust
            break
        robust = new_robust

    result = prior_all.copy()
    result[free_verts] = x
    result[pin] = prior_all[pin]
    abs_final = np.abs(final_residual)
    stats = {
        "edge_count": int(gd.size),
        "free_count": int(n_free),
        "pcg_iterations": int(total_pcg),
        "irls_updates": int(updates_done),
        "converged": bool(converged),
        "residual_rms": float(np.sqrt(np.mean(final_residual ** 2))),
        "residual_p95": float(np.percentile(abs_final, 95)),
        "downweighted_fraction": float(np.mean(robust < 0.999)),
        "position_screen_median": float(np.median(screen)),
    }
    report_progress(progress_total)
    return result.astype(np.float32), stats


# ---------------------------------------------------------------------------
# UV 岛: 面级并查集 + 逐岛去趋势 + 岛间缝合 (v1 机器, 实测自洽)
# ---------------------------------------------------------------------------

def face_islands(loop_vert, loop_uv, poly_of_loop, poly_count):
    """按 (顶点, 量化UV) 归并面 → UV 岛标签。

    共享同一顶点且 UV 重合的两个 loop 所属的面判为同岛(覆盖共边与顶点粘连)。
    返回 (labels[poly_count] int32, 岛数)。
    """
    qu = np.round(loop_uv[:, 0] * 65536.0).astype(np.int64)
    qv = np.round(loop_uv[:, 1] * 65536.0).astype(np.int64)
    lv = loop_vert.astype(np.int64)
    order = np.lexsort((qv, qu, lv))
    lv_s, qu_s, qv_s = lv[order], qu[order], qv[order]
    pp = poly_of_loop[order]
    same = (lv_s[1:] == lv_s[:-1]) & (qu_s[1:] == qu_s[:-1]) & (qv_s[1:] == qv_s[:-1])

    parent = np.arange(poly_count, dtype=np.int64)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]  # 路径减半
            a = parent[a]
        return a

    for i in np.nonzero(same)[0]:
        ra, rb = find(pp[i]), find(pp[i + 1])
        if ra != rb:
            parent[rb] = ra

    roots = np.array([find(p) for p in range(poly_count)], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int32), int(labels.max()) + 1 if poly_count else 0


def detrend_per_island(h_loop, uv_loop, island_of_loop, n_islands, mode='PLANE'):
    """逐 UV 岛去趋势: 消除岛间高度台阶与岛内积分残留的斜坡。

    全局 FFT 积分会让每个岛带上任意常量偏移(MEAN 消除)和跨岛泄漏的线性斜坡
    (PLANE 消除)。island_of_loop 来自基面拓扑映射, 精确且不怕 UV 重叠。
    PLANE 用逐岛中心化最小二乘平面拟合, 全向量化。
    """
    if n_islands < 1 or mode == 'OFF':
        return h_loop
    isl = island_of_loop
    cnt = np.bincount(isl, minlength=n_islands).astype(np.float64)
    cnt_safe = np.maximum(cnt, 1.0)
    h64 = h_loop.astype(np.float64)
    mean_h = np.bincount(isl, weights=h64, minlength=n_islands) / cnt_safe
    if mode == 'MEAN':
        med = _segment_median(h64, isl, n_islands, cnt)
        return (h64 - med[isl]).astype(np.float32)

    # PLANE: h ≈ a·u + b·v + c, 逐岛中心化后 2x2 正规方程
    u = uv_loop[:, 0].astype(np.float64)
    v = uv_loop[:, 1].astype(np.float64)
    mean_u = np.bincount(isl, weights=u, minlength=n_islands) / cnt_safe
    mean_v = np.bincount(isl, weights=v, minlength=n_islands) / cnt_safe
    du = u - mean_u[isl]
    dv = v - mean_v[isl]
    dh = h64 - mean_h[isl]
    suu = np.bincount(isl, weights=du * du, minlength=n_islands)
    svv = np.bincount(isl, weights=dv * dv, minlength=n_islands)
    suv = np.bincount(isl, weights=du * dv, minlength=n_islands)
    suh = np.bincount(isl, weights=du * dh, minlength=n_islands)
    svh = np.bincount(isl, weights=dv * dh, minlength=n_islands)
    # 岭系数防细长条岛退化(u,v 近共线)
    ridge = 1e-10 * np.maximum(suu + svv, 1e-30)
    det = (suu + ridge) * (svv + ridge) - suv * suv
    det = np.where(np.abs(det) < 1e-30, 1.0, det)
    a = (suh * (svv + ridge) - svh * suv) / det
    b = (svh * (suu + ridge) - suh * suv) / det
    resid = dh - a[isl] * du - b[isl] * dv
    # 斜坡去除后再用中位数锚定常量(凸起细节不拉偏基准面)
    med = _segment_median(resid, isl, n_islands, cnt)
    return (resid - med[isl]).astype(np.float32)


def _segment_median(values, isl, n_islands, cnt):
    """逐岛中位数(向量化): 按 (岛, 值) 排序后取每段中点。"""
    order = np.lexsort((values, isl))
    starts = np.zeros(n_islands, np.int64)
    np.cumsum(cnt.astype(np.int64), out=starts)
    starts -= cnt.astype(np.int64)          # 每岛段起点
    mid = starts + (cnt.astype(np.int64) - 1) // 2
    mid = np.clip(mid, 0, max(len(values) - 1, 0))
    med = values[order[mid]] if len(values) else np.zeros(n_islands)
    med = np.where(cnt > 0, med, 0.0)
    return med


def stitch_islands(h_loop, loop_vert, island_of_loop, n_islands):
    """岛间缝合: 最小二乘求每岛常量修正, 使相邻岛在共享网格顶点处高度对齐。

    以"同一网格顶点上不同岛的高度均值应相等"为约束建 n_islands 维正规方程,
    岭正则固定全局自由度。全向量化, 岛数级别的稠密解, 开销可忽略。
    """
    if n_islands < 2:
        return h_loop
    ni = np.int64(n_islands)
    key = loop_vert.astype(np.int64) * ni + island_of_loop
    uniq, inv = np.unique(key, return_inverse=True)
    sums = np.bincount(inv, weights=h_loop.astype(np.float64))
    cnts = np.bincount(inv)
    group_mean = sums / cnts
    gv = uniq // ni          # 组所属顶点
    gi = (uniq % ni).astype(np.int64)   # 组所属岛
    # uniq 按 key 有序 → 同顶点的组相邻; 相邻对即缝合约束
    same_vert = gv[1:] == gv[:-1]
    a = gi[:-1][same_vert]
    b = gi[1:][same_vert]
    r = (group_mean[:-1] - group_mean[1:])[same_vert]   # h̄_a - h̄_b
    if a.shape[0] == 0:
        return h_loop
    deg = (np.bincount(a, minlength=n_islands)
           + np.bincount(b, minlength=n_islands)).astype(np.float64)
    rhs = np.zeros(n_islands, np.float64)
    np.add.at(rhs, a, -r)
    np.add.at(rhs, b, r)
    if n_islands <= 4096:
        mat = np.zeros((n_islands, n_islands), np.float64)
        np.add.at(mat, (a, a), 1.0)
        np.add.at(mat, (b, b), 1.0)
        np.add.at(mat, (a, b), -1.0)
        np.add.at(mat, (b, a), -1.0)
        mat[np.diag_indices(n_islands)] += 1e-6 + 1e-9 * deg.max()
        c = np.linalg.solve(mat, rhs)
    else:
        # 岛数过大时用 Jacobi 迭代(拉普拉斯系统, 收敛快且全向量化)
        c = np.zeros(n_islands, np.float64)
        dd = deg + 1e-6
        for _ in range(128):
            s = (np.bincount(a, weights=c[b], minlength=n_islands)
                 + np.bincount(b, weights=c[a], minlength=n_islands))
            c = (rhs + s) / dd
    c -= c.mean()
    return (h_loop.astype(np.float64) + c[island_of_loop]).astype(np.float32)


# ---------------------------------------------------------------------------
# 逐顶点归并
# ---------------------------------------------------------------------------

def average_loops_to_verts(loop_vals, loop_vert, vert_count):
    """逐 loop 标量 → 逐顶点平均(接缝顶点自动取两侧均值)。"""
    s = np.bincount(loop_vert, weights=loop_vals.astype(np.float64), minlength=vert_count)
    c = np.bincount(loop_vert, minlength=vert_count)
    return (s / np.maximum(c, 1)).astype(np.float32)


# ---------------------------------------------------------------------------
# 自测: python core.py
# ---------------------------------------------------------------------------

def _selftest():
    rng = np.random.default_rng(7)

    # ---- 双线性采样 ----
    h = w = 64
    field = rng.uniform(0.0, 1.0, (h, w, 3)).astype(np.float32)
    iu = ((np.arange(w) + 0.5) / w).astype(np.float32)
    iv = np.full(w, (17 + 0.5) / h, np.float32)
    got = sample_bilinear_wrap(field, iu, iv)
    err = np.abs(got - field[17]).max()
    print(f"[双线性] 像素中心最大误差 = {err:.2e}")
    assert err < 1e-5
    got2 = sample_bilinear_wrap(field, iu + 1.0, iv)
    assert np.abs(got2 - got).max() < 1e-5
    got1 = sample_bilinear_wrap(field[..., 0], iu, iv)
    assert np.abs(got1 - field[17, :, 0]).max() < 1e-5

    # B 样条: 常量/线性场精确再现 + wrap 等价 + 权重归一
    const = np.full((h, w, 2), 0.7, np.float32)
    gc = sample_bspline_wrap(const, iu, iv)
    assert np.abs(gc - 0.7).max() < 1e-6, "B样条常量再现失败"
    ramp = np.broadcast_to(np.arange(w, dtype=np.float32)[None, :], (h, w)).copy()
    mid = np.full(16, (31 + 0.5) / h, np.float32)
    xs = ((np.arange(16) * 3 + 8) + 0.5) / w
    gr = sample_bspline_wrap(ramp, xs.astype(np.float32), mid)
    assert np.abs(gr - (np.arange(16) * 3 + 8)).max() < 1e-3, "B样条线性再现失败"
    gw = sample_bspline_wrap(field, iu + 1.0, iv)
    assert np.abs(gw - sample_bspline_wrap(field, iu, iv)).max() < 1e-5, "B样条wrap失败"
    print("[B样条] 常量/线性再现 + wrap 校验通过")

    # ---- 端到端: 平面片高度场 → (n1, n0, P) 烘焙图合成 → 梯度装配 → 积分还原 ----
    # 平面绕 Z 转 30°(模拟"UV 轴 ≠ 物体轴"), 尺度 A×B 各不相同(模拟逐岛密度)
    w2 = h2 = 256
    A, B = 2.0, 1.3
    ang = np.deg2rad(30.0)
    ex = np.array([np.cos(ang), np.sin(ang), 0.0])
    ey = np.array([-np.sin(ang), np.cos(ang), 0.0])
    ez = np.array([0.0, 0.0, 1.0])
    uu = (np.arange(w2, dtype=np.float64) + 0.5) / w2
    vv = (np.arange(h2, dtype=np.float64) + 0.5) / h2
    ug, vg = np.meshgrid(uu, vv)
    height = np.zeros((h2, w2))
    for _ in range(5):
        ku, kv = int(rng.integers(1, 9)), int(rng.integers(1, 9))
        amp = float(rng.uniform(0.003, 0.01))
        ph = float(rng.uniform(0, 2 * np.pi))
        height += amp * np.sin(2 * np.pi * (ku * ug + kv * vg) + ph)
    # 边界渐落窗: 真实烘焙的图像边缘是无内容背景; 环绕差分在边界列会被
    # 稳健阈值剔除(梯度归零), 内容必须在边界处平坦才与算法前提一致
    ramp = np.minimum(np.minimum(ug, 1.0 - ug), np.minimum(vg, 1.0 - vg)) / 0.1
    window = 0.5 - 0.5 * np.cos(np.pi * np.clip(ramp, 0.0, 1.0))
    height = (height - height.mean()) * window
    dh_du = np.gradient(height, axis=1) * w2
    dh_dv = np.gradient(height, axis=0) * h2

    pos = (ug[..., None] * (A * ex) + vg[..., None] * (B * ey)).astype(np.float32)
    n0_vec = np.broadcast_to(ez, (h2, w2, 3))
    # 表面梯度(3D): dh/dx·ex + dh/dy·ey, 其中 dh/dx = dh/du / A
    grad3d = (dh_du / A)[..., None] * ex + (dh_dv / B)[..., None] * ey
    n1_vec = n0_vec - grad3d
    n1_vec = n1_vec / np.linalg.norm(n1_vec, axis=-1, keepdims=True)
    enc = lambda n: (n.astype(np.float32) + 1.0) * 0.5
    gx, gy, wmask = height_gradients(enc(n1_vec), enc(np.array(n0_vec)), pos)
    assert wmask.mean() > 0.99, "合成图应全部有效"
    rec = integrate_height(gx, gy).astype(np.float64)
    diff = rec - height
    diff -= diff.mean()
    rel = np.sqrt(np.mean(diff ** 2)) / np.sqrt(np.mean(height ** 2))
    print(f"[端到端] 旋转+异尺度平面 相对RMS误差 = {rel:.2e}  (阈值 2e-2)")
    assert rel < 2e-2, "梯度装配/积分还原失败"

    # 镜像岛(U 轴反向)——P 图携带反向, 结果仍应正确
    pos_m = ((1.0 - ug)[..., None] * (A * ex) + vg[..., None] * (B * ey)).astype(np.float32)
    grad3d_m = (-dh_du / A)[..., None] * ex + (dh_dv / B)[..., None] * ey
    n1_m = n0_vec - grad3d_m
    n1_m = n1_m / np.linalg.norm(n1_m, axis=-1, keepdims=True)
    gxm, gym, _ = height_gradients(enc(n1_m), enc(np.array(n0_vec)), pos_m)
    rec_m = integrate_height(gxm, gym).astype(np.float64)
    diff_m = rec_m - height
    diff_m -= diff_m.mean()
    rel_m = np.sqrt(np.mean(diff_m ** 2)) / np.sqrt(np.mean(height ** 2))
    print(f"[镜像岛] 相对RMS误差 = {rel_m:.2e}")
    assert rel_m < 2e-2, "镜像 UV 岛未被 P 图自动纠正"

    # 未烘焙背景(黑)与岛间大跳变: 权重应归零
    n1_bad = enc(n1_vec).copy()
    n1_bad[:8] = 0.0
    gxb, gyb, wb = height_gradients(n1_bad, enc(np.array(n0_vec)), pos)
    assert wb[:8].max() == 0.0 and np.abs(gxb[:4]).max() == 0.0, "背景未归零"

    # ---- 直算前端: 光栅化切线帧标量 + 免烘焙梯度, 应还原同一高度场 ----
    # 两个大三角覆盖全 UV 方格; 平面绕 Z 30°, 尺度 A×B(同上), T=ex, B=ey:
    # au = T·Pu = A, bu = 0, av = 0, bv = B
    corner = np.array([[-0.2, -0.2], [1.4, -0.2], [-0.2, 1.4],
                       [1.4, -0.2], [1.4, 1.4], [-0.2, 1.4]], np.float32)
    tri_uv = corner.reshape(2, 3, 2)
    attr1 = np.array([A, 0.0, 0.0, B], np.float32)
    tri_attr = np.broadcast_to(attr1, (2, 3, 4)).copy()
    grid, gmask = rasterize_tris(tri_uv, tri_attr, w2)
    assert gmask.all(), "全覆盖光栅化出现空洞"
    # 合成切线空间法线贴图: t ∝ (−dh/dx, −dh/dy, 1) (x=u·A 世界轴)
    t_map = np.stack([-(dh_du / A), -(dh_dv / B), np.ones_like(dh_du)], axis=-1)
    t_map = (t_map / np.linalg.norm(t_map, axis=-1, keepdims=True)).astype(np.float32)
    gxd, gyd, wd = gradients_from_frame_scalars(
        t_map, grid[..., 0], grid[..., 1], grid[..., 2], grid[..., 3], gmask)
    assert wd.min() > 0.99, "直算权重异常"
    rec_d = integrate_height(gxd, gyd).astype(np.float64)
    diff_d = rec_d - height
    diff_d -= diff_d.mean()
    rel_d = np.sqrt(np.mean(diff_d ** 2)) / np.sqrt(np.mean(height ** 2))
    print(f"[直算前端] 光栅化+切线帧标量梯度 相对RMS误差 = {rel_d:.2e}")
    assert rel_d < 2e-2, "直算前端还原失败"

    # 累加平均光栅化: 双份同属性三角形 → count=2, 均值等于覆盖值(马赛克免疫)
    tri_uv2 = np.concatenate([tri_uv, tri_uv], axis=0)
    tri_attr2 = np.concatenate([tri_attr, tri_attr * 3.0], axis=0)
    ssum, scnt = rasterize_tris(tri_uv2, tri_attr2, 64, accumulate=True)
    inner = scnt[8:56, 8:56]
    assert inner.min() >= 2.0, "累加计数异常"
    avg0 = ssum[32, 32, 0] / scnt[32, 32]
    assert abs(avg0 - 2.0 * A) < 1e-4, f"累加平均异常: {avg0} vs {2.0 * A}"

    # 膨胀: 挖洞后 4 圈填充应恢复邻域值; 布尔膨胀覆盖扩圈
    hole = grid.copy()
    hmask = gmask.copy()
    hmask[100:104, 100:104] = False
    hole[100:104, 100:104] = 0.0
    filled, fmask = dilate_grid(hole, hmask, 4)
    assert fmask.all() and abs(float(filled[101, 101, 0]) - float(grid[101, 101, 0])) < 1e-5
    bm = np.zeros((16, 16), bool)
    bm[8, 8] = True
    assert dilate_mask(bm, 2).sum() == 13, "布尔膨胀圈数异常"

    # ---- FC 积分基准面锚定 ----
    r2 = (ug - 0.5) ** 2 + (vg - 0.5) ** 2
    bump = 0.05 * np.exp(-r2 / 0.005)
    gx_b = (np.gradient(bump, axis=1) * w2).astype(np.float32)
    gy_b = (np.gradient(bump, axis=0) * h2).astype(np.float32)
    gx_b[np.abs(gx_b) < 1e-6] = 0.0
    gy_b[np.abs(gy_b) < 1e-6] = 0.0
    hb = integrate_height(gx_b, gy_b).astype(np.float64)
    corner_lvl = hb[:32, :32].mean()
    peak = hb.max()
    print(f"[锚定] 平地电平 = {corner_lvl:+.2e}  峰值 = {peak:.4f} (真值 0.05)")
    assert abs(corner_lvl) < 1e-3 and abs(peak - 0.05) < 0.005

    # ---- Neumann: 非周期斜坡精确还原(周期 FFT 因 wrap-around 必然失败) ----
    ramp_g = np.full((64, 64), 0.7, np.float32)
    zero_g = np.zeros_like(ramp_g)
    uu64 = (np.arange(64, dtype=np.float64) + 0.5) / 64.0
    hx = integrate_height(ramp_g, zero_g).astype(np.float64)
    ex = hx - 0.7 * uu64[None, :]
    ex -= ex.mean()
    rms_ref = np.sqrt(np.mean((0.7 * (uu64 - 0.5)) ** 2))
    rel_x = np.sqrt(np.mean(ex ** 2)) / rms_ref
    hy = integrate_height(zero_g, ramp_g).astype(np.float64)
    ey = hy - 0.7 * uu64[:, None]
    ey -= ey.mean()
    rel_y = np.sqrt(np.mean(ey ** 2)) / rms_ref
    print(f"[Neumann斜坡] x向相对RMS = {rel_x:.2e}  y向 = {rel_y:.2e}")
    assert rel_x < 2e-2 and rel_y < 2e-2, "镜像 Neumann 积分未能还原非周期斜坡"

    # ---- 边缘衰减场 ----
    seg = np.array([[[0.0, 0.5], [1.0, 0.5]]], np.float32)   # 横贯 v=0.5 的开放边
    fall = edge_falloff_field(seg, 64, 8)
    assert fall[32].max() < 0.2, "边界行未压向 0"
    assert fall[0].min() > 0.8, "远离边界处应≈1"
    assert np.all(np.diff(fall[32:41, 10]) >= -1e-6), "衰减带非单调"
    assert edge_falloff_field(np.zeros((0, 2, 2), np.float32), 32, 8).min() == 1.0
    print("[边缘衰减] 距离场/单调性校验通过")

    # ---- UV 岛并查集 ----
    loop_uv = np.array([
        [0.05, 0.05], [0.45, 0.05], [0.45, 0.45], [0.05, 0.45],
        [0.55, 0.55], [0.95, 0.55], [0.95, 0.95], [0.55, 0.95],
    ], np.float32)
    loop_vert = np.array([0, 1, 2, 3, 4, 5, 6, 7], np.int32)
    poly_of_loop = np.array([0, 0, 0, 0, 1, 1, 1, 1], np.int64)
    labels, n_isl = face_islands(loop_vert, loop_uv, poly_of_loop, 2)
    assert n_isl == 2 and labels[0] != labels[1]
    lv2 = np.array([0, 1, 2, 2, 1, 3], np.int32)
    uv2 = np.array([[0, 0], [1, 0], [0, 1], [0, 1], [1, 0], [1, 1]], np.float32)
    pol2 = np.array([0, 0, 0, 1, 1, 1], np.int64)
    _, n2 = face_islands(lv2, uv2, pol2, 2)
    assert n2 == 1
    print("[UV岛] 并查集校验通过")

    # ---- 逐岛去趋势 ----
    m = 4000
    rng2 = np.random.default_rng(3)
    u0 = rng2.uniform(0, 0.4, m)
    v0 = rng2.uniform(0, 0.4, m)
    sine = 0.002 * np.sin(2 * np.pi * 40 * u0)
    h0 = 3.0 * u0 - 2.0 * v0 + 0.7 + sine
    u1 = rng2.uniform(0.6, 0.9, m)
    v1 = rng2.uniform(0.6, 0.9, m)
    h1 = np.full(m, -5.0)
    hh_loop = np.concatenate([h0, h1]).astype(np.float32)
    uv_loop2 = np.stack([np.concatenate([u0, u1]),
                         np.concatenate([v0, v1])], axis=1).astype(np.float32)
    isl_loop = np.concatenate([np.zeros(m, np.int64), np.ones(m, np.int64)])
    out = detrend_per_island(hh_loop, uv_loop2, isl_loop, 2, 'PLANE')
    resid0 = out[:m] - (sine - sine.mean())
    assert np.abs(resid0).max() < 5e-4 and np.abs(out[m:]).max() < 1e-6
    print(f"[去趋势] 岛0残差 = {np.abs(resid0).max():.2e}")

    # ---- 岛间缝合 ----
    lv3 = np.array([0, 1, 1, 2, 2, 3], np.int64)
    il3 = np.array([0, 0, 1, 1, 2, 2], np.int64)
    hh3 = np.array([1.0, 1.0, 4.0, 4.0, -2.0, -2.0], np.float32)
    out3 = stitch_islands(hh3, lv3, il3, 3)
    assert abs(out3[2] - out3[1]) < 1e-3 and abs(out3[4] - out3[3]) < 1e-3
    print("[缝合] 岛间台阶对齐通过")

    # ---- 位置 + 法线联合优化 ----
    # 规则网格上的已知位移，其每条边差分是精确的法线梯度观测；固定一个顶点后应
    # 无需 UV 全局积分即可恢复同一位移。随后注入单条强离群观测，Huber IRLS
    # 应比普通最小二乘更接近真值。
    ny, nx = 18, 24
    yy, xx = np.mgrid[0:ny, 0:nx]
    truth = (0.03 * np.sin(2 * np.pi * xx / (nx - 1))
             * np.cos(2 * np.pi * yy / (ny - 1)))
    truth -= truth.ravel()[0]
    ids = np.arange(nx * ny).reshape(ny, nx)
    edge_i = np.concatenate([ids[:, :-1].ravel(), ids[:-1, :].ravel()])
    edge_j = np.concatenate([ids[:, 1:].ravel(), ids[1:, :].ravel()])
    target = truth.ravel()[edge_j] - truth.ravel()[edge_i]
    pin = np.zeros(nx * ny, bool)
    pin[0] = True
    solved, joint_stats = solve_joint_position_normal(
        edge_i, edge_j, target, nx * ny, position_weight=0.0, pinned=pin,
        irls_iters=0, max_iter=2000, tolerance=1e-10)
    joint_rms = np.sqrt(np.mean((solved.astype(np.float64) - truth.ravel()) ** 2))
    assert joint_rms < 2e-6, f"联合优化精确场恢复失败: {joint_rms}"

    target_bad = target.copy()
    target_bad[target_bad.size // 3] += 0.5
    solved_ls, _ = solve_joint_position_normal(
        edge_i, edge_j, target_bad, nx * ny, position_weight=0.001, pinned=pin,
        irls_iters=0, max_iter=2000, tolerance=1e-9)
    solved_robust, robust_stats = solve_joint_position_normal(
        edge_i, edge_j, target_bad, nx * ny, position_weight=0.001, pinned=pin,
        irls_iters=5, max_iter=2000, tolerance=1e-9)
    ls_rms = np.sqrt(np.mean((solved_ls.astype(np.float64) - truth.ravel()) ** 2))
    robust_rms = np.sqrt(np.mean((solved_robust.astype(np.float64) - truth.ravel()) ** 2))
    assert robust_rms < ls_rms * 0.5, \
        f"IRLS 未能限制离群误差: {robust_rms} vs {ls_rms}"
    assert robust_stats["downweighted_fraction"] > 0.0

    # 位置项必须随细分边面积缩放；同一连续场加密一倍后，屏蔽强度不应暴涨。
    def screened_amplitude(grid_y, grid_x):
        sy, sx = np.mgrid[0:grid_y, 0:grid_x]
        known = (0.03 * np.sin(2 * np.pi * sx / (grid_x - 1))
                 * np.cos(2 * np.pi * sy / (grid_y - 1)))
        grid_ids = np.arange(grid_x * grid_y).reshape(grid_y, grid_x)
        si = np.concatenate(
            [grid_ids[:, :-1].ravel(), grid_ids[:-1, :].ravel()])
        sj = np.concatenate(
            [grid_ids[:, 1:].ravel(), grid_ids[1:, :].ravel()])
        sd = known.ravel()[sj] - known.ravel()[si]
        measure = np.concatenate([
            np.full(grid_y * (grid_x - 1), (1.0 / (grid_x - 1)) ** 2 / 2.0),
            np.full((grid_y - 1) * grid_x, (1.0 / (grid_y - 1)) ** 2 / 2.0),
        ])
        estimate, _ = solve_joint_position_normal(
            si, sj, sd, grid_x * grid_y, position_weight=10.0,
            position_measure=measure, irls_iters=0,
            max_iter=2000, tolerance=1e-10)
        return float(np.dot(estimate, known.ravel())
                     / np.dot(known.ravel(), known.ravel()))

    amp_coarse = screened_amplitude(18, 24)
    amp_fine = screened_amplitude(36, 48)
    assert abs(amp_coarse - amp_fine) < 0.02, \
        f"位置先验随细分级别漂移: {amp_coarse} vs {amp_fine}"

    # 非零固定位置也应正确移入右端，而不是被隐式当作 0。
    nonzero_pin, _ = solve_joint_position_normal(
        np.array([0]), np.array([1]), np.array([2.0]), 2,
        position_weight=0.0, pinned=np.array([True, False]),
        prior=np.array([3.0, 5.0]), irls_iters=0,
        max_iter=32, tolerance=1e-12)
    assert np.allclose(nonzero_pin, [3.0, 5.0], atol=1e-7)

    # 梯度平滑应精确保留常量坡度；单个四边面的面内边观测应匹配解析积分。
    const_grad = np.full((32, 32, 2), [0.7, -0.2], np.float32)
    const_smooth = smooth_fields_neumann(const_grad, 0.04)
    assert np.abs(const_smooth - const_grad).max() < 1e-6
    quad_lv = np.array([0, 1, 2, 3], np.int64)
    quad_uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    quad_gx = np.full((16, 16), 0.7, np.float32)
    quad_gy = np.full((16, 16), -0.2, np.float32)
    quad_w = np.ones((16, 16), np.float32)
    qi, qj, qd, qw = gradient_constraints_from_loops(
        quad_lv, quad_uv, np.array([0]), np.array([4]),
        quad_gx, quad_gy, quad_w, merge_duplicates=False)
    assert np.array_equal(qi, [0, 1, 2, 3]) and np.array_equal(qj, [1, 2, 3, 0])
    assert np.allclose(qd, [0.7, -0.2, -0.7, 0.2], atol=1e-6)
    assert np.allclose(qw, 1.0)
    print(f"[联合优化] 精确场 RMS {joint_rms:.2e} | "
          f"离群 LS/IRLS {ls_rms:.2e}/{robust_rms:.2e} | "
          f"屏蔽幅值 粗/细 {amp_coarse:.3f}/{amp_fine:.3f} | "
          f"PCG {joint_stats['pcg_iterations']} iter")

    # ---- 顶点归并 ----
    lv = np.array([0, 1, 1], np.int32)
    avg = average_loops_to_verts(np.array([1.0, 2.0, 4.0], np.float32), lv, 2)
    assert np.allclose(avg, [1.0, 3.0])
    print("[顶点平均] 标量归并通过")

    print("[OK] core.py 自测全部通过")


if __name__ == "__main__":
    _selftest()
