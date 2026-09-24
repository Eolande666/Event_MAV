"""One local IOC call followed by a parameter-free shared rigid-rotation test."""
import numpy as np
from scipy import ndimage as ndi
from scipy.special import fdtr


def surface_flow(surface, cfg):
    active = np.isfinite(surface)
    weights = ndi.gaussian_filter(active.astype(float), 0.8)
    smooth = ndi.gaussian_filter(np.where(active, surface, 0.0), 0.8)/np.maximum(weights, 1e-12)
    gy, gx = np.gradient(smooth)
    base = active & (weights >= 0.25) & (gx*gx+gy*gy > 1e-12)
    gx, gy = np.where(base, gx, 0), np.where(base, gy, 0)

    def total(a):
        return ndi.uniform_filter(a.astype(float), cfg.ioc_size, mode='constant')*cfg.ioc_size**2

    a, b, c = total(gx*gx), total(gx*gy), total(gy*gy)
    bx, by, support = total(gx), total(gy), total(base)
    det = a*c-b*b
    disc = np.sqrt(np.maximum((a-c)**2+4*b*b, 0))
    condition = (a+c-disc)/np.maximum(a+c+disc, 1e-20)
    valid = base & (support >= cfg.ioc_min_support) & (det > 1e-22) & (condition >= cfg.ioc_condition_ratio)
    safe = np.where(valid, det, 1)
    vx, vy = (c*bx-b*by)/safe, (a*by-b*bx)/safe
    valid &= np.hypot(vx, vy) <= cfg.max_speed_px_s
    return np.where(valid, vx, 0), np.where(valid, vy, 0), valid


def _block_means(positive, negative, common, block_size):
    """Collapse correlated IOC pixels into equally weighted, non-overlapping blocks."""
    yy, xx = np.nonzero(common)
    if not len(xx):
        return np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2))
    block_height = (common.shape[0]+block_size-1)//block_size
    block_width = (common.shape[1]+block_size-1)//block_size
    ids = (yy//block_size)*block_width+(xx//block_size)
    counts = np.bincount(ids, minlength=block_height*block_width).astype(float)
    occupied = counts > 0

    def mean(values):
        sums = np.bincount(ids, weights=np.asarray(values, float),
                           minlength=len(counts))
        return sums[occupied]/counts[occupied]

    coordinates = np.column_stack((mean(xx), mean(yy)))
    velocity_p = np.column_stack((mean(positive[0][common]), mean(positive[1][common])))
    velocity_n = np.column_stack((mean(negative[0][common]), mean(negative[1][common])))
    return coordinates, velocity_p, velocity_n


def _single_omega(coordinates, velocity):
    tangent = np.column_stack((-coordinates[:, 1], coordinates[:, 0]))
    leverage = float(np.sum(tangent*tangent))
    if leverage <= np.finfo(float).eps:
        return 0.0
    return float(np.sum(tangent*velocity)/leverage)


def score_fields(positive, negative, cfg):
    """Test whether both polarities share one rigid angular velocity.

    The score is the product of three dimensionless quantities derived from
    nested least-squares models: angular-velocity significance, the energy
    explained beyond translation, and rotation energy relative to affine
    deformation. No learned or hand-selected rotation scale is used.
    """
    common = positive[2] & negative[2]
    common &= ndi.uniform_filter(common.astype(float), 3, mode='constant') >= 0.55
    valid_pixels = int(common.sum())
    result = {
        'observable': False, 'valid_pixels': valid_pixels, 'rotation_blocks': 0,
        'rotation_q': 0.0, 'rotation_significance': 0.0,
        'rotation_fit': 0.0, 'rotation_rigidity': 0.0,
        'rotation_omega': 0.0, 'rotation_residual': 0.0,
        'rotation_polarity_sign': 0,
        'rotation_omega_positive': 0.0, 'rotation_omega_negative': 0.0,
    }
    coordinates, velocity_p, velocity_n = _block_means(
        positive, negative, common, cfg.ioc_size)
    blocks = len(coordinates)
    result['rotation_blocks'] = blocks
    if blocks < 3:
        return result

    coordinates -= np.mean(coordinates, axis=0)
    velocity_p -= np.mean(velocity_p, axis=0)
    velocity_n -= np.mean(velocity_n, axis=0)
    scatter = coordinates.T@coordinates
    scatter_det = float(scatter[0, 0]*scatter[1, 1]-scatter[0, 1]*scatter[1, 0])
    scatter_scale = float(np.trace(scatter))
    if scatter_det <= np.finfo(float).eps*max(scatter_scale*scatter_scale, 1.0):
        return result

    result['observable'] = True
    omega_p = _single_omega(coordinates, velocity_p)
    omega_n = _single_omega(coordinates, velocity_n)
    result.update(rotation_omega_positive=omega_p,
                  rotation_omega_negative=omega_n)

    tangent = np.column_stack((-coordinates[:, 1], coordinates[:, 0]))
    leverage = 2*float(np.sum(tangent*tangent))
    null_energy = float(np.sum(velocity_p*velocity_p)+np.sum(velocity_n*velocity_n))
    best = None
    for polarity_sign in (1, -1):
        aligned_n = polarity_sign*velocity_n
        omega = float(np.sum(tangent*(velocity_p+aligned_n))/leverage)
        residual_p = velocity_p-omega*tangent
        residual_n = aligned_n-omega*tangent
        residual_energy = float(np.sum(residual_p*residual_p)+np.sum(residual_n*residual_n))
        if best is None or residual_energy < best[0]:
            best = residual_energy, polarity_sign, omega, aligned_n

    residual_energy, polarity_sign, omega, aligned_n = best
    residual_dof = 4*blocks-5  # four translations plus one shared omega
    fit = (float(np.clip(1-residual_energy/null_energy, 0, 1))
           if null_energy > np.finfo(float).eps else 0.0)
    explained_energy = max(null_energy-residual_energy, 0.0)
    if explained_energy <= np.finfo(float).eps:
        significance = 0.0
    elif residual_energy <= np.finfo(float).eps*max(null_energy, 1.0):
        significance = 1.0
    else:
        f_statistic = explained_energy/(residual_energy/residual_dof)
        significance = float(np.clip(fdtr(1, residual_dof, f_statistic), 0, 1))

    cross = coordinates.T@(velocity_p+aligned_n)
    # The normal matrix for two polarity fields is 2*scatter.  Expanding its
    # 2x2 inverse avoids a general least-squares decomposition per candidate.
    inverse_scatter = np.array(((scatter[1, 1], -scatter[0, 1]),
                                (-scatter[1, 0], scatter[0, 0])))/scatter_det
    gradient = (0.5*inverse_scatter@cross).T
    skew = 0.5*(gradient-gradient.T)
    symmetric = 0.5*(gradient+gradient.T)
    rotation_field = coordinates@skew.T
    deformation_field = coordinates@symmetric.T
    rotation_energy = float(np.sum(rotation_field*rotation_field))
    deformation_energy = float(np.sum(deformation_field*deformation_field))
    gradient_energy = rotation_energy+deformation_energy
    rigidity = rotation_energy/gradient_energy if gradient_energy > np.finfo(float).eps else 0.0

    rotation_q = float(np.clip(significance*fit*rigidity, 0, 1))
    result.update(
        rotation_q=rotation_q,
        rotation_significance=significance,
        rotation_fit=fit,
        rotation_rigidity=float(np.clip(rigidity, 0, 1)),
        rotation_omega=omega,
        rotation_residual=float(np.sqrt(residual_energy/(2*blocks))),
        rotation_polarity_sign=polarity_sign,
    )
    return result


def evaluate_slice(events, box, end, cfg):
    """Compute IOC on one selected time interval."""
    x0, y0, x1, y1 = box
    shape = (max(3, y1-y0), max(3, x1-x0))
    x, y = events[:, 0].astype(int)-x0, events[:, 1].astype(int)-y0
    fields = []
    for sign in (1, -1):
        surface = np.full(shape, -np.inf)
        use = events[:, 3] == sign
        np.maximum.at(surface, (y[use], x[use]), events[use, 2]-end)
        fields.append(surface_flow(surface, cfg))
    return score_fields(*fields, cfg)


def evaluate(events, box, end, cfg, *, start=None):
    """Run IOC once on the trailing rotation_window_ms interval."""
    if start is None:
        start=end-cfg.window_ms/1000
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        raise ValueError('rotation window must have finite start < end')
    events = np.asarray(events).reshape(-1, 4)
    left = max(start, end-cfg.rotation_window_ms/1000)
    local = events[(events[:, 2] >= left) & (events[:, 2] < end)]
    return evaluate_slice(local, box, end, cfg)
