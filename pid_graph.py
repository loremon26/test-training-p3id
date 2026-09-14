"""Shared pixel-coordinate graph geometry, patching, export and overlays."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from prepare_pid2graph_paper_dataset import parse_graph, write_graph, PID_NODE_CLASS_TO_ID

BOX_KEYS = ("xmin", "ymin", "xmax", "ymax")


def box(node):
    return np.array([node[k] for k in BOX_KEYS], dtype=float)


def center(node):
    b = box(node)
    return (b[:2] + b[2:]) / 2


def set_box(node, bounds):
    return dict(node, **dict(zip(BOX_KEYS, map(float, bounds))))


def point_node(node_id, xy, label, size=8):
    x, y = xy
    return dict(id=str(node_id), label=label, score=1.,
                xmin=x-size/2, ymin=y-size/2, xmax=x+size/2, ymax=y+size/2)


def clip_segment(a, b, bounds):
    """Slab clipping, including segments whose two endpoints lie outside."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    lo, hi = 0., 1.
    delta = b-a
    for axis in range(2):
        if abs(delta[axis]) < 1e-12:
            if not bounds[axis] <= a[axis] <= bounds[axis+2]:
                return None
        else:
            t0 = (bounds[axis]-a[axis])/delta[axis]
            t1 = (bounds[axis+2]-a[axis])/delta[axis]
            lo, hi = max(lo, min(t0, t1)), min(hi, max(t0, t1))
    if hi-lo <= 1e-10:
        return None
    return a+lo*delta, a+hi*delta, lo, hi


def crop_graph(nodes, edges, bounds, border_size=8):
    """Clip boxes/straight graph segments and create a border node at each cut."""
    bounds = np.asarray(bounds, float)
    offset = np.tile(bounds[:2], 2)
    original = {n['id']: n for n in nodes}
    kept = {}
    for n in nodes:
        b = box(n)
        b[:2], b[2:] = np.maximum(b[:2], bounds[:2]), np.minimum(b[2:], bounds[2:])
        if np.all(b[2:] > b[:2]):
            kept[n['id']] = set_box(n, b-offset)
    result_edges = []
    for i, e in enumerate(edges):
        a, b = center(original[e['source']]), center(original[e['target']])
        clipped = clip_segment(a, b, bounds)
        if clipped is None:
            continue
        p, q, _, _ = clipped
        ends = []
        for end, (key, xy) in enumerate(((e['source'], p),(e['target'], q))):
            # A partially visible symbol remains a symbol, not a spurious border.
            if key not in kept:
                key = f'cut:{i}:{end}'
                kept[key] = point_node(key, xy-bounds[:2], 'border', border_size)
            ends.append(key)
        if ends[0] != ends[1]:
            result_edges.append(dict(e, source=ends[0], target=ends[1]))
    return list(kept.values()), result_edges


def transform_graph(nodes, edges, matrix):
    result = []
    for n in nodes:
        x1, y1, x2, y2 = box(n)
        corners = np.array([[x1,y1,1], [x1,y2,1], [x2,y1,1], [x2,y2,1]]) @ np.asarray(matrix).T
        result.append(set_box(n, [*corners.min(0), *corners.max(0)]))
    return result, [dict(e) for e in edges]


def patch_offsets(length, size, stride):
    if size <= 0 or stride <= 0 or stride > size/2:
        raise ValueError('Patch size must be positive; stride must give at least 50% overlap')
    last = max(length-size, 0)
    return sorted(set([*range(0, last+1, stride), last]))


def patch_plan(image_path, graph_path, output_dir, size=1500, stride=750, resize=(7000,4500)):
    image_path, output_dir = Path(image_path), Path(output_dir)
    if len(resize)!=2 or min(resize)<=0:
        raise ValueError('Resize must contain positive width and height')
    patch_offsets(resize[0],size,stride)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f'Output already contains a drawing: {output_dir}')
    with Image.open(image_path) as src:
        original_size = src.size
        image = src.convert('RGB').resize(tuple(resize), Image.Resampling.LANCZOS)
    nodes, edges = parse_graph(Path(graph_path)) if graph_path else ([], [])
    sx, sy = resize[0]/original_size[0], resize[1]/original_size[1]
    nodes, edges = transform_graph(nodes, edges, [[sx,0,0],[0,sy,0]])
    manifest = dict(schema_version=1, drawing_id=image_path.stem,
                    image=str(image_path.resolve()), graph=str(Path(graph_path).resolve()) if graph_path else None,
                    original_size=list(original_size), resized_size=list(resize),
                    patch_size=size, stride=stride, scale=[sx,sy], patches=[])
    for y in patch_offsets(resize[1], size, stride):
        for x in patch_offsets(resize[0], size, stride):
            idx = f'{len(manifest["patches"]):05d}'
            patch = Image.new('RGB', (size,size), 'white')
            patch.paste(image.crop((x,y,min(x+size,resize[0]),min(y+size,resize[1]))))
            patch.save(output_dir/f'{idx}.png')
            ns, es = crop_graph(nodes, edges, [x,y,x+size,y+size])
            if graph_path:
                write_graph(output_dir/f'{idx}.graphml', ns, es)
            manifest['patches'].append(dict(id=idx, image=f'{idx}.png',
                graph=f'{idx}.graphml' if graph_path else None, offset=[x,y], size=[size,size],
                nodes=len(ns), edges=len(es)))
    (output_dir/'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def save_graph_json(path, nodes, edges, **metadata):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(dict(schema_version=1, coordinate_space='pixels',
                                        nodes=nodes, edges=edges, **metadata), indent=2))


def draw_line(draw, a, b, label, fill, width=2):
    if label == 'solid':
        draw.line([tuple(a),tuple(b)], fill=fill, width=width)
    else:
        a, b = np.asarray(a), np.asarray(b)
        length = float(np.linalg.norm(b-a))
        for d in np.arange(0, length, 14):
            p, q = a+(b-a)*d/max(length,1e-9), a+(b-a)*min(d+8,length)/max(length,1e-9)
            draw.line([tuple(p),tuple(q)], fill=fill, width=width)


def overlay(image, nodes, edges, output, confidence=False):
    if not isinstance(image, Image.Image):
        with Image.open(image) as src:
            image = src.convert('RGB')
    else:
        image = image.convert('RGB').copy()
    draw = ImageDraw.Draw(image)
    lookup = {n['id']: n for n in nodes}
    for e in edges:
        draw_line(draw, center(lookup[e['source']]), center(lookup[e['target']]),
                  e['label'], '#00a040' if e['label']=='solid' else '#db6500', 3)
    colors = ['#2455d6','#a04020','#cb2060','#833ec5','#0096a0','#778000','#206b30','#f02020','#0055ee','#ed00db']
    for n in nodes:
        c = colors[PID_NODE_CLASS_TO_ID[n['label']]-1]
        draw.rectangle(tuple(box(n)), outline=c, width=2)
        if PID_NODE_CLASS_TO_ID[n['label']] >= 8:
            x,y = center(n)
            draw.ellipse((x-3,y-3,x+3,y+3), fill=c)
        else:
            label = n['label'] + (f" {n.get('score',1):.2f}" if confidence else '')
            draw.text((n['xmin'],max(0,n['ymin']-12)), label, fill=c)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
