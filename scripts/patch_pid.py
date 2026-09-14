"""Patch one drawing or a directory of complete image/GraphML pairs."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pid_graph import patch_plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', type=Path)
    p.add_argument('--graph', type=Path)
    p.add_argument('--input-dir', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--patch-size', type=int, default=1500)
    p.add_argument('--stride', type=int, default=750)
    p.add_argument('--resize', type=int, nargs=2, default=[7000,4500], metavar=('WIDTH','HEIGHT'))
    a = p.parse_args()
    if bool(a.image) == bool(a.input_dir):
        p.error('Specify exactly one of --image or --input-dir')
    pairs = [(a.image,a.graph,a.output_dir)] if a.image else []
    if a.input_dir:
        for g in sorted(a.input_dir.rglob('*.graphml')):
            images = [g.with_suffix(ext) for ext in ('.png','.jpg','.jpeg') if g.with_suffix(ext).is_file()]
            if len(images) != 1:
                raise ValueError(f'Expected one image beside {g}')
            pairs.append((images[0],g,a.output_dir/g.relative_to(a.input_dir).with_suffix('')))
        if not pairs:
            raise ValueError('No GraphML pairs found')
    for image, graph, output in pairs:
        manifest = patch_plan(image,graph,output,a.patch_size,a.stride,a.resize)
        print(f'{image}: {len(manifest["patches"])} patches -> {output}', flush=True)


if __name__ == '__main__':
    main()
