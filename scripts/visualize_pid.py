"""Render node boxes and graph edges over one generated drawing or patch."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pid_graph import overlay


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--graph', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    from prepare_pid2graph_paper_dataset import parse_graph
    nodes, edges = parse_graph(a.graph)
    overlay(a.image, nodes, edges, a.output)
    print(f'{len(nodes)} nodes, {len(edges)} edges -> {a.output}')


if __name__ == '__main__':
    main()
