#!/usr/bin/env python3
"""Print XIR child-subgraph device placement for a compiled xmodel."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xmodel", required=True)
    args = parser.parse_args()
    try:
        import xir
    except ImportError as exc:
        raise SystemExit("xir is unavailable; run this inside a Vitis AI environment") from exc
    graph = xir.Graph.deserialize(args.xmodel)
    children = graph.get_root_subgraph().toposort_child_subgraph()
    for index, child in enumerate(children):
        device = child.get_attr("device") if child.has_attr("device") else "UNKNOWN"
        print(index, child.get_name(), device, child.get_op_num())


if __name__ == "__main__":
    main()
