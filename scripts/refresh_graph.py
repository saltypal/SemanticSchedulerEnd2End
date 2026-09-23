"""Refresh Graphify over canonical source; preserve exact source revision metadata.

Use --ast-only for extraction. --semantic accepts a Graphify semantic fragment
reviewed against current documentation; without one the graph is code-only.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    from graphify.extract import extract
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ast-only', action='store_true')
    parser.add_argument('--semantic', type=Path)
    parser.add_argument('--reuse-ast', action='store_true')
    args = parser.parse_args()
    output = ROOT / 'graphify-out'
    output.mkdir(exist_ok=True)
    files = sorted(path for name in ('EncDec', 'SemParam', 'Scheduler', 'INFRA')
                   for path in (ROOT/name).rglob('*.py') if '__pycache__' not in path.parts)
    ast_path = output / '.graphify_ast.json'
    ast = json.loads(ast_path.read_text()) if args.reuse_ast else extract(files)
    ast_path.write_text(json.dumps(ast, indent=2), encoding='utf-8')
    if args.ast_only:
        print(f"AST: {len(files)} files; {len(ast['nodes'])} nodes; {len(ast['edges'])} edges")
        return
    from graphify.build import build_from_json
    from graphify.cluster import cluster, score_all
    from graphify.analyze import god_nodes, surprising_connections, suggest_questions
    from graphify.report import generate
    from graphify.export import to_json, to_html
    semantic = json.loads(args.semantic.read_text(encoding='utf-8')) if args.semantic else {'nodes': [], 'edges': []}
    known = {node['id'] for node in ast['nodes']}
    merged = {**ast, 'nodes': ast['nodes'] + [node for node in semantic['nodes'] if node['id'] not in known],
              'edges': ast['edges'] + semantic['edges'], 'hyperedges': semantic.get('hyperedges', [])}
    graph = build_from_json(merged)
    if not graph.number_of_nodes():
        raise RuntimeError('Graph extraction produced no nodes')
    communities = cluster(graph)
    labels = {}
    for key, members in communities.items():
        counts = Counter()
        for member in members:
            source = str(graph.nodes[member].get('source_file', '')).replace('\\', '/')
            for module in ('EncDec', 'SemParam', 'Scheduler', 'INFRA'):
                if module + '/' in source:
                    counts[module] += 1
        labels[key] = (counts.most_common(1)[0][0] if counts else 'Architecture') + f' community {key}'
    def git(*args):
        return subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}', '-C', str(ROOT), *args], text=True).strip()
    digest = hashlib.sha256()
    source_hashes = {}
    for path in files:
        name = path.relative_to(ROOT).as_posix()
        content = hashlib.sha256(path.read_bytes()).hexdigest()
        source_hashes[name] = content
        digest.update(name.encode() + content.encode())
    metadata = {'git_commit': git('rev-parse', 'HEAD'), 'dirty_state': git('status', '--porcelain'),
                'source_sha256': digest.hexdigest(), 'source_files': source_hashes,
                'scope': 'canonical modules, tests and supplied architecture semantic fragment; legacy wrappers/assets excluded',
                'semantic_fragment': str(args.semantic) if args.semantic else None,
                'semantic_sha256': hashlib.sha256(args.semantic.read_bytes()).hexdigest() if args.semantic else None}
    metadata_path = output / 'source_revision.json'
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    detection = {'root': str(ROOT), 'files': {'code': [str(path) for path in files]}, 'total_files': len(files),
                 'total_words': 0, 'counts': {'code': len(files)}}
    report = generate(graph, communities, score_all(graph, communities), labels, god_nodes(graph),
                      surprising_connections(graph, communities), detection, {'input':0,'output':0}, str(ROOT),
                      suggested_questions=suggest_questions(graph, communities, labels))
    (output/'GRAPH_REPORT.md').write_text(report + '\n\nSource revision and content hashes: source_revision.json\n', encoding='utf-8')
    to_json(graph, communities, str(output/'graph.json'), force=True, built_at_commit=metadata['git_commit'])
    to_html(graph, communities, str(output/'graph.html'), community_labels=labels)
    (output/'.graphify_extract.json').write_text(json.dumps(merged, indent=2), encoding='utf-8')
    (output/'.graphify_labels.json').write_text(json.dumps(labels, indent=2), encoding='utf-8')
    dangling = [edge for edge in merged['edges'] if edge['source'] not in graph or edge['target'] not in graph]
    (output/'AUDIT.md').write_text(f"# Graph audit\n\nNodes: {graph.number_of_nodes()}; edges: {graph.number_of_edges()}; communities: {len(communities)}.\n\nDangling extracted edges after build: {len(dangling)}.\n\nCommit: {metadata['git_commit']}\n\nWorking-source SHA256: {metadata['source_sha256']}\n\nA dirty working tree is represented by these per-file hashes, not by the commit alone. Semantic extraction token usage was not instrumented.\n", encoding='utf-8')
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges; {output}")


if __name__ == '__main__':
    main()
