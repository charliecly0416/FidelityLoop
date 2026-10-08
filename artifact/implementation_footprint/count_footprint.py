"""Recompute audited source footprint without importing experimental dependencies."""
import ast
import hashlib
import io
import json
from pathlib import Path
import tokenize

ROOT = Path(__file__).resolve().parent


def effective_lines(source):
    """Physical nonblank lines touched by code tokens, excluding AST docstrings."""
    tree = ast.parse(source)
    docstrings = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                expr = node.body[0]
                if isinstance(expr.value, ast.Constant) and isinstance(expr.value.value, str):
                    docstrings.append(((expr.lineno, expr.col_offset),
                                       (expr.end_lineno, expr.end_col_offset)))
    ignored = {tokenize.ENCODING, tokenize.ENDMARKER, tokenize.COMMENT,
               tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT}
    lines = set()
    physical = source.splitlines()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in ignored:
            continue
        if any(start <= token.start and token.end <= end for start, end in docstrings):
            continue
        lines.update(i for i in range(token.start[0], token.end[0] + 1)
                     if i <= len(physical) and physical[i - 1].strip())
    return lines


def symbols(source):
    out = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = min([node.lineno] + [x.lineno for x in node.decorator_list])
            out[node.name] = (start, node.end_lineno)
    return out


def component(source, names):
    found = symbols(source)
    ranges = [found[name] for name in names]
    return {line for start, end in ranges for line in range(start, end + 1)}


def partitions(path, source):
    """Disjoint boundaries; all remaining source lines go to an explicit row."""
    if path.endswith('/formal_v2/baselines.py'):
        return [('B-HPA policy', ['HPA']), ('B-PRED policy', ['Predictor']),
                ('B-MMC policy', ['MMC'])], 'Baseline shared adapter and scaffolding'
    if path.endswith('/formal_v2/runtime.py'):
        return [('Guard adapter', ['GuardAdapter'])], 'Shared physical runtime and admission'
    if path.endswith('/maxopt_v3/policy.py'):
        return [], 'Guard policy and causal observation helpers'
    if path.endswith('/ppo_deployment.py'):
        return [('PPO decision and validation', ['validate_public', 'DecisionPolicy'])], 'PPO replay and physical deployment'
    if path.endswith('/e2_deployment.py'):
        return [], 'PPO E2 checkpoint deployment'
    if path.endswith('/ppo_capacity.py'):
        return [('PPO training interface', ['CapacityTransition', 'CapacityTrainer',
                                           'make_transition', 'collect_rollout'])], 'PPO network adapter and scaffolding'
    if path.endswith('/ppo_env.py'):
        return [('PPO training environment', ['_StopEpisode', 'CapacityEnv', 'environment_pair'])], 'PPO observation and replay interface'
    raise ValueError(path)


def calculate(root=ROOT):
    manifest = json.loads((root / 'SOURCE_MANIFEST.json').read_text())
    rows, context = [], []
    for entry in manifest['files']:
        if entry['role'] == 'hash_only_context':
            context.append(dict(entry))
            continue
        raw = (root / entry['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry['sha256'], entry['path']
        source = raw.decode()
        if entry['role'] == 'configuration':
            context.append(dict(entry, top_level_keys=sorted(json.loads(source))))
            continue
        code = effective_lines(source)
        if entry['role'] == 'support_context':
            context.append(dict(entry, effective_physical_lines=len(code),
                                physical_lines=len(source.splitlines())))
            continue
        explicit, remainder = partitions(entry['path'], source)
        allocated = set()
        for label, names in explicit:
            coverage = component(source, names)
            selected = coverage & code
            assert not allocated & selected
            allocated |= selected
            rows.append(dict(component=label, source=entry['path'], source_sha256=entry['sha256'],
                             symbols={name: list(symbols(source)[name]) for name in names},
                             effective_physical_lines=len(selected), counted_lines=sorted(selected)))
        selected = code - allocated
        rows.append(dict(component=remainder, source=entry['path'], source_sha256=entry['sha256'],
                         scope='whole module excluding explicitly allocated symbols',
                         effective_physical_lines=len(selected), counted_lines=sorted(selected)))
        assert len(code) == sum(r['effective_physical_lines'] for r in rows if r['source'] == entry['path'])
    counts = {row['component']: row['effective_physical_lines'] for row in rows}
    network = next(row['effective_physical_lines'] for row in context
                   if row['path'].endswith('/ppo_network.py'))
    paper_rows = [dict(policy=label, policy_lines=counts[label + ' policy'],
                       adapter_lines=None, adapter_reference='shared baseline adapter: ' +
                       str(counts['Baseline shared adapter and scaffolding']))
                  for label in ('B-HPA', 'B-PRED', 'B-MMC')]
    paper_rows += [dict(policy='Guard', policy_lines=counts['Guard policy and causal observation helpers'],
                        adapter_lines=counts['Guard adapter'],
                        dependency_note='Also uses the existing TargetPolicy and calibrated service model.'),
                   dict(policy='PPO', policy_lines=counts['PPO decision and validation'] +
                        counts['PPO network adapter and scaffolding'] + network,
                        adapter_lines=sum(counts[k] for k in ('PPO replay and physical deployment',
                            'PPO E2 checkpoint deployment', 'PPO observation and replay interface')),
                        dependency_note='Policy count includes the full 136-line inherited network module; training and provenance support separately inventoried.')]
    return dict(schema='fidelityloop-implementation-footprint-v1',
                method='Nonblank physical lines touched by Python code tokens; comments, indentation/newline tokens, and AST-recognized module/class/function docstrings excluded; decorators and multiline code included. No semicolon splitting.',
                interpretation='Implementation size of frozen experimental source, not changed lines, porting time, or total framework size. Shared modules counted once; context dependencies separately inventoried.',
                source_archive_sha256=manifest['archive_sha256'], components=rows,
                suggested_paper_rows=paper_rows,
                partitioned_effective_physical_lines=sum(r['effective_physical_lines'] for r in rows),
                supporting_context=context,
                historical_change_evidence=dict(backend_delta=None, ledger_delta=None,
                                               llama_porting_delta=None,
                                               reason='No complete matched before/after implementation snapshots established; no zero-change or config-only claim.'))


if __name__ == '__main__':
    result = calculate()
    print(json.dumps(result, indent=2, sort_keys=True))
