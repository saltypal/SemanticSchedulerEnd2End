"""Modular entry point with temporary compatibility for historical stage actions."""
import sys
from pathlib import Path

def main():
    legacy = {'semantic-run','semantic-validate','validate-manifest'}
    use_legacy = any(item in legacy for item in sys.argv[1:])
    if '--config' in sys.argv and '--module' not in sys.argv:
        index = sys.argv.index('--config') + 1
        if index < len(sys.argv):
            from INFRA.configuration import read_yaml
            content = read_yaml(sys.argv[index])
            use_legacy = use_legacy or ('stage' in content and 'module' not in content)
    if use_legacy:
        from INFRA.legacy_cli import main as run
    else:
        from INFRA.cli import main as run
    run()

def __getattr__(name):
    # Existing notebook imports continue to work without eager neural imports.
    from INFRA import legacy_cli
    return getattr(legacy_cli,name)

if __name__ == '__main__':
    main()
