import importlib.resources as ir
import os
import yaml


def _deep_merge(base, override):
    """Recursively merge two dicts (override wins)."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _ensure_yml_suffix(path):
    p = os.fspath(path)
    root, ext = os.path.splitext(p)
    if ext.lower() in ('.yml', '.yaml'):
        return p
    return p + '.yml'


def _normalize_config_path(path):
    p = _ensure_yml_suffix(path)
    return os.path.abspath(p)


def load_yaml_with_comments(path):
    """Load YAML (supports comments by YAML spec)."""
    p = _normalize_config_path(path)
    with open(p, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a mapping (top-level dict): {p}")
    return data


def load_project_and_system_config(project_config_path, system_config_path=None):
    project_cfg = load_yaml_with_comments(project_config_path)
    if system_config_path is None:
        system_cfg = load_yaml_with_comments(SYSTEM_CONFIG_DEFAULT)
    else:
        system_cfg = load_yaml_with_comments(system_config_path)
    return _deep_merge(system_cfg, project_cfg)


def _get_default_asset(*relpath):
    try:
        return ir.files('massivora').joinpath(*relpath)
    except Exception as e:
        raise FileNotFoundError(f"Missing config template asset: {relpath}") from e


PROJECT_CONFIG_TEMPLATE = _get_default_asset('config_template', 'project.yml')
SYSTEM_CONFIG_DEFAULT = _get_default_asset('config_template', 'system.yml')


def write_text_if_missing(path, overwrite=False):
    p = _normalize_config_path(path)
    os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
    if os.path.exists(p) and not overwrite:
        return p

    text = PROJECT_CONFIG_TEMPLATE.read_text(encoding='utf-8')
    with open(p, 'w', encoding='utf-8') as f:
        f.write(text)
    return p
