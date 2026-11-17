"""The agent environment's settings (docs/specs/spec-ember.md section 2): the environment first, then the XDG defaults."""
import os

from .. import config

DEFAULT_USER = 'caminus-agent'
DEFAULT_BRIDGE = '/srv/caminus-env-bridge'
BRIDGE_GROUP = 'caminus-env'


def user():
    """the second user the agent work runs as"""
    return os.environ.get('CAMINUS_ENV_USER') or DEFAULT_USER


def bridge_group():
    """the group the owner and the agent share the bridge through. A second setup on one machine (its own agent account
    and bridge) needs its own group too: with a shared one, either side could write into the other's bridge."""
    return os.environ.get('CAMINUS_ENV_GROUP') or BRIDGE_GROUP


def key_path():
    """the harness's private key for that user (authorized_keys pins it to 127.0.0.1)"""
    return os.environ.get('CAMINUS_ENV_KEY') or os.path.join(os.path.expanduser('~/.ssh'), 'caminus_env_key')


def bridge_dir():
    """the request/result directory shared by both users through the BRIDGE_GROUP group (770)"""
    return os.environ.get('CAMINUS_ENV_BRIDGE') or DEFAULT_BRIDGE


def audit_path():
    """the harness-side audit log (the agent user cannot write it)"""
    return os.environ.get('CAMINUS_ENV_AUDIT') or os.path.join(config.state_dir(), 'env-audit.jsonl')


def control_path():
    """the SSH ControlMaster socket template"""
    return os.path.join(config.state_dir(), 'env-ssh-%r@%h:%p')


def agent_bin():
    """the ask-root script's path inside the agent user's home"""
    return os.path.join('~', 'bin', 'ask-root')


def robots_path():
    """ROBOTS.md's path inside the agent user's home"""
    return os.path.join('~', 'ROBOTS.md')


def agent_bin_dir():
    return os.path.join('~', 'bin')


def memory_path():
    """the agent's persistent memory file inside its home"""
    return os.path.join('~', 'MEMORY.md')
