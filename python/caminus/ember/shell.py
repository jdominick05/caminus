"""Quoting an argv into one remote shell command: the script builders' shared rule.

`shlex.quote` protects every argument, but it also single-quotes a leading tilde, so `~/homes/NAME/run.sh` stopped
expanding and the remote shell looked for a literal `~` directory (exit 127). The tilde is expanded at run time
instead: `"$HOME"` stays unquoted and the rest of the path is quoted, so any name is safe and the path is the agent
user's own home.
"""
import shlex


def quote_arg(a):
    """one argv element: a leading ~/ (or a bare ~) becomes the remote $HOME"""
    a = str(a)
    if a == '~':
        return '"$HOME"'
    if a.startswith('~/'):
        return '"$HOME"' + shlex.quote(a[1:])
    return shlex.quote(a)


def quote_argv(argv):
    """the argv as one shell command line, quoted for a remote bash"""
    return ' '.join(quote_arg(a) for a in argv)
