from pathlib import Path

"""
Iteratively visit a subtree of the file system.
Use the return value of the directory_callback to maintain something like a stack state.
"""

class FSVisitor:
    def __init__(self, *, directory_callback=None, file_callback=None):
        self.directory_callback = directory_callback
        self.file_callback = file_callback

    def _visit(self, dir_path, extra_data):
        if self.directory_callback:
            retval = self.directory_callback(dir_path, extra_data)
            if retval:
                extra_data = retval
        for child in dir_path.iterdir():
            self.go(child, extra_data)

    def _act(self, file_path, extra_data):
        if self.file_callback:
            self.file_callback(file_path, extra_data)

    def go(self, path: Path, extra_data = None):
        path.resolve()
        if path.is_dir():
            self._visit(path, extra_data)
        else:
            self._act(path, extra_data)
