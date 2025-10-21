import os
import subprocess
from pathlib import Path


def pick_java_home(preferred=None, version=None):
    if preferred and Path(preferred).exists():
        return preferred

    if version:
        try:
            home = subprocess.check_output(
                ["/usr/libexec/java_home", "-v", str(version)], text=True
            ).strip()
            if Path(home).exists():
                return home
        except Exception:
            pass

    candidates = [
        "/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home",
        "/opt/homebrew/opt/openjdk",
    ]
    for c in candidates:
        if Path(c).exists():
            return c

    raise RuntimeError("JDK HOME is not found")


JAVA_HOME = pick_java_home(None)
os.environ["JAVA_HOME"] = JAVA_HOME
os.environ["PATH"] = (
    str(Path(JAVA_HOME) / "bin") + os.pathsep + os.environ.get("PATH", "")
)

print("JAVA_HOME =", os.environ["JAVA_HOME"])

try:
    result = subprocess.run(["java", "-version"], capture_output=True, text=True)
    print(result.stderr if result.stderr else result.stdout)
except FileNotFoundError:
    print("⚠️ Java not found — check if the path is correct or JAVA_HOME is set.")


import scyjava as sj

from scyjava import jimport

System = jimport("java.lang.System")
print(System.getProperty("java.version"))

version_digits = sj.jvm_version()
print(version_digits)
import imagej

ij = imagej.init("sc.fiji:fiji", headless=True)
commands = [c for c in ij.command().getCommands().keySet() if "Stitch" in c]
print(commands)
