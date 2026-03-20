import subprocess
import sys

SRC_PATHS = ["src", "devtools", "scripts"]


def main() -> int:
    errcount = 0
    errcount += run(["ruff", "check", "--fix", *SRC_PATHS])
    errcount += run(["ruff", "format", *SRC_PATHS])
    errcount += run(["basedpyright", *SRC_PATHS])

    if errcount:
        print(f"\n✘ Lint failed with {errcount} error(s).")
    else:
        print("\n✔︎ Lint passed.")

    return errcount


def run(cmd: list[str]) -> int:
    print(f"\n>> {' '.join(cmd)}")
    try:
        subprocess.run(cmd, text=True, check=True)
    except subprocess.CalledProcessError:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
