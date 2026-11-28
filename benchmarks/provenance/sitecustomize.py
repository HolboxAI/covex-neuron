"""Per-process provenance for benchmark arms. Active only when COVEX_PROVENANCE_DIR is set (serve.sh puts this directory
first on PYTHONPATH). Every Python process of an arm - API server, engine core, each TP worker - writes one JSON record:
where `vllm_neuron` resolves from on this process's import path, the installed distribution, the interpreter, and the
compile-cache roots. Recorded at the real import by a meta-path observer that returns the same spec the import
system would have used, so behaviour is unchanged.
If another sitecustomize exists further down the path, it is executed after recording."""
import json, os, sys


def _write(spec):
    d = os.environ.get("COVEX_PROVENANCE_DIR")
    try:
        import importlib.metadata as md
        try:
            dist = md.distribution("vllm-neuron")
            dist_info = {"version": dist.version, "location": str(dist.locate_file(""))}
        except md.PackageNotFoundError:
            dist_info = None
        rec = {"pid": os.getpid(), "ppid": os.getppid(), "arm": os.environ.get("COVEX_ARM"), "executable": sys.executable,
               "prefix": sys.prefix, "cwd": os.getcwd(), "argv0": (sys.argv or [""])[0],
               "vllm_neuron_origin": spec.origin if spec else None, "distribution": dist_info,
               "cache_roots": {k: os.environ.get(k) for k in ("NEURON_LIBTORCH_CACHE_ROOT", "VLLM_CACHE_ROOT",
                                                               "TORCH_NEURONX_NEFF_CACHE_DIR")},
               "covex_flags": sorted(k for k in os.environ if k.startswith("COVEX_") and k not in
                                     ("COVEX_ARM", "COVEX_PROVENANCE_DIR"))}
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{rec['arm']}.{rec['pid']}.json"), "w") as f:
            json.dump(rec, f)
    except Exception as e:  # never break the process being observed
        sys.stderr.write(f"covex provenance: {e}\n")


class _RecordVllmNeuron:
    """Meta-path observer: when `vllm_neuron` is actually imported, ask the remaining finders (the ones the import
    system would use, with the final sys.path including the script/cwd entry) and record the spec they return.
    Resolving at sitecustomize time would miss the cwd entry that `python -m` adds later."""
    def find_spec(self, name, path=None, target=None):
        kernel = os.environ.get("SIX_KERNEL_EVIDENCE") == "1" and name.startswith(
            ("nkilib.", "nkilib_src.nkilib.", "neuronxcc.private_nkl.nkilib.")
        )
        if name != "vllm_neuron" and not kernel:
            return None
        for f in sys.meta_path:
            if f is self or not hasattr(f, "find_spec"):
                continue
            spec = f.find_spec(name, path, target)
            if spec is not None:
                if kernel:
                    try:
                        from kernel_files import record
                        record(os.environ["COVEX_PROVENANCE_DIR"], os.environ["COVEX_ARM"], os.getpid(),
                               name, spec.origin, os.environ.get("COVEX_NKILIB_FA_TILE_SIZE"))
                    except (OSError, KeyError, TypeError) as exc:
                        sys.stderr.write(f"campaign kernel provenance: {exc}\n")
                else:
                    _write(spec)
                return spec
        if not kernel:
            _write(None)
        return None


if os.environ.get("COVEX_PROVENANCE_DIR"):
    sys.meta_path.insert(0, _RecordVllmNeuron())
# chain to the next sitecustomize on the path, if any (this module shadows it)
_here = os.path.dirname(os.path.abspath(__file__))
for _p in sys.path:
    if _p and os.path.abspath(_p) != _here and os.path.isfile(os.path.join(_p, "sitecustomize.py")):
        exec(compile(open(os.path.join(_p, "sitecustomize.py")).read(), os.path.join(_p, "sitecustomize.py"), "exec"))
        break
