"""Kernel sources used by the campaign, identified by their Run 1 or tested candidate bytes."""
import hashlib
import json
from pathlib import Path

QWEN_STOCK = {
    "core.attention.attention_tkg": "850762acdde0159ded51586866d27ed8c68d7298ca60266420142b5e104fa9c2",
    "core.attention.attention_tkg_utils": "c522e3c8731af02bd1cc81c7f33a915c1a7762b6202f04edc990df384d012f2a",
}
QWEN_4K = {
    "core.attention.attention_tkg": "154ffa945f5d3672a384b4361d77e30743ba22d8dbd27cf22d463aa830c3ff76",
    "core.attention.attention_tkg_utils": "428c26cb405d6fa3f56a57096036bbb04cb183610deb36ae2c5cf0d363ccb55c",
}
LLAMA_PAIR = {
    "core.attention.attention_tkg": "ad0485d7dea09e3da3055ccf49a63b18015cf87a0743daf7265744da192ea6c9",
    "experimental.transformer.transformer_tkg": "d56cde14c96e8b34f0698ba36452057003f65ce32716b86c3760815fadfde051",
    "experimental.transformer.attention_block_tkg": "79611627134739a6df6ead1992f910739cf9ad4b1fe1c802d04e2d84e2114803",
    "core.mlp.mlp_tkg.mlp_tkg": "384a0478a799237ab7433740707e91bcefc8f81295cc76e4603ef82b74a77b78",
    "core.subkernels.rmsnorm_tkg": "621882d1455b9baeec18a58ebcb6efddc8b2b44563ae0aa7d0ad494937aee9b6",
}
BY_ARM = {"q32-U": QWEN_STOCK, "q32-Ut": QWEN_STOCK, "q32-F4": QWEN_4K, "llama-K2": LLAMA_PAIR}


def record(folder, arm, pid, name, origin, tile):
    for suffix in BY_ARM.get(arm, {}):
        if name.endswith(".nkilib." + suffix) or name == "nkilib." + suffix:
            target = Path(folder) / "kernels" / arm / str(pid)
            target.mkdir(parents=True, exist_ok=True)
            data = {"origin": origin, "sha256": hashlib.sha256(Path(origin).read_bytes()).hexdigest(),
                    "tile": tile}
            (target / (suffix + ".json")).write_text(json.dumps(data))


def verify_sources(folder, arm, roles, tp):
    for rank in range(tp):
        pid = roles.get(f"TP{rank}")
        for name, digest in BY_ARM.get(arm, {}).items():
            path = Path(folder) / "kernels" / arm / str(pid) / (name + ".json")
            try:
                row = json.loads(path.read_text())
                if row["sha256"] != digest or row["tile"] != ("4096" if arm == "q32-F4" else None):
                    raise ValueError("unexpected source or tile setting")
            except (OSError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{arm} TP{rank} {name}: {exc}") from exc
