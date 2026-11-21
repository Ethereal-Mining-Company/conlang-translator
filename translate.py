from src.engine import DynamicTranslator
from src.models import MODEL_REGISTRY, diagnose_all_models, get_tagger
from src.utils import load_config


def _print_diagnostics(text: str, cfg: dict, translator: DynamicTranslator, src: str, tgt: str) -> None:
    diag = diagnose_all_models(text, cfg)
    keys = list(diag.keys())
    if not keys:
        print("No models enabled.")
        return

    print("\n--- MULTI-MODEL DIAGNOSTICS ---")
    for k in keys:
        print(f"  * {diag[k]['display_name']} ({k}): {diag[k]['latency_ms']} ms")

    base_tokens = diag[keys[0]]["tokens"]
    header = f"{'TOKEN':<16}" + "".join(f"{diag[k]['display_name']:<28}" for k in keys) + "STATUS"
    print("\n" + header)
    print("-" * len(header))

    for idx, tok in enumerate(base_tokens):
        word = tok["word"]
        labels = [diag[k]["tokens"][idx]["label"] for k in keys]
        status = "AGREE" if len(set(labels)) == 1 else "DIFF"
        row = f"{word:<16}" + "".join(f"{lbl:<28}" for lbl in labels) + status
        print(row)

    print("\n--- TRANSLATION COMPARISON ---")
    for k in keys:
        pred_fn = lambda s, _toks=diag[k]["tokens"]: _toks
        out = translator.translate(text, src, tgt, pred_fn) if src == "english" else translator.translate(text, src, tgt, lambda _: [])
        print(f"  [{diag[k]['display_name']}]: {out}")
    print("-" * 40 + "\n")


def _print_menu(model_name: str, src: str, tgt: str, diag_mode: bool) -> None:
    print(f"\n=== Conlang Translator | Model: {model_name} | {src.upper()} -> {tgt.upper()} | Diag: {'ON' if diag_mode else 'OFF'} ===")
    print("Commands: /to <lang>, /from <lang>, /swap, /model <name>, /diag, ,/trace /reload, /info, /menu, /help, /quit\n")


def _print_trace(rows: list[dict], final_out: str, tgt_lang: str) -> None:
    print(f"\n--- CONVEYOR BELT TRACE (-> {tgt_lang.upper()}) ---")
    header = f"{'TOKEN':<16} {'ML TAG':<18} {'S0:GUARD':<9} {'S1:IDIOM':<9} {'S2:NAME':<10} {'S3:LEMMA':<14} {'S4:LEXICON':<12} {'S5:AFFIX':<12} {'OUTPUT'}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['token']:<16} {r['tag']:<18} {r['s0']:<9} {r['s1']:<9} {r['s2']:<10} {r['s3']:<14} {r['s4']:<12} {r['s5']:<12} {r['output']}")
    print(f"\n-> {final_out}\n")


def main() -> None:
    cfg = load_config()
    translator = DynamicTranslator(cfg["paths"]["languages_dir"], cfg["paths"]["english_rules"])
    src = cfg["translator"]["default_source"]
    tgt = cfg["translator"]["default_target"]
    active_key = cfg["translator"]["active_model"]
    tagger, model_name = get_tagger(cfg, active_key)
    diag_mode = False
    trace_mode = False

    _print_menu(model_name, src, tgt, diag_mode)
    while True:
        mode_tag = f"{' [DIAG]' if diag_mode else ''}{' [TRACE]' if trace_mode else ''}"
        try:
            raw = input(f"[{src.upper()} -> {tgt.upper()}]{mode_tag} > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not raw:
            continue
        if raw.startswith("/"):
            parts = raw.split(maxsplit=1)
            cmd, arg = parts[0].lower(), (parts[1].strip().lower() if len(parts) > 1 else "")
            if cmd in ("/quit", "/exit"):
                break
            elif cmd in ("/menu", "/help"):
                _print_menu(model_name, src, tgt, diag_mode)
            elif cmd == "/diag":
                diag_mode = not diag_mode
                print(f"Diagnostics Mode: {'ON' if diag_mode else 'OFF'}")
            elif cmd == "/trace":
                trace_mode = not trace_mode
                print(f"Trace Mode: {'ON' if trace_mode else 'OFF'}")
            elif cmd == "/swap":
                src, tgt = tgt, src
            elif cmd == "/to" and arg in translator.available_languages():
                tgt = arg
            elif cmd == "/from" and arg in translator.available_languages():
                src = arg
            elif cmd == "/model" and arg in MODEL_REGISTRY:
                active_key = arg
                tagger, model_name = get_tagger(cfg, active_key)
                print(f"Switched model to {model_name}")
            elif cmd == "/reload":
                print("Languages:", translator.reload_languages())
            elif cmd == "/info":
                print("Languages:", translator.available_languages(), "| Models:", list(MODEL_REGISTRY.keys()))
            else:
                print("Unknown command or invalid argument.")
            continue

        if diag_mode:
            _print_diagnostics(raw, cfg, translator, src, tgt)

        if trace_mode:
            bridge = translator._con_to_eng(raw, src) if src != "english" else raw
            if tgt == "english":
                print(f"-> {bridge}\n")
            else:
                res = translator._eng_to_con(bridge, tgt, tagger, return_trace=True)
                if isinstance(res, tuple):
                    rows, final_out = res
                    _print_trace(rows, final_out, tgt)
        elif not diag_mode:
            print(f"-> {translator.translate(raw, src, tgt, tagger)}\n")


if __name__ == "__main__":
    main()
