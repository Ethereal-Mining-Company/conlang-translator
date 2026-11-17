from src.engine import DynamicTranslator
from src.models import MODEL_REGISTRY, get_tagger
from src.utils import load_config


def main():
    cfg = load_config()
    tagger, active_display = get_tagger(cfg)
    translator = DynamicTranslator(cfg["paths"]["languages_dir"])
    langs = translator.available_languages()
    src = cfg["translator"]["default_source"].lower()
    tgt = cfg["translator"]["default_target"].lower()
    if tgt not in langs:
        tgt = langs[-1]
    avail_models = [k for k, v in cfg["models"].items() if v.get("enabled", True) and k in MODEL_REGISTRY]

    print(f"CONLANG TRANSLATOR | Model: {active_display} ({', '.join(avail_models)})")
    print(f"Languages: {', '.join(langs)} | Active: [{src.upper()} -> {tgt.upper()}]")
    print("Commands: /to <lang> | /from <lang> | /swap | /model <key> | /info | /reload | /quit")

    while True:
        try:
            inp = input(f"\n[{src.upper()} -> {tgt.upper()}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not inp:
            continue
        if inp.startswith("/"):
            parts = inp.split()
            cmd = parts[0].lower()
            if cmd in ("/quit", "/exit"):
                break
            elif cmd == "/swap":
                src, tgt = tgt, src
                print(f"[{src.upper()} -> {tgt.upper()}]")
            elif cmd == "/to" and len(parts) > 1 and parts[1].lower() in translator.available_languages():
                tgt = parts[1].lower()
                print(f"Target: [{tgt.upper()}]")
            elif cmd == "/from" and len(parts) > 1 and parts[1].lower() in translator.available_languages():
                src = parts[1].lower()
                print(f"Source: [{src.upper()}]")
            elif cmd == "/model" and len(parts) > 1:
                try:
                    tagger, active_display = get_tagger(cfg, parts[1].lower())
                    print(f"Model: {active_display}")
                except Exception as e:
                    print(e)
            elif cmd == "/info":
                for k in (src, tgt):
                    if k in translator.profiles:
                        p = translator.profiles[k]
                        print(f"[{k.upper()}] Files: {p.loaded_sources} | Order: {p.word_order} | Affixes: {p.affixes} | Words: {len(p.eng_fallback)}")
            elif cmd == "/reload":
                print("Languages:", ", ".join(["english"] + translator.reload_languages()))
            else:
                print("Invalid command or unknown language.")
            continue
        print(f"-> {translator.translate(inp, src, tgt, tagger)}")


if __name__ == "__main__":
    main()