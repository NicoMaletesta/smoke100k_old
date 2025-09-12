#!/usr/bin/env python3
"""
Lancia TensorBoard puntando a una directory contenente i log (add_scalar, add_image, ecc).
Esempio:
    python tools/launch_tensorboard.py --logdir /path/to/logs --port 6007
"""

import argparse
import os
import subprocess
import time
import webbrowser


def launch_tensorboard(logdir: str, port: int = 6006):
    """
    Avvia TensorBoard su una porta specificata e apre il browser.
    """
    if not os.path.isdir(logdir):
        raise FileNotFoundError(f"La directory '{logdir}' non esiste.")

    print(f"✨ Avvio TensorBoard su logdir: {logdir} | porta: {port}")
    url = f"http://localhost:{port}"

    # Avvia TensorBoard come processo secondario
    cmd = ["tensorboard", f"--logdir={logdir}", f"--port={str(port)}"]
    process = subprocess.Popen(cmd)

    print(f"🌐 Apri il browser su: {url}")
    time.sleep(2)  # piccolo delay per il boot
    webbrowser.open(url)

    try:
        process.wait()  # resta attivo fino a chiusura manuale
    except KeyboardInterrupt:
        print("\n🛑 Interrotto da tastiera, chiusura TensorBoard.")
        process.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avvia TensorBoard su una directory di log.")
    parser.add_argument("--logdir", type=str,
                        default=r"/home/nicola/Documenti/smokedetector_1109/logs/tb/20250911-223931",
                        help="Percorso alla directory dei log (.tfevents.*)")
    parser.add_argument("--port", type=int, default=6006,
                        help="Porta (default: 6006)")
    args = parser.parse_args()

    launch_tensorboard(logdir=args.logdir, port=args.port)