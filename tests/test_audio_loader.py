from pathlib import Path
import numpy as np

from src.data.audio_loader import load_audio


DATA_ROOT = Path(r"D:\SIH\AI_ANC\data\raw\dataset")

TEST_FILES = {
    "16 kHz mono": DATA_ROOT / "clean" / "1098-133695-0000.wav",
    "44.1 kHz stereo": DATA_ROOT / "noise" / "impulsive" / "1 (56).wav",
    "48 kHz stereo": DATA_ROOT / "noise" / "impulsive" / "1 (31).wav",
}


def main():
    print("=" * 60)
    print("AUDIO LOADER TEST")
    print("=" * 60)

    all_passed = True

    for name, path in TEST_FILES.items():

        print(f"\n[{name}]")
        print(f"File: {path}")

        if not path.exists():
            print("FAIL: file does not exist")
            all_passed = False
            continue

        try:
            audio = load_audio(path, target_sr=16000)

            print(f"Shape : {audio.shape}")
            print(f"Dtype : {audio.dtype}")
            print(f"Samples: {len(audio)}")
            print(f"Duration: {len(audio) / 16000:.4f} sec")
            print(f"Min   : {np.min(audio):.6f}")
            print(f"Max   : {np.max(audio):.6f}")

            errors = []

            if not isinstance(audio, np.ndarray):
                errors.append("Output is not numpy.ndarray")

            if audio.dtype != np.float32:
                errors.append(
                    f"Expected float32, got {audio.dtype}"
                )

            if audio.ndim != 1:
                errors.append(
                    f"Expected 1-D mono output, got {audio.shape}"
                )

            if not np.all(np.isfinite(audio)):
                errors.append("Output contains NaN or Inf")

            if len(audio) == 0:
                errors.append("Output is empty")

            if errors:
                print("FAIL")
                for error in errors:
                    print(f"  - {error}")
                all_passed = False
            else:
                print("PASS")

        except Exception as exc:
            print(f"FAIL: {type(exc).__name__}: {exc}")
            all_passed = False

    print("\n" + "=" * 60)
    print(
        "FINAL RESULT:",
        "PASS" if all_passed else "FAIL"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()