import json
from pathlib import Path
from typing import Any, Dict, List
import sys
import numpy as np


def load_old_record(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def first_tensor_from_dict(d: Dict[str, Any]) -> List[Any]:
    """
    Prende il primo tensore da un dict tipo:
      { "QLinearConv_0_input": [[[[...]]]] }
    e lo restituisce come lista flat.
    """
    if not d:
        raise ValueError("Nessun tensore trovato nel dizionario")

    # prende il primo valore (ignoriamo il nome della chiave)
    _, value = next(iter(d.items()))
    arr = np.array(value)
    flat = arr.flatten()
    # Convertiamo in normali int di Python
    return [int(x) for x in flat]


def convert_old_json_to_zant_user_tests(
    old_json_path: Path,
    model_name: str,
    out_path: Path,
) -> None:
    """
    Converte un file JSON in formato "benchmark" (quello che generi ora)
    nel formato atteso da Zant:

    [
      {
        "name": "...",
        "type": "exact",
        "input": [...],
        "output": [...],
        "expected_class": 0
      }
    ]
    """
    record = load_old_record(old_json_path)

    # estrai e appiattisci input/output
    inputs_dict = record.get("inputs", {})
    outputs_dict = record.get("outputs", {})

    if not inputs_dict:
        raise ValueError("Campo 'inputs' mancante o vuoto nel JSON sorgente")
    if not outputs_dict:
        raise ValueError("Campo 'outputs' mancante o vuoto nel JSON sorgente")

    input_flat = first_tensor_from_dict(inputs_dict)
    output_flat = first_tensor_from_dict(outputs_dict)

    user_test = {
        "name": model_name,
        "type": "exact",
        "input": input_flat,
        "output": output_flat,
        "expected_class": 0,
    }

    tests_array = [user_test]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(tests_array, f, indent=2)

    print(f"Creato file user tests Zant in: {out_path}")


# ...existing code...

def main():
    """
    Uso:
      python convert_to_zant_user_test.py <input_json>

    Scrive un file `user_tests.json` nella stessa directory di <input_json>.
    Il nome modello viene dedotto dal nome del file, ad es.:
      QLinearConv_0_metadata.json -> model_name = QLinearConv_0
    """
    if len(sys.argv) < 2:
        print("Uso: python convert_to_zant_user_test.py <input_json>")
        sys.exit(1)

    old_json = Path(sys.argv[1])
    if not old_json.is_file():
        print(f"Errore: file JSON sorgente non trovato: {old_json}")
        sys.exit(1)

    # deduci il modello dal nome file
    stem = old_json.stem  # es. 'QLinearConv_0_metadata'
    if stem.endswith("_metadata"):
        model_name = stem[: -len("_metadata")]
    else:
        model_name = stem

    # output: user_tests.json accanto al file di input
    out_json = old_json.parent / "user_tests.json"

    convert_old_json_to_zant_user_tests(old_json, model_name, out_json)


if __name__ == "__main__":
    main()
