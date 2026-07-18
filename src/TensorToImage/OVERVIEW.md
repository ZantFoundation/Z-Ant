# TensorToImage — Panoramica del modulo

Modulo per convertire **serie temporali 1D** in **rappresentazioni 2D** (matrici n×n) adatte all'addestramento di CNN, e per esportarle come immagini BMP. Implementa tre trasformazioni classiche dello stato dell'arte (GASF, GADF, MTF) e un'utility "compound" che le impacchetta come canali RGB di un singolo tensore.

Tutto il codice segue il pattern **lean / standard**:
- `lean_*` → core matematico a **zero allocazioni**, tutti i buffer sono di proprietà del chiamante
- `*` (wrapper standard) → alloca internamente, valida l'input, restituisce uno slice che il chiamante deve liberare

---

## Struttura dei file

```
src/TensorToImage/
├── mod.zig              → entry point: ri-esporta tutti i sotto-moduli
├── gaf_utils.zig        → utility condivise GASF/GADF (normalizzazione)
├── gasf.zig             → Gramian Angular Summation Field
├── gadf.zig             → Gramian Angular Difference Field
├── mtf/
│   ├── mod.zig          → entry point sotto-modulo MTF
│   ├── mtf_utils.zig    → quantile bin + matrice di transizione di Markov
│   └── mtf.zig          → Markov Transition Field
├── compound.zig         → impacchetta GASF+GADF+MTF in un tensore CHW [3,n,n]
├── colormap.zig         → mappature scalare→RGB (Grayscale, Viridis, Jet)
└── matrixToBmp.zig      → serializzazione BMP (singola matrice o tile orizzontale)
```

---

## Dettaglio per file

### [mod.zig](src/TensorToImage/mod.zig)
Entry point del modulo. Ri-esporta `mtf`, `gasf`, `gadf`, `gaf_utils`, `colormap`, `matrixToBmp`, `compound`.

---

### [gaf_utils.zig](src/TensorToImage/gaf_utils.zig)
Utility condivise tra GASF e GADF.

- `NormRange` — enum: `ZeroToOne` (mappa biettiva, φ ∈ [0, π/2]) oppure `MinusOneToOne` (range angolare pieno φ ∈ [0, π]).
- `normalize(input, output, range_type)` — riscala min-max, gestisce serie costanti (output a zero), clampa per evitare errori floating-point.
- `NormalizeError` — `EmptyInput`, `LengthMismatch`.

---

### [gasf.zig](src/TensorToImage/gasf.zig)
**Gramian Angular Summation Field**: matrice n×n con `G[i][j] = xᵢ·xⱼ − √(1−xᵢ²)·√(1−xⱼ²)` (forma algebrica di `cos(φᵢ + φⱼ)`).

- `lean_gasf(input, cosines_buffer, output)` — pre-calcola `√(1−xᵢ²)` in O(n), poi riempie la matrice in O(n²). Zero allocazioni.
- `gasf(allocator, input, norm)` — wrapper che normalizza e alloca i buffer; ritorna `[]f32` di lunghezza n*n (row-major).
- `GasfError` — `InputTooShort`, `OutputSizeMismatch`.

---

### [gadf.zig](src/TensorToImage/gadf.zig)
**Gramian Angular Difference Field**: matrice n×n con `G[i][j] = √(1−xᵢ²)·xⱼ − xᵢ·√(1−xⱼ²)` (forma algebrica di `sin(φᵢ − φⱼ)`).

- `lean_gadf(x_tilde, sines_buffer, gadf_out)` — stessa struttura del GASF (pre-calcolo seni in O(n) + riempimento O(n²)).
- `gadf(allocator, input, norm)` — wrapper standard.
- `GadfError` — `InputTooShort`.

Note: serie costanti producono GADF identicamente nullo (corretto matematicamente, ma visivamente indistinguibile da serie con differenze che si cancellano).

---

### [mtf/mtf_utils.zig](src/TensorToImage/mtf/mtf_utils.zig)
- `quantileBins(input, q, sorted_buf, bins_out)` — assegna ogni campione a un bin in `[0, q-1]` usando q-1 bordi presi da indici `k*n/q` della copia ordinata.
- `transitionMatrix(bins, q, matrix_out)` — costruisce la matrice Q×Q row-stochastic delle transizioni di Markov a un passo. Le righe a somma zero restano zero.

### [mtf/mtf.zig](src/TensorToImage/mtf/mtf.zig)
**Markov Transition Field**: `output[i][j] = W[bins[i]][bins[j]]`, dove W è la matrice di transizione tra bin quantilici.

- `lean_mtf(input, q, sorted_buf, bins_buf, matrix_buf, output)` — core zero-alloc.
- `mtf(allocator, input, q)` — wrapper standard.
- `MtfError` — `InputTooShort`, `InvalidBins`.

### [mtf/mod.zig](src/TensorToImage/mtf/mod.zig)
Entry point del sotto-modulo: esporta `mtf` e `mtf_utils`.

---

### [compound.zig](src/TensorToImage/compound.zig)
Combina le tre trasformazioni in un singolo tensore CHW `[3, n, n]` adatto a CNN RGB.

Layout output (lunghezza `3*n*n`):
- canale 0 → GASF, riscalato da [−1,1] a [0,1] via `(x+1)/2`
- canale 1 → GADF, riscalato da [−1,1] a [0,1] via `(x+1)/2`
- canale 2 → MTF, già in [0,1] (copiato as-is)

API:
- `CompoundScratch` — struct che possiede tutti i buffer intermedi (`norm_buf`, `cosines_buf`, `sines_buf`, `sorted_buf`, `bins_buf`, `matrix_buf`). Si alloca una volta e si riusa per molte serie. Metodi `init` / `deinit`.
- `lean_compound(input, norm, q, *scratch, output)` — core zero-alloc per una singola serie. Normalizza una volta sola; scrive i tre canali back-to-back.
- `toRGBImageF32(allocator, input, norm, q)` — wrapper allocante per una sola serie.
- `batchToRGBImageF32(allocator, inputs, norm, q)` — processa un batch di B serie tutte di lunghezza N e restituisce un buffer NCHW `[B, 3, N, N]`. Lo scratch è allocato **una sola volta** fuori dal loop.

Nota architetturale (vedi commento in cima al file): l'output è un flat `[]f32` invece di `Tensor(f32)` per evitare un ciclo di import (`compound → tensor → zant → TensorToImage → compound`). La conversione si fa al call site con `Tensor(f32).fromArray(...)`.

---

### [colormap.zig](src/TensorToImage/colormap.zig)
Conversione scalare→RGB per la visualizzazione.

- `Colormap` — enum: `Grayscale`, `Viridis`, `Jet`.
- `mapToRgb(val, cmap)` — mappa `val ∈ [-1, 1]` (clampato) in un triplet `[R, G, B]`. Internamente normalizza a `t = (val+1)/2` e fa interpolazione lineare tra "stops" predefiniti per Viridis e Jet.

---

### [matrixToBmp.zig](src/TensorToImage/matrixToBmp.zig)
Serializzazione BMP a 24-bit, top-down, BGR, con padding di riga a 4 byte.

- `matrixToBmp(matrix, n, path, cmap)` — scrive una singola matrice n×n.
- `matricesToBmp(matrices, n, path, cmap)` — affianca orizzontalmente più matrici n×n separate da un bordo bianco di 4 px (utile per confrontare GASF | GADF | MTF in un unico file).
- Funzione interna `writeBmp` — scrive `BITMAPFILEHEADER` (14 B) + `BITMAPINFOHEADER` (40 B) + dati pixel applicando la colormap scelta.

---

## Pipeline tipiche

### 1. Singola trasformazione → BMP
```
serie f32 ──► gaf_utils.normalize ──► gasf.lean_gasf / gadf.lean_gadf / mtf.lean_mtf
                                              │
                                              ▼
                                  matrice n×n in [-1,1] o [0,1]
                                              │
                                              ▼
                              matrixToBmp.matrixToBmp(matrix, n, path, cmap)
                                              │
                                              ▼
                       colormap.mapToRgb per ogni pixel  →  file BMP
```

### 2. Tensore RGB per CNN (compound)
```
serie f32 ──► CompoundScratch.init(n, q)
                       │
                       ▼
            compound.lean_compound (per ogni serie)
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
    canale 0       canale 1       canale 2
   GASF [0,1]     GADF [0,1]      MTF [0,1]
        └──────────────┼──────────────┘
                       ▼
        flat []f32 in layout CHW [3, n, n]
                       │
                       ▼
        (opzionale) Tensor(f32).fromArray(...)
```

### 3. Batch per training
```
B serie tutte di lunghezza N
            │
            ▼
batchToRGBImageF32(allocator, inputs, norm, q)
   ├── alloca scratch UNA volta
   ├── alloca output [B, 3, N, N]
   └── loop su inputs → lean_compound nel sotto-buffer di ogni serie
            │
            ▼
flat []f32 in layout NCHW pronto per Tensor(f32) / CNN
```

---

## Convenzioni ricorrenti

- **Layout matrici**: row-major, `output[i * n + j]`.
- **Layout tensori**: CHW per la singola serie, NCHW per i batch (`data[((b * 3 + c) * N + r) * N + col]`).
- **Range di valori**: GASF/GADF nativamente in `[-1, 1]`; MTF nativamente in `[0, 1]`. Il modulo `compound` allinea tutto a `[0, 1]` per uso CNN.
- **Ownership**: i wrapper `*` allocano e restituiscono uno slice che il chiamante deve liberare; i `lean_*` non allocano nulla.
- **Validazione**: gli errori `InputTooShort` / `InvalidBins` sono sollevati dai wrapper standard; i `lean_*` usano solo `std.debug.assert`.
