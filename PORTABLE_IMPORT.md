# Large image sequence import

This source build fixes the image import bottleneck in the supplied pyALDIC
1.0.5 source. The changes have no computer name, user path, image count, or
resolution specific branch. Windows, macOS, and Linux use the same Python,
NumPy, Pillow, OpenCV, and Qt implementation.

## Use this source build

Use Python 3.10 or later. From this extracted directory, preferably in a
dedicated virtual environment, run:

```console
python -m pip install .
python -m al_dic
```

This starts the modified Python installation. An existing desktop app bundle
or installer is a separate installation and does not acquire these changes.
The original repository directory and installed application were not edited.
The archive includes the complete code, runtime assets, examples, tests,
and `large_sequence_changes.patch`. Large demonstration videos and GIFs
remain available in the upstream repository.

## Changes

1. The image list inserts lightweight tree items in one batch. A Qt delegate
   paints visible ROI actions and routes clicks to the selected frame. There
   is no QPushButton allocation for every image. Selection, per frame ROI,
   context menu actions, and natural sorting remain available.
2. GUI import discovers files, checks headers, and decodes the first preview
   on one background worker. Progress and cancellation are visible. Only
   the newest folder request can update the interface. Repeated requests
   reuse one worker slot instead of starting an unbounded number of threads.
3. Cancellation and failures preserve the previous sequence. Closing the
   main window cancels import and lets the event loop run until the worker
   exits. Native filesystem reads cannot be forcibly interrupted; a stalled
   remote read must return before cancellation can finish.
4. Grayscale, RGB preview, and streaming computation caches have both item
   and byte limits. With psutil installed, each GUI cache uses at most 2%
   of currently available RAM, capped at 128 MiB; the streaming cache uses
   at most 4%, capped at 256 MiB. Without psutil, each cache conservatively
   uses at most 64 MiB. A frame larger than its budget is returned without
   retaining it in the cache. Eviction affects speed, not pixel values.
5. Reference frames that inherit one ROI share one immutable uint8 snapshot.
   Unmasked frames share one immutable all ones mask. Custom masks keep
   their original values and are copied once per distinct source array.
6. Changing sort order does not repeat the header scan. Deleting the whole
   sequence also clears the controller's source list.

The existing synchronous `load_folder()` API remains available for scripts
and session helpers. GUI browse and drop use `load_folder_async()`.

## Reference to pyALDIC 3D

The reference project's public README and architecture record describe lazy
frames and masks, bounded LRU caches, and shared all ones masks:

https://github.com/zachtong/pyALDIC-3D/blob/main/README.md

https://github.com/zachtong/pyALDIC-3D/blob/main/docs/architecture/00_INDEX.md

This implementation adopts those documented policies, with byte limits added
for varying image resolutions. The relevant implementation source page could
not be retrieved in this environment, so this is an independent implementation,
not a claim that its code was copied verbatim. Its stereo solver memory
projection is not reused as an estimate for this different 2D solver.

## Validation

25 behavioral regression tests passed locally with Python 3.12 and Qt 6.11.1
on macOS ARM64 in offscreen mode. They cover worker responsiveness, cancellation,
queued stale results, repeated requests, file drops, Unicode paths, malformed
images, size mismatch reporting, window close, ROI interactions, bounded cache
eviction, and shared mask isolation. A synthetic incremental DIC run preserves
displacement, cumulative displacement, and deformation gradients exactly when
using the streaming cache and shared masks. Tests also verify exact normalized
pixels for uint8 and uint16 inputs across cache budgets, including zero.

```console
python -m unittest tests.test_large_sequence_support -v
```

The full existing pytest suite was not run because pytest was unavailable in
the local test environment. The new focused tests use standard library unittest.
The existing grayscale cache regression has been updated for the new bounded
cache contract. A separate CI workflow runs the focused tests on Windows,
macOS, and Linux with Python 3.10 and 3.12. That remote matrix is configured,
but was not executed here; cross platform execution is not claimed as verified.

Measured local checks:

| Check | Result |
| --- | --- |
| Original list, 3000 rows, no image reads | 48.87 seconds |
| Modified list, 10000 rows including event processing | 0.072 seconds |
| Actual ROI button widgets in modified list | 0 |
| Full window import, 10000 paths to 4053 × 4053 uint16 TIFF | 0.749 seconds |
| GUI timer callbacks during that import | 66 |
| Largest observed timer gap | 0.093 seconds |
| Inherited ROI masks, 10000 frames at 4053 × 4053 | 2 buffers, 32853618 bytes |

The full resolution import fixture used 10000 hard links to one uncompressed
TIFF, avoiding hundreds of gigabytes of duplicated test data. It verifies the
file count, full resolution decode, signal path, and list behavior, but does
not measure storage throughput for 10000 different experimental images.
Times are observations from this test environment, not universal guarantees.

## Scope

This change bounds import and image cache overhead and removes repeated
inherited ROI allocation. It does not make an arbitrarily large solver run
fit every computer. The solver still needs its current frames and working
arrays. Per frame mesh, displacement, and strain results still accumulate
with the number of nodes and frames. No image downsampling, precision
reduction, or automatic change of the DIC parameters is introduced.

The newly added import status messages use Qt translation calls and fall back
to English until the translation catalogs are updated and rebuilt.
