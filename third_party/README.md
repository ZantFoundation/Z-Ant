# Third-Party Dependencies

This folder is reserved for SDKs and vendor libraries that are not distributed with Z-Ant by default.

## CMSIS-NN (optional)

The STM32 N6 accelerator backend can take advantage of Arm's CMSIS-NN optimized kernels. Fetch the sources with:

```bash
./scripts/fetch_cmsis_nn.sh
```

This downloads the CMSIS-NN repository into `third_party/CMSIS-NN`. The build system automatically picks it up when you pass
`-Dstm32n6_accel=true` and do not supply an explicit `-Dstm32n6_cmsis_path`.

For offline environments, set `CMSIS_NN_ARCHIVE=/absolute/path/to/CMSIS-NN-main.zip` to install from a pre-downloaded archive.

Set `CMSIS_NN_REPO` or `CMSIS_NN_REF` before running the script if you need to use a custom mirror or a specific release tag.

You can update the checkout later by re-running the same script.

## Ethos-U Driver (optional)

The Ethos-U integration stubs can link against Arm's reference driver headers. Fetch them with:

```bash
./scripts/fetch_ethos_u_driver.sh
```

This installs the sources into `third_party/ethos-u-driver`, which the build picks up automatically when you enable
`-Dstm32n6_use_ethos=true` and do not pass a manual `-Dstm32n6_ethos_path` override.
