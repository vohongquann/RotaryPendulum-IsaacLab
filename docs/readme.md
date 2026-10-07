# docs

| Folder | What it holds |
|---|---|
| [media/](media/readme.md) | the video, GIF and charts shown in the README and the guide, with the command that made each one |
| `hardware/` | datasheets the constants come from (see below) |

The documentation itself is [guide/](../guide/readme.md); every value taken from these sources is listed with its source in
[guide/01_pendulum.md](../guide/01_pendulum.md).

## Sources

| File | Used for |
|---|---|
| `hardware/qube-servo2/ni_qube_servo_2_for_myrio_datasheet.pdf` | nominal voltage 18 V, 0.54 A, 4050 RPM no load; encoder 512 counts/rev non-quadrature (guide 1.2) |

Sources that are web pages or code, not stored here:

| Source | Used for |
|---|---|
| [vision-based-furuta-pendulum](https://github.com/Data-Science-in-Mechanical-Engineering/vision-based-furuta-pendulum), `gym_brt/quanser/qube_simulator.py` | $R$, $k_t$, $k_e$ (datasheet values in its comments, identified values in its code), $D_r$, $D_p$, arm and pendulum lengths, the ±18 V limit of the hardware wrapper (guide 1.2) |
| Quanser QUBE-Servo 2 workbook (not stored: take it from your Quanser licence) | $L$ = 1.16 mH and $J_m$ = 4.0e-6 as quoted by search results, arm 0.095 kg, pendulum 0.024 kg. **Read the PDF and confirm $L$** |
| QUBE-Servo lab sheet, Missouri S&T (`ece.mst.edu`, Experiment 7) | $J_m$ = 4.0e-6; lists another revision (6.3 Ω, 0.036 N m/A, 0.85 mH), used only to warn that revisions differ |
