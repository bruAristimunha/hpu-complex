@rbabukv @Silv3S — code owners, one roadmap question only:

**Is `complex` in SynapseAI's type system on any Intel roadmap? Yes or no?**

Why only you can answer: `synDataType` has no complex entry, and it ships in
the closed graph compiler — the bridge just loads it. Every bridge-side gate
(`tensor_utils.cpp:673`, `create_tensor.cpp:1062`, `kernel_input_checks.cpp:3608`)
is downstream of that one missing entry. No outside patch can fix it.

No GPU has complex hardware — it's two floats and software operators
(PyTorch's own `c10::complex` multiply is `a*c-b*d, a*d+b*c`). What's missing
on Gaudi is one type entry, then three ops (`_fft_r2c/_fft_c2r/_fft_c2c`).
Correct-first is fine; nobody expects cuFFT.

Until then, every downstream library pays per model — Braindecode has
hand-rolled device STFT three times; phase/complex-valued models have no
workaround at all.

Two asks: (1) the yes/no above, dated if yes; (2) if no, say so plainly so
downstream projects can plan. Silence is the only unhelpful answer.
