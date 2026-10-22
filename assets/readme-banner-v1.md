# README Banner v1

Asset: `assets/readme-banner-v1.jpg` (1408x469, 3:1, 236 KB)

Tool/model: xAI Grok CLI, built-in `image_gen` tool, plus local compositing.

Part of a shared visual identity across the openfluids repositories —
`dynachaos`, `fftkit`, `chaos-atlas` — all 3:1, all on a charcoal ground with a
warm off-white lowercase wordmark and cyan/teal structure with coral accents.

## Approach

The wordmark is **not** generated. Image models render short lowercase words
unpredictably, and accepting whatever letterforms come back is most of what
makes a generated banner look cheap. The artwork is generated deliberately
textless, and the type is set locally in Lato Light, sized to a fixed fraction
of the frame width so names of different lengths carry comparable optical
weight.

## Subject

A stack of mode fields receding in perspective: the front one brilliant and
sharply resolved, each successive layer finer grained and fainter than the one
before it. That is what this package returns — an energy-ranked sequence of
spatial modes extracted from a spatiotemporal dataset, whether by POD, SPOD, DMD
or BSMD. The lobed, cellular pattern on each layer is the shape a mode field
takes when plotted.

## Prompt (artwork only, no text)

```text
A stunning abstract scientific artwork, wide 2:1 landscape: a stack of
translucent eigenmode fields layered one behind another in deep perspective,
receding toward the right. Each layer is a smooth continuous spatial pattern of
alternating lobes and cells, like a vibration mode shape or a standing wave
pattern on a surface, and each successive layer is finer grained and fainter
than the one in front, suggesting a spectrum of modes ordered by energy. The
nearest, most energetic layer is brilliant electric cyan with hot coral and
amber lobes; the deeper ones fade through teal into indigo darkness with
atmospheric haze. Deep near-black charcoal ground, volumetric glow, depth of
field, fine film grain, rich blacks and luminous highlights. Cinematic, refined,
expensive, gallery-quality abstract data art. ABSOLUTELY NO TEXT, no letters, no
words, no numbers, no labels, no axes, no logos, no watermarks. Pure abstract
imagery only. Leave the left third comparatively dark, calm and uncluttered as
negative space.
```

## Post-processing

- `image_gen` rejects a 3:1 request and falls back to 2:1, so the returned
  1408x704 image was centre-cropped to 1408x469. The subject is a repeating
  field texture, so the crop reads as the mode fields continuing past the frame
  rather than as clipping.
- Wordmark composited locally: Lato Light, auto-sized to 30% of frame width
  (65 px here — `openmodalpy` is a long name), tracking 6% of point size, warm
  off-white `#F7F3EC`, with a wide blurred dark halo underneath.
- JPEG q95, no chroma subsampling. The image is a smooth gradient render with no
  flat colour fields, which is the case PNG handles worst and JPEG handles best.

## Rejected alternatives

- **Turbulence decomposed into lobed modes** — a tangled field on the left
  resolving into monopole, dipole, quadrupole and hexapole shapes on the right.
  Conceptually the sharpest of the three, and the closest to what POD actually
  does, but the busy left side left nowhere clean for the wordmark.
- **Ranked vortical structures** — a sequence of mode structures growing in
  energy. Murkier, and read more as galaxies than as modes.
