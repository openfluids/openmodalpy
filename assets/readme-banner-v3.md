# README banner — readme-banner-v3.jpg

Asset: `readme-banner-v3.jpg` (1408x469, 3:1)

Tool/model: xAI Grok CLI, built-in `image_gen` tool, plus local compositing.

## Subject prompt

```text
A receding stack of rectangular scientific field maps, like a deck of cards seen in gentle perspective, marching from upper left down to lower right. The frontmost panel is the most defined, filled with a smooth swirling cellular flow pattern drawn in fine cyan and teal contour linework with a few small coral cores. Each panel behind is progressively smaller, fainter and hazier, dissolving into the dark.
```

## The shared specification

Every banner in the openfluids family is generated from an identical
specification block; only one subject sentence changes per repository. The block
pins the rendering, the framing and the palette:

```text
RENDERING — this governs everything: rendered as exquisitely fine, delicate,
hairline glowing lines and fine stippled luminous points. Atmospheric depth of
field, volumetric glow, fine film grain, rich deep blacks and luminous
highlights. Generous empty dark space; the artwork should feel sparse and
restrained, with only a small fraction of the frame actually lit. Cinematic,
elegant, refined, expensive, gallery-quality scientific data art.

EXPLICITLY AVOID: thick or bold strokes, heavy lines, chunky shapes, neon,
garish or oversaturated colour, poster-like flat high contrast, dense solid
blocks of glow, a busy or crowded frame. Restraint and delicacy matter more
than impact.

FRAMING: the image will afterwards be cropped to a very wide 3:1 letterbox,
keeping only the middle horizontal band. Leave the LEFT THIRD dark, calm and
completely empty as negative space — a wordmark goes there.

PALETTE: a deep near-black charcoal ground with a cool blue cast, approximately
#0D1116. Electric cyan and teal as the primary luminous colour, with a hot coral
used sparingly on only a few selected features. NO amber, NO gold, NO
orange-yellow, NO violet, NO purple, NO green, NO magenta, NO rainbow or
spectral colourmaps.

Full bleed: no border, no frame, no matte, no letterbox bars, no vignette ring.
ABSOLUTELY NO TEXT: no letters, no words, no numbers, no axis labels, no tick
marks, no logos, no watermarks, no signatures.
```

The `EXPLICITLY AVOID` clause exists because an earlier revision asked for
"thick", "bold", "punchy" and "very high contrast" artwork and got exactly that:
strokes 3–5 px, lit area up to 2.5x higher, accent saturation 177 against a
baseline of 126. Delicacy has to be stated, and its opposite has to be forbidden.

## Typography

The wordmark is **not** generated. Image models render short lowercase words
unpredictably, and accepting whatever letterforms come back is most of what makes
a generated banner look cheap. The artwork is generated deliberately textless and
the type is set locally:

- **Lato Light at 96 px**, constant across the whole family, tracking 6% of point
  size, ink `#F7F3EC`, left margin 82 px.
- Vertical placement by **optical centring**: the x-height band is centred on the
  frame midline. All the repository names are lowercase, so x-height carries the
  visual mass; centring it — rather than the bounding box, which ascenders and
  descenders distort differently per name — is what makes `dsgbr` and
  `openmodalpy` sit at the same apparent height.
- A small coral `openfluids` eyebrow sits above each repository wordmark. The
  organisation banner omits it, since it would be captioning itself.

Earlier revisions sized each wordmark to a *fraction of frame width*, which
equalises the width of the type block but not the letters: `openmodalpy` ended up
at 65 px and `fftkit` at 112 px, a 1.7x spread.

## Grading

- Ground normalised to `#0D1116` through a shadow-weighted mask, with the black
  point estimated from the darkest 8% of pixels wherever they fall — sampling the
  left margin measures artwork in banners whose streaklines reach the edge.
- Saturation lifted **only on the brightest accent cores** — a spike tip, a vortex
  centre — leaving the surrounding glow and the ground untouched. A small hot
  centre reads as punch; a uniformly saturated frame reads as neon.

## Measured against the previous banner

| metric | previous | this one |
|---|---|---|
| wordmark point size | 65–112 px across the family | 96 px everywhere |
| wordmark placement spread | 123 px | 13 px |
| ground black point spread | 14.8 | 4.7 |
| accent saturation | 125.7 | 139.0 |

Format: 1408x469 (3:1), JPEG quality 95, no chroma subsampling.

