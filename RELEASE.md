### Changed

- Refreshed the published benchmarks against emlx 0.4.1 (was 0.3.1). EMLX 0.4
  is a much stronger GPU baseline, so the Emily-vs-EMLX comparisons in the
  README and benchmark report now read 1.44× (DistilBERT QA), 1.26×
  (Qwen3-0.6B decode) and 1.17× (Qwen3-4B) in Emily's favour, down from
  2.72×/5.82×/~3.2×. Emily's own lane numbers are essentially unchanged; EMLX
  still does not complete the ViT-base or Whisper-tiny tiers.
