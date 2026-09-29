# CA1 pilot input distribution audit — 2026-09-29

**Finding:** The 320/64 `sft-pilot-v1` packet passed label and token checks but fails a basic input-realism check against the 13 eligible development reports. It should not be used for model training or treated as an accepted learning dataset. This finding does not change the approved CA1 prompt, schema, or frozen historical packets.

The comparison uses the full visible text between `<report>` tags. Original inputs are the 13 development IDs named in the [plan](plan.md); synthetic inputs are all 320 training and 64 validation rows from `sft-pilot-v1`. Length is only one diagnostic, not a target to optimize by padding.

| Group | n | Characters, median (range) | Lines, median (range) | Below shortest original by characters | Below shortest original by lines |
| --- | ---: | ---: | ---: | ---: | ---: |
| Eligible originals | 13 | 3,289 (1,516–6,754) | 61 (42–143) | — | — |
| All training | 320 | 718 (139–2,410) | 21 (4–82) | 297 | 279 |
| New training only | 304 | 746 (310–2,329) | 21 (11–82) | 283 | 263 |
| New validation only | 56 | 612 (334–1,517) | 19.5 (12–62) | 55 | 52 |

Structural evidence is stronger than length alone. Five of 13 originals contain `IMAGE`, five contain `<PERSON>`, six contain website text, and all 13 contain some patient, contact, organization, or other parser/admin artifact. None of the 320 training examples contains `IMAGE`, `<PERSON>`, `<ORGANIZATION>`, or website text. All 304 newly generated training reports start with the same `ЛАБОРАТОРНЫЙ БЛАНК /` title and contain `Раздел 1:`; neither pattern occurs in the 13 originals. Seventy new training reports use artificial lettered marker names. The screenshots supplied during user review illustrate the same gap: sparse, regular synthetic layouts versus dense, noisy, multi-section parser output.

The previous review checked target consistency with synthetic visible cells, schema, split boundaries, and token fit. Those checks cannot establish that the generated inputs resemble the intended deployment inputs. The original set is small and includes several sources, so these counts do not define a production prevalence or justify adding arbitrary boilerplate to hit a length target. The corrective action is to author a **separate revised packet** from realistic layout scenarios grounded only in eligible development conventions, with fictitious measurements and administrative details, then repeat pair review and distribution assessment. Preserve `sft-pilot-v1` as failed historical evidence and retain the corrected 16/8 smoke set for mechanics checks. Do not train on `sft-pilot-v1`.

## Revised packet response

The [v2 packet](../../data/research/extraction/sft-pilot-v2/README.md) was
assistant-reviewed on 2026-09-29. Its 304 new training inputs have 2,591 median
characters and 47 median lines; the 56 new validation inputs have 2,537 and 49.
The generator uses varied page and panel structures, coherent fictitious assays,
and source conventions from eligible development reports. The review covers all
64 validation and 85 stratified training pairs. This closes the specific v1
universal-header and extreme-shortness failure, while leaving distribution
matching uncertain: 13 originals cannot establish deployment prevalence, and
17 new v2 reports are shorter than the shortest eligible original by characters.
The user accepted the revised v2 packet on 2026-09-29 with these limitations documented. The audit remains a warning about v1, not a claim of statistical equivalence for v2.
