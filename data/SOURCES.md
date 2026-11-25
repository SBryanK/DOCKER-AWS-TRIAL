# Sample policy corpus

Three public PDFs, used to demonstrate grounding and citations. All documents are
public; they are committed only as a small, stable demo corpus.

| File | Publisher | Pages | Chars | SHA-256 | Source |
|---|---|---|---|---|---|
| `nist-ai-rmf-1.0.pdf` | NIST (US) | 48 | 106,432 | `7576edb5…c4a89f1` | https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf |
| `nist-genai-profile.pdf` | NIST (US) | 64 | 160,489 | `6e73620a…6cff89b6a` | https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf |
| `uk-ai-regulation-white-paper.pdf` | UK DSIT / OGL v3 | 93 | 215,896 | `79d0b329…f2ce0d9c7` | https://assets.publishing.service.gov.uk/media/64cb71a547915a00142a91c4/a-pro-innovation-approach-to-ai-regulation-amended-web-ready.pdf |

Total: **205 pages, 482,817 characters (~120k tokens)**, 8.5 MB.

## Licensing

- NIST publications are US government works and in the public domain.
- The UK white paper is published under the Open Government Licence v3.0 (OGL v3).
- The PDFs are redistributed unmodified, with source URLs and checksums above.

## Refreshing the corpus

```bash
curl -sL -o data/policies/nist-ai-rmf-1.0.pdf \
  https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf
curl -sL -o data/policies/nist-genai-profile.pdf \
  https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf
curl -sL -o data/policies/uk-ai-regulation-white-paper.pdf \
  "https://assets.publishing.service.gov.uk/media/64cb71a547915a00142a91c4/a-pro-innovation-approach-to-ai-regulation-amended-web-ready.pdf"
shasum -a 256 data/policies/*.pdf   # compare with the table
```
