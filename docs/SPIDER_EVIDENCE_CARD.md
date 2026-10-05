# Spider evidence card: what the current corpus supports

This card is a research handoff for the spider modeling lane. The evidence
supports bounded behavioral and anatomical observations. It does not validate
a spider controller or establish a complete neural circuit. Corpus queries and
the ten passage-level numeric checks are reproducible with
`python tools/spider_corpus.py build`, `python tools/spider_corpus.py embed`,
and `python tools/spider_corpus_check.py` after source files are available in
the ignored source cache. The existing embedding model must be served at
`127.0.0.1:8098`.

## Jump performance: useful reference, very narrow sample

Nabawy et al. measured exploratory jumps from one adult female *Phidippus
regius* (150 mg, 15 mm). Although four spiders were obtained, only this female
jumped in the test chamber; the study recorded one example jump for each of 15
tasks. Take-off speed ranged from 0.52 to 0.97 m/s, time to take-off from 18.1
to 31.6 ms, and take-off angle from -10.5° to 53.5°. These are task outcomes
from one animal, not a population range and not direct launch targets for the
23.5 mg engine model. The paper points to per-task Supplementary Dataset S3;
that table still needs to be retrieved and audited before freezing task-level
thresholds. [Nabawy et al. 2018, “Energy and time optimal trajectories in
exploratory jumps of the spider *Phidippus regius*](https://pmc.ncbi.nlm.nih.gov/articles/PMC5940701/),
Jumping experiment and Supplementary Dataset S3.

## Eye roles: secondary-eye results are species- and task-specific

De Agrò et al. studied *Menemerus semilimbatus*. Their eye-field experiment
used 31 spiders (16 PLE and 15 ALE); the main eye-masking experiment used 179
spiders (15 males, 89 females, 75 juveniles) in ALE, PLE, and ALE+PLE groups.
Moving stimuli elicited body pivots used to map eye-pair detection angles; the
paper also tested responses to biological versus random motion with secondary
eye pairs unoccluded. Do not treat its apparatus angles (monitor orientations
of 65° and 120°) as measured visual-field limits. The study supports secondary
eye contributions in these tested conditions; it does not supply a universal
salticid eye transfer function. [De Agrò et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11418026/),
Methods and Results.

Winsor et al.'s review summarizes a *Phidippus audax* visual-system figure in
which the principal-eye retinas cover about 10° at a time while eye-tube motion
supports a larger total visual angle of about 58°. Treat this as a review-level,
species-specific anatomical summary rather than a general camera calibration.
The same review distinguishes established eye-pathway anatomy from functional
interpretations that remain hypotheses. [Winsor, Morehouse & Jakob 2023,
Distributed Vision in Spiders](https://doi.org/10.1007/978-3-031-23216-9_10),
Chapter 10, Fig. 10.2 and Sect. 10.3.6.

## Portia prey number: a detour based expectancy test

Cross and Jackson tested *Portia africana* with an expectancy violation task
that used detours to change the spider's view of prey groups. The abstract
reports behavioral differences for 1 versus 2, 3, or 4 prey and for 2 versus
3, 4, or 6 prey, but not for 3 versus 4 or 6 prey. The authors interpret this
pattern as discrimination of one, two, and a larger-number category. This
supports task-specific prey-number discrimination under the tested conditions;
it does not establish a general planning mechanism or neural circuit.
[Cross & Jackson 2017, “Representation of different exact numbers of prey by a
spider-eating predator”](https://pmc.ncbi.nlm.nih.gov/articles/PMC5413887/),
Abstract, Methods and Results.

Cross and Jackson 2016 (PMC4751061) remains unavailable in this packet as
directed by source review, despite a separate cached text artifact. The
requested Tarsitano & Jackson detour papers (1992, 1994, 1997) were found as
citations, but full text could not be retrieved through local crawl4ai.
Harland & Jackson 2000 and 2002 abstracts were found in PubMed; publisher full
text crawls were blocked. These records remain unavailable in the corpus.

Aguilar-Arguello et al.'s two Salticoida species were *Marpissa marina* and
*Trite planiceps*, not *Portia*. The source index now labels it accordingly.
Keep its route-choice results separate from Portia-specific findings.
[Aguilar-Arguello et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7234261/).

## Circuit anatomy: atlases are not connectomes

The *Uloborus diversus* atlas aligns whole-brain immunostains and describes
major neuropils, neurotransmitter populations, and visual-pathway extensions.
It is anatomical evidence from that web-building species, not a synaptic
connectome and not a *Phidippus* functional circuit. The atlas authors identify
at least one region whose identity remains uncertain without tracing. The
project has no spider connectome; specific controller wiring and computation
remain hypotheses unless separately supported. [Artiushin et al. 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC12919972/),
Introduction and visual-pathway sections; [Jin et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10697844/)
provides a spider brain cell-type atlas, not a species-complete functional
wiring map.

## Other numeric observations in the checked audit

| Source observation | Species and context | Value | Limitation |
|---|---|---:|---|
| Retinal-movement bout duration | *Evarcha arcuata* spiderlings; 29 subjects, 330 observations | median 77.13 s | Not a universal sleep-cycle parameter; adult behavior was analyzed separately. |
| Prey mass on web | *Drosophila melanogaster* on *Uloborus diversus* web; n=24 | 0.0016 ± 0.0002 g | Prey mass, not spider mass. |
| Behavior-classification score | *Uloborus diversus* interacting with *D. melanogaster* | 82.9% across 10 manually annotated trials out of 30 recordings | Dataset-specific HMM evaluation; not neural prediction accuracy. |
| Route completion association | *Marpissa marina* detour task | 34% | Passage does not include the denominator; this is not Portia. |

The numeric audit also checks the Phidippus individual count, speed, take-off time,
Menemerus sample composition, and monitor orientations. Every audited value is
bound to an exact cleaned-source passage, species label, unit, and section
locator. A deliberately wrong species and a deliberately swapped unit both fail
the checker.

## Missing evidence to keep visible

- Retrieve and independently inspect Nabawy's task-level Supplementary Dataset S3.
- Obtain the unavailable Steck 2024 secondary-eye color paper (PMC11756594) or
  keep claims that depend on it out of implementation thresholds.
- Seek full text for the unavailable Tarsitano & Jackson and Harland & Jackson
  works listed in [`SPIDER_ASSET_INDEX.md`](SPIDER_ASSET_INDEX.md); the corpus is
  a checked starter packet, not an exhaustive bibliography.
- No Portia neural anatomy, spider connectome, or validated mapping from these
  taxa to the engine spider has been established here.
