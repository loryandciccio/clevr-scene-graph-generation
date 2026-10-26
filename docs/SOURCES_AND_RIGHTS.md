# Sources, ownership and redistribution status

This document distinguishes third-party input rights from project source-code and model rights. It is not a blanket license grant.

| Material | Source and current evidence | Treatment in this copy |
| --- | --- | --- |
| CLEVR v1.0 dataset, scene annotations and original images | [Official CLEVR site](https://cs.stanford.edu/people/jcjohns/clevr/) declares the **data** under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/). Redistribution and adaptations are permitted, including commercially, if attribution, license link and change notices are provided. | Dataset omitted for size/privacy and independent acquisition. Any future copied images must be attributed separately. |
| CLEVR generation code | [facebookresearch/clevr-dataset-gen license](https://github.com/facebookresearch/clevr-dataset-gen/blob/main/LICENSE): BSD, for that software. | Not included as source here. Its BSD grant does not automatically relicense dataset images or this project's code. |
| CLEVR paper (arXiv 1612.06890) | [Official article](https://arxiv.org/abs/1612.06890); arXiv hosting alone does not prove third-party PDF redistribution permission. | Original `1612.06890v1.pdf` excluded; cite/link the publication. |
| University assignment sheet | `Progetto 3.1.pdf` in original repository; no independently verified redistribution permission. | Excluded; kept in original private repository. |
| Project report | `RelazioneDeep_Cozzolino_Nardoni.pdf` lists two authors. Joint authorship warrants joint-rights review; academic or institutional limitations were not determined. | Excluded; preserved in original private repository. |
| Project Python/shell code | Authored within the private repository; no source-level LICENSE identified in the checked main tree. Collaborator credited in README. | Copied without granting MIT, Apache, CC or any other new license. **Public publication of jointly authored code requires rights-holder authorization.** |
| Training outputs and project checkpoints | Two ~93 MB current `.pth` files; further previous checkpoint revisions in Git history. ZIP/PyTorch pickle metadata inspected statically without deserializing. Dataset-trained parameters may carry independent ownership/licensing restrictions. | Omitted until provenance, consent and distribution terms are established. |
| TorchVision ImageNet/ResNet34 pretrained weights | [TorchVision docs](https://docs.pytorch.org/vision/stable/models.html) warn of model-specific terms deriving from training datasets. | No pretrained files included. `--backbone-init imagenet` may trigger a download only when requested. |
| Historical validation metrics | Copied from this project's recorded `predicted_scene_graphs_metrics.json` in the historical research run; split described as CLEVR validation in project documentation. | Included unchanged as evidence of historical *reported* validation, not independently rerun test results. |
| Training plots / illustrative examples | Historical run PNGs, some potentially derived from source dataset and training presentations; no independent metadata/provenance proof for every image. | Omitted. A clearly labeled, hand-authored JSON graph is supplied for **visualizer only**. |

**Outstanding decision:** before public disclosure, confirm permission to publish jointly authored project code and ownership of model outputs. A repository can be technically clean without being legally authorized for public redistribution.

No blanket open-source license file is issued in this preparation. Other third-party package licenses continue to apply when installed as dependencies. Original source commit and SHA-256 file manifest document what this copy contains.
