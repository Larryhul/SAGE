# Third-party notices

## MID

The top-level baseline and compatibility modules derive from [Gutianpei/MID](https://github.com/Gutianpei/MID), the implementation of *Stochastic Trajectory Prediction via Motion Indeterminacy Diffusion* (CVPR 2022). Its upstream README declares an MIT license. The historical checkout did not include a standalone license file; the original README is retained in [notices/MID_README.md](third_party/notices/MID_README.md) as the upstream declaration and attribution.

Original MID authors: Tianpei Gu, Guangyi Chen, Junlong Li, Chunze Lin, Yongming Rao, Jie Zhou and Jiwen Lu. The inherited checkout commit is recorded in [provenance/upstream.json](provenance/upstream.json).

## Trajectron++ and embedded utilities

The MID compatibility stack incorporates Trajectron++-derived components. Preserve the [MIT notice from Stanford Autonomous Systems Lab](third_party/notices/Trajectron++_LICENSE.txt). Individual source comments identify additional embedded utility origins, including `numpy_ringbuffer`; its [upstream notice](third_party/notices/numpy_ringbuffer_LICENSE.txt) is also retained.

## Scope

The repository's [MIT license](LICENSE) applies to original code and repository documentation. Upstream components retain their original notices. IEEE class and bibliography-style files retain their embedded notices. Manuscripts, original figures and datasets retain their respective authors' and providers' rights; they are not relicensed by the code license.
