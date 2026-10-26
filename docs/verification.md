# Verification

The standalone publication directory was checked locally on 2026-10-27.

| Check | Outcome |
| --- | --- |
| Synthetic forward/backward and guided generation | Passed, CPU; loss `1.298166` |
| One-epoch synthetic training and model serialization | Passed; finite loss and checkpoint saved |
| Wheel build and separate-target installation | Passed; installed-package CPU smoke check also produced loss `1.298166` |
| Retained source identity and numerical CSV preservation | Passed for 196 copied source/asset files |
| Credential-pattern, internal-material and file-size scan | Passed for the publication candidate files |

These checks exercise the distributed implementation and selected file identities. They do not rerun benchmark training or establish physical-robot deployment performance. Asset availability and data requirements are stated in [assets](assets.md).

The GitHub workflow runs the source/asset check and CPU smoke test on push and pull request. A live GitHub workflow result is separate from the local results above.
