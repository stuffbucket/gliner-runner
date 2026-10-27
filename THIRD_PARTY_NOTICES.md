# Third-party notices

This document covers the standalone GLiNER Runner source distribution, Python
package, container image, `@gliner-runner/client` package, and the Python-free
ONNX runtime distribution. It is
informational and does not replace the license terms shipped by each
dependency.

## Distribution boundaries and release inventory

- The GLiNER Runner source and Python wheel do not bundle third-party Python
  dependencies or model weights. Package installers resolve Python
  dependencies separately.
- The container image installs the optional PyTorch dependency set. Its build
  creates `/usr/share/doc/gliner-runner/PYTHON_PACKAGES.md` from the exact
  installed distributions and retains the distributions' installed license
  and notice files. The image also retains Debian notices under
  `/usr/share/doc`.
- The TypeScript client declares, but does not bundle, its production
  dependency. Its package includes `LICENSE`, `NOTICE`, and this file.
- The Python-free ONNX runtime distribution
  (`gliner-runner-onnx-<version>-<platform>-<arch>.tar.gz`, built by
  `mise run package:onnx-runtime`) bundles a Node.js runtime, the compiled
  `@gliner-runner/onnx-runner` application, and its exact production npm
  closure, including the ONNX Runtime native library. It contains no Python,
  PyTorch, Transformers, GLiNER2, or model weights. Each archive carries its own
  generated `THIRD_PARTY_NOTICES.md` and `release-manifest.json` with the exact
  bundled versions, licenses, sizes, and SHA-256 digests; see
  [Python-free ONNX runtime distribution](#python-free-onnx-runtime-distribution).
- Production Python dependency ranges other than `gliner2==2.1.0` are not
  locked. The research-only `onnx-export` extra is fully pinned to its qualified
  versions but is not installed in release containers. A release must use the
  container's generated inventory rather than treating the table below as an
  exact resolved bill of materials.

## Declared Python dependencies

These are the dependencies declared by `pyproject.toml`, not a claim that every
listed package is copied into the source or wheel.

| Distribution | Declared range | Upstream license |
| --- | --- | --- |
| `aiofiles` | `>=24.1.0,<25` | Apache-2.0 |
| `fastapi` | `>=0.115.0,<1` | MIT |
| `httpx` | `>=0.27.0,<1` | BSD-3-Clause |
| `huggingface-hub` | `>=0.28.0,<2` | Apache-2.0 |
| `platformdirs` | `>=4.3.0,<5` | MIT |
| `psutil` | `>=6.1.0,<8` | BSD-3-Clause |
| `pydantic` | `>=2.10.0,<3` | MIT |
| `typer` | `>=0.15.0,<1` | MIT |
| `typing-extensions` | `>=4.12.0,<5` | PSF-2.0 |
| `uvicorn` | `>=0.34.0,<1` | BSD-3-Clause |
| `gliner2` (optional `local` extra) | `==2.1.0` | Apache-2.0 |
| `protobuf` (optional PyTorch extra) | `>=4,<8` | BSD-3-Clause |
| `sentencepiece` (optional PyTorch extra) | `>=0.2,<1` | Apache-2.0 |
| `transformers` (optional PyTorch extra) | `>=5,<6` | Apache-2.0 |
| `gliner2` (optional ONNX export extra) | `==2.1.0` | Apache-2.0 |
| `torch` (optional ONNX export extra) | `==2.14.0` | Apache-2.0 and bundled permissive licenses |
| `transformers` (optional ONNX export extra) | `==5.19.0` | Apache-2.0 |
| `onnx` (optional ONNX export extra) | `==1.23.2` | Apache-2.0 |
| `onnxruntime` (optional ONNX export extra) | `==1.31.0` | MIT |
| `onnxscript` (optional ONNX export extra) | `==0.7.2` | MIT |

The `onnx-export` extra supports an opt-in, research-only exporter and is not a
runtime backend or release-container dependency. Its exact direct dependency
versions are the versions used for the recorded Small and Decide feasibility
runs. Their transitive dependencies and bundled native components remain under
their own terms; installed license metadata is authoritative for any local
redistribution of the research environment.

Fastino GLiNER2 2.1.0's `local` extra declares NumPy, PEFT, safetensors,
PyTorch, and Transformers in addition to GLiNER2's base dependencies. Those
packages have further transitive and, for PyTorch and NumPy, bundled native
dependencies under multiple permissive licenses. The installed distributions'
license directories and the generated container inventory are authoritative
for the versions selected for a release. Runtime qualification used Transformers
5.19.0, protobuf 7.36.2, and sentencepiece 0.2.2; compatible versions may be
selected within the declared ranges, and the generated inventory records the
exact release result. In particular:

- Fastino GLiNER2 2.1.0 is Apache-2.0. The Apache-2.0 text is in `LICENSE`.
- PyTorch is distributed with its own `LICENSE` and `NOTICE`, including
  attribution and BSD terms for PyTorch, Caffe, and Caffe2 code and terms for
  bundled components. Those files must remain with every redistributed PyTorch
  binary.
- Transformers, PEFT, Accelerate, safetensors, and tokenizers declare
  Apache-2.0. Their installed license and notice files must be retained.
- NumPy is primarily BSD-3-Clause and includes separately licensed bundled
  components. Its installed `LICENSE.txt` must be retained in full.

On Linux, the PyTorch resolver may select CUDA, NVIDIA, and Triton wheels that
do not appear in `pyproject.toml`. NVIDIA wheels can use proprietary
`LicenseRef-NVIDIA-Proprietary` terms rather than an open-source license.
Their EULAs and `License.txt` files must be retained and reviewed for the
intended redistribution. The generated inventory deliberately reports the
actual platform-specific result and fails when a distribution provides
neither license metadata nor a packaged license file.

`cuda-toolkit==13.0.3.0` is a dependency-only metapackage whose wheel provides
neither license metadata nor a license file. The container generator therefore
applies an explicit `NVIDIA CUDA Toolkit EULA (metadata override)` label. This
override documents the required manual review; it does not replace or alter
the NVIDIA terms carried by the CUDA component wheels.

## TypeScript client dependency

The exact production dependency in `pnpm-lock.yaml` is:

| Package | Version | License |
| --- | --- | --- |
| `zod` | `4.6.6` | MIT |

`@types/node`, `typescript`, and `undici-types` are build/test inputs and are
not production dependencies of the published client package.

### Zod MIT notice

> MIT License
>
> Copyright (c) 2025 Colin McDonnell
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in
> all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

### ONNX Runtime and ONNX Script MIT notice

> MIT License
>
> Copyright (c) Microsoft Corporation
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in
> all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

## Python-free ONNX runtime distribution

The ONNX runtime archive is platform-specific and assembled from the pinned
`pnpm-lock.yaml` and the pinned Node.js release in `mise.toml`. It prunes
`onnxruntime-node` to the native binaries of its own platform and architecture.
The generated file inside each archive is authoritative for that archive; the
tables below record the exact resolved production closure of
`@gliner-runner/onnx-runner` so that a dependency change is a reviewed change
to this file.

Direct dependencies:

| Package | Version | License |
| --- | --- | --- |
| `@huggingface/tokenizers` | `0.2.0` | Apache-2.0 |
| `onnxruntime-node` | `1.30.0` | MIT |
| `zod` | `4.6.6` | MIT |

Transitive production dependencies (all installed by `onnxruntime-node` or its
dependencies):

| Package | Version | License |
| --- | --- | --- |
| `adm-zip` | `0.6.1` | MIT |
| `define-data-property` | `1.1.4` | MIT |
| `define-properties` | `1.2.1` | MIT |
| `es-define-property` | `1.0.1` | MIT |
| `es-errors` | `1.3.0` | MIT |
| `escape-string-regexp` | `4.0.0` | MIT |
| `global-agent` | `4.1.3` | BSD-3-Clause |
| `globalthis` | `1.0.4` | MIT |
| `gopd` | `1.2.0` | MIT |
| `has-property-descriptors` | `1.0.2` | MIT |
| `matcher` | `4.0.0` | MIT |
| `object-keys` | `1.1.1` | MIT |
| `onnxruntime-common` | `1.30.0` | MIT |
| `semver` | `7.8.5` | ISC |
| `serialize-error` | `8.1.0` | MIT |
| `type-fest` | `0.20.2` | (MIT OR CC0-1.0) |

Other bundled components:

| Component | Version | License |
| --- | --- | --- |
| Node.js runtime (unmodified official build) | `24.21.0` | MIT and bundled third-party licenses |
| ONNX Runtime native library (inside `onnxruntime-node`) | `1.30.0` | MIT |

- The Node.js runtime is copied unchanged from the pinned Node.js 24
  distribution. Its complete `LICENSE`, which also covers the third-party
  components built into Node.js, is shipped as `licenses/node-LICENSE`, and the
  archive manifest records the binary's version, size, and SHA-256.
- `onnxruntime-node` and `onnxruntime-common` ship no license file. The
  Microsoft MIT notice above applies to them and to the ONNX Runtime native
  library; ONNX Runtime's own third-party notices are published at
  <https://github.com/microsoft/onnxruntime/blob/v1.30.0/ThirdPartyNotices.txt>.
- `@huggingface/tokenizers` is Apache-2.0 and ships its `LICENSE` in the
  package; it declares no `NOTICE` file.
- Development-only packages (`typescript`, `@types/node`, `undici-types`, and
  the workspace client) are not bundled.
- Model weights are never bundled. They are installed separately by
  `gliner-runner prepare-onnx` under their own license metadata (see
  [Curated model metadata](#curated-model-metadata)).

## Major BSD notices

The following notices cover prominent BSD-licensed direct dependencies. The
complete license files installed by package installers remain controlling.

### HTTPX

Copyright (c) 2019, Encode OSS Ltd. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

### psutil

Copyright (c) 2009, Jay Loden, Dave Daeschler, Giampaolo Rodola. All rights
reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the psutil authors nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

## Curated model metadata

GLiNER Runner does not redistribute the curated model weights. Model
installation is a separate, operator-approved download, and the downloaded
snapshot retains its own license obligations.

The curated model revisions for this release are:

| Repository | Pinned revision | Model-card license | Root license files |
| --- | --- | --- | --- |
| `fastino/GLiNER2.5-Decide` | `9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90` | `apache-2.0` | No `LICENSE` or `NOTICE` |
| `fastino/GLiNER2.5-multi-Decide` | `e173bca1f0c4217d7a55b3b0c705d5ece8a0218d` | `apache-2.0` | No `LICENSE` or `NOTICE` |
| `fastino/GLiNER2.5-Decide-1B` | `67af949e17bebd20f05fe020a971079889e7a462` | `apache-2.0` | No `LICENSE` or `NOTICE` |

For each pinned revision:

- the model card's YAML metadata declares `license: apache-2.0`;
- the revision's repository file listing has no root `LICENSE` or `NOTICE`
  file; and
- the Apache-2.0 declaration is therefore model-card metadata, not a license
  file bundled by this repository or found at the root of that model snapshot.

Operators who download or redistribute that model must preserve the model card
and verify that the declared license is sufficient for their use. GLiNER
Runner's `LICENSE` applies to GLiNER Runner and is not a representation that
the separately downloaded model weights are bundled with or relicensed by this
project.

The pinned evidence is:

- model card:
  <https://huggingface.co/fastino/GLiNER2.5-Decide/blob/9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90/README.md>
- revision file listing:
  <https://huggingface.co/api/models/fastino/GLiNER2.5-Decide/revision/9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90>
- multilingual model card and file listing:
  <https://huggingface.co/fastino/GLiNER2.5-multi-Decide/blob/e173bca1f0c4217d7a55b3b0c705d5ece8a0218d/README.md>,
  <https://huggingface.co/api/models/fastino/GLiNER2.5-multi-Decide/revision/e173bca1f0c4217d7a55b3b0c705d5ece8a0218d>
- 1B model card and file listing:
  <https://huggingface.co/fastino/GLiNER2.5-Decide-1B/blob/67af949e17bebd20f05fe020a971079889e7a462/README.md>,
  <https://huggingface.co/api/models/fastino/GLiNER2.5-Decide-1B/revision/67af949e17bebd20f05fe020a971079889e7a462>
- GLiNER2 2.1.0 license:
  <https://github.com/fastino-ai/GLiNER2/blob/v2.1.0/LICENSE>

## Release gate

Every release must:

1. run `mise run check:licensing`;
2. build Python artifacts and confirm `LICENSE`, `NOTICE`, and this file are
   present;
3. pack the TypeScript client and confirm the same three files are present;
4. for an ONNX runtime archive, run `mise run package:onnx-runtime` and
   `mise run verify:onnx-runtime`, and confirm the archive's generated
   `THIRD_PARTY_NOTICES.md` matches the tables above;
5. for a container release, retain the generated exact Python inventory,
   installed distribution license files, PyTorch's `LICENSE` and `NOTICE`, and
   Debian's `/usr/share/doc` notices; and
6. review any changed dependency, lockfile entry, base image, curated model
   revision, generated entry, or explicit license override before publication.

The container inventory is a release gate, not legal approval. In particular,
the release owner must review any `LicenseRef`, proprietary, `UNKNOWN`, or
newly introduced license entry before distributing the image.
