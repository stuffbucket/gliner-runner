# Third-party notices

This document covers the standalone GLiNER Runner source distribution, Python
package, container image, and `@gliner-runner/client` package. It is
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
- Python dependency ranges other than `gliner2==2.0.0` are not locked in this
  repository. A release must therefore use the container's generated inventory
  rather than treating the table below as an exact resolved bill of materials.

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
| `gliner2` (optional `local` extra) | `==2.0.0` | Apache-2.0 |

Fastino GLiNER2 2.0.0's `local` extra declares NumPy, PEFT, safetensors,
PyTorch, and Transformers in addition to GLiNER2's base dependencies. Those
packages have further transitive and, for PyTorch and NumPy, bundled native
dependencies under multiple permissive licenses. The installed distributions'
license directories and the generated container inventory are authoritative
for the versions selected for a release. In particular:

- Fastino GLiNER2 2.0.0 is Apache-2.0. The Apache-2.0 text is in `LICENSE`.
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
| `zod` | `4.1.11` | MIT |

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
| `fastino/GLiNER2.5-Decide` | `5a7adf72a23b4d311abae6ce050d7f0012bb3416` | `apache-2.0` | No `LICENSE` or `NOTICE` |
| `fastino/GLiNER2.5-multi-Decide` | `a35a0cd3b7a0f00f2effc576f454cd48fa98aa5f` | `apache-2.0` | No `LICENSE` or `NOTICE` |
| `fastino/GLiNER2.5-Decide-1B` | `688cd7ba8917a0855ad3ce929cba5a9998932e79` | `apache-2.0` | No `LICENSE` or `NOTICE` |

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
  <https://huggingface.co/fastino/GLiNER2.5-Decide/blob/5a7adf72a23b4d311abae6ce050d7f0012bb3416/README.md>
- revision file listing:
  <https://huggingface.co/api/models/fastino/GLiNER2.5-Decide/revision/5a7adf72a23b4d311abae6ce050d7f0012bb3416>
- multilingual model card and file listing:
  <https://huggingface.co/fastino/GLiNER2.5-multi-Decide/blob/a35a0cd3b7a0f00f2effc576f454cd48fa98aa5f/README.md>,
  <https://huggingface.co/api/models/fastino/GLiNER2.5-multi-Decide/revision/a35a0cd3b7a0f00f2effc576f454cd48fa98aa5f>
- 1B model card and file listing:
  <https://huggingface.co/fastino/GLiNER2.5-Decide-1B/blob/688cd7ba8917a0855ad3ce929cba5a9998932e79/README.md>,
  <https://huggingface.co/api/models/fastino/GLiNER2.5-Decide-1B/revision/688cd7ba8917a0855ad3ce929cba5a9998932e79>
- GLiNER2 2.0.0 license:
  <https://github.com/fastino-ai/GLiNER2/blob/v2.0.0/LICENSE>

## Release gate

Every release must:

1. run `mise run check:licensing`;
2. build Python artifacts and confirm `LICENSE`, `NOTICE`, and this file are
   present;
3. pack the TypeScript client and confirm the same three files are present;
4. for a container release, retain the generated exact Python inventory,
   installed distribution license files, PyTorch's `LICENSE` and `NOTICE`, and
   Debian's `/usr/share/doc` notices; and
5. review any changed dependency, lockfile entry, base image, curated model
   revision, generated entry, or explicit license override before publication.

The container inventory is a release gate, not legal approval. In particular,
the release owner must review any `LicenseRef`, proprietary, `UNKNOWN`, or
newly introduced license entry before distributing the image.
