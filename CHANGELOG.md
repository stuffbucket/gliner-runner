# Changelog

## [0.5.0](https://github.com/stuffbucket/gliner-runner/compare/v0.4.0...v0.5.0) (2026-10-16)


### Features

* browser playground on the docs site, and a saved CORS allow-list ([bd870d6](https://github.com/stuffbucket/gliner-runner/commit/bd870d6e006726730eec01131f5258717493df8a))
* **cors:** save allowed browser origins in settings.json ([5b36e44](https://github.com/stuffbucket/gliner-runner/commit/5b36e44c68a25a60eb4c8a51c4fd36469c20d357))
* run gliner-runner inside the playground page, and one-command "use my machine" ([6be2eb4](https://github.com/stuffbucket/gliner-runner/commit/6be2eb400a981da8acce063d9512ac94f1f8c848))
* **runner:** gliner-runner playground and install.sh --playground ([c7fdd76](https://github.com/stuffbucket/gliner-runner/commit/c7fdd7659e62345998593c8352552207d5944d7d))
* **site:** 20 questions in the playground ([b25c2ec](https://github.com/stuffbucket/gliner-runner/commit/b25c2ec42792961eda9322267b304b47f32445c8))
* **site:** 20 questions in the playground, and a steeper fade in the stack ([036bb2e](https://github.com/stuffbucket/gliner-runner/commit/036bb2e4f73debe2933f91faca075894f757024c))
* **site:** a 20 questions game tuned on 605 answers, and timing that skips the model load ([5c937e5](https://github.com/stuffbucket/gliner-runner/commit/5c937e5caa90a806e53142239e176dcc1db8f2f1))
* **site:** a rotating gallery of real example answers on the landing page ([8814f55](https://github.com/stuffbucket/gliner-runner/commit/8814f551ce8ad0f2951c324b5f75f1d0ea9d9b90))
* **site:** a simple landing page and a full-window playground ([8fb51e3](https://github.com/stuffbucket/gliner-runner/commit/8fb51e318c9c785ecfe3811ed559e5e5eeee396f))
* **site:** a stack of answer cards beside the text box ([4a0ce13](https://github.com/stuffbucket/gliner-runner/commit/4a0ce13c1c088872c9e3adfcebd64adb869b3aaf))
* **site:** browser playground that calls the local runner ([5f35a5e](https://github.com/stuffbucket/gliner-runner/commit/5f35a5e98a734f774c79b1e0c8329099c2094737))
* **site:** Classify, Start and answering a guess show the newest card ([e67d2ad](https://github.com/stuffbucket/gliner-runner/commit/e67d2adc980e8a0f313ec26fe1c86fbe33e0bf70))
* **site:** jump to the newest card when you ask for one ([5c897c0](https://github.com/stuffbucket/gliner-runner/commit/5c897c0e182a49d45ac85da37b768993307eb288))
* **site:** one type ramp and control set for the landing page and playground ([6caeb5e](https://github.com/stuffbucket/gliner-runner/commit/6caeb5ea056246c912dbf7d27f8412f9b4ed007d))
* **site:** simple landing page and a full-window playground ([d10341e](https://github.com/stuffbucket/gliner-runner/commit/d10341e6d2bb93533cbc897bfca39ee3e14972b1))
* **site:** step back and forth through the answer stack ([e0fabbc](https://github.com/stuffbucket/gliner-runner/commit/e0fabbca073a4a2628dc1e5e9047dc541aa2cebe))
* **site:** the playground runs gliner-runner in the tab or on this machine ([2d4b6f9](https://github.com/stuffbucket/gliner-runner/commit/2d4b6f92e52b5de0fcfc11aa9eb6126f36e220aa))
* **site:** tune 20 questions to open fewer unlikely groups ([7f16f80](https://github.com/stuffbucket/gliner-runner/commit/7f16f80cb9566d113c2c083912162bd1c4119d23))
* **web:** run gliner-runner in a browser Web Worker ([de58657](https://github.com/stuffbucket/gliner-runner/commit/de58657807037b677412dccf7b54c91db7da8d20))


### Bug Fixes

* **deps:** resolve postcss-selector-parser to 7.1.6 (GHSA-rj75-hqrm-r3gf) ([#32](https://github.com/stuffbucket/gliner-runner/issues/32)) ([d5f7eae](https://github.com/stuffbucket/gliner-runner/commit/d5f7eae811294fe056c398e0bcbe7a54cb16d41a))
* **release:** carry the #platform import map into the packaged app ([d862e16](https://github.com/stuffbucket/gliner-runner/commit/d862e16a8e47cb5f1e1ea36bf4fe8d3b397963ef))
* **runner:** call fetch on the global object, as browsers require ([e6300cb](https://github.com/stuffbucket/gliner-runner/commit/e6300cb05db30b172114a801c646c9c3961421cc))
* **site:** make 20 questions clear about whose turn it is ([24f83d3](https://github.com/stuffbucket/gliner-runner/commit/24f83d3873ed2f1f3e5f83107a6498c823fccdda))
* **site:** slide the answer stack as one column ([1acd8eb](https://github.com/stuffbucket/gliner-runner/commit/1acd8eb277e6d9215465966c5e2f196721459f78))
* **site:** steer the 20 questions sample to the zebra in 4 questions ([43f1c27](https://github.com/stuffbucket/gliner-runner/commit/43f1c277e5682eb6f4523ba5154633c342d43079))


### Documentation

* simpler introduction and examples organized by job ([f3e8113](https://github.com/stuffbucket/gliner-runner/commit/f3e81133f30823cdea6617f814fb5b4b560c4b83))
* simpler introduction and examples organized by job ([b364068](https://github.com/stuffbucket/gliner-runner/commit/b36406859b8b8de3dd2b127ad0b406fdd41aa440))

## [0.4.0](https://github.com/stuffbucket/gliner-runner/compare/v0.3.1...v0.4.0) (2026-10-15)


### Features

* install all models with a size plan, and optional CORS for local web apps ([082acfe](https://github.com/stuffbucket/gliner-runner/commit/082acfe583da6e2b7216ac62f6ea7a7591d24e7b))
* install all models with a size plan, and optional CORS for local web apps ([ee9a7d9](https://github.com/stuffbucket/gliner-runner/commit/ee9a7d9eb3891a48158220511d79ac96aa46334f))


### Documentation

* plain-language guides, practical examples, and a React example ([3999331](https://github.com/stuffbucket/gliner-runner/commit/39993314a4977aa3a72125356a964adfef68d12b))
* plain-language guides, practical examples, and a React example ([3177b24](https://github.com/stuffbucket/gliner-runner/commit/3177b2487825b7522d9825ebc9ca0f0f5fa0ad3a))

## [0.3.1](https://github.com/stuffbucket/gliner-runner/compare/v0.3.0...v0.3.1) (2026-10-14)


### Bug Fixes

* trust the Homebrew tap before installing; drop release-as after 0.3.0 ([6e86464](https://github.com/stuffbucket/gliner-runner/commit/6e864640a8d7eee590dc40eafcb746764c49ced7))
* trust the Homebrew tap before installing; drop release-as after 0.3.0 ([3013871](https://github.com/stuffbucket/gliner-runner/commit/30138711289f2c9b5e10c07ae47cdba3009fc473))

## [0.3.0](https://github.com/stuffbucket/gliner-runner/compare/v0.2.1...v0.3.0) (2026-10-14)


### Features

* ship a Python-free ONNX runner with Homebrew releases ([e736c79](https://github.com/stuffbucket/gliner-runner/commit/e736c7965cfd651888358de57981779256e4eb6a))
* ship a Python-free ONNX runner with Homebrew releases ([e664600](https://github.com/stuffbucket/gliner-runner/commit/e664600e906605dabf0e2946857c6c8a47ede2a0))


### Bug Fixes

* build workspace dependencies before the ONNX runner ([acc9746](https://github.com/stuffbucket/gliner-runner/commit/acc9746ab602afe674fec9a3ed7c1ce53de9c856))
* load Node types with TypeScript 7 ([8dc8e2e](https://github.com/stuffbucket/gliner-runner/commit/8dc8e2e56c171ab77139a2c9e5a2f9cf39435b65))
* update Zod licensing notice ([1e11460](https://github.com/stuffbucket/gliner-runner/commit/1e114608ca187442ef3dcf5a553baafbffef4c0b))

## [0.2.1](https://github.com/stuffbucket/gliner-runner/compare/v0.2.0...v0.2.1) (2026-10-03)


### Bug Fixes

* decouple API and package versions ([e447c2c](https://github.com/stuffbucket/gliner-runner/commit/e447c2cbcf3bd76a9d78e15accb128bc296da93b))
* isolate licensing tests from mutation workspace ([da1c758](https://github.com/stuffbucket/gliner-runner/commit/da1c75867b97f847823d096b782191a6f4f519c2))
* keep release checks reproducible ([ed6f8eb](https://github.com/stuffbucket/gliner-runner/commit/ed6f8eb818f68f774155278f6e1e88ac1d64b9d6))

## [0.2.0](https://github.com/stuffbucket/gliner-runner/compare/v0.1.0...v0.2.0) (2026-10-03)


### Features

* adapt accelerator precision profiles ([9929fe7](https://github.com/stuffbucket/gliner-runner/commit/9929fe7e34016452fa61a923c73be007832dcc0f))
* add approved release publishing ([cea0ec7](https://github.com/stuffbucket/gliner-runner/commit/cea0ec7f3a547f528ee1c6f4e8cdc916f5071962))
* add Electron sidecar supervision ([4d0e12e](https://github.com/stuffbucket/gliner-runner/commit/4d0e12e7943485bf8b22028407cfdef4dd1444e3))
* add managed model inventory lifecycle ([7476edc](https://github.com/stuffbucket/gliner-runner/commit/7476edc13283d77bc36c4ec77a31ad2ec45ae586))
* add real-model characterization suite ([7adaf22](https://github.com/stuffbucket/gliner-runner/commit/7adaf22e4e28d7cfad9b4a4ac9eda2d11ba65771))
* add validated MPS FP16 inference ([cec3aa7](https://github.com/stuffbucket/gliner-runner/commit/cec3aa7af6754aab9f0a70b3a2bf8d6e71b271d5))
* preserve classification label descriptions ([8c8245a](https://github.com/stuffbucket/gliner-runner/commit/8c8245abf32ef9eb211f0374b22500596cbbc741))
* publish initial gliner runner release ([45204d4](https://github.com/stuffbucket/gliner-runner/commit/45204d457d8734e7ef2cc2e366fe7cb265a89a44))
* report exact inference token usage ([e713676](https://github.com/stuffbucket/gliner-runner/commit/e713676f01a8ae65eead3a0ce599eba169ecc904))
* strengthen language-neutral client boundary ([042ffa0](https://github.com/stuffbucket/gliner-runner/commit/042ffa0567b20241ad973b64c09b9e7b74486fe8))

## Changelog

Notable changes to GLiNER Runner are recorded here by Release Please.
