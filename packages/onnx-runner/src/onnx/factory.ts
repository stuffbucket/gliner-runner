import type { RunnerComponents } from "../components.js";
import type { Settings } from "../settings.js";
import { OnnxArtifactCatalog } from "./artifacts.js";
import { OnnxBackendFactory } from "./backend.js";
import { HuggingFaceClient } from "./huggingface.js";
import { OnnxModelProvisioner } from "./install.js";

/**
 * Entry point imported dynamically by the CLI (see `components.ts`): assembles the on-disk artifact catalog,
 * the ONNX backend and the model provisioner. Scanning the models directory here seeds the advertised
 * `validated_models`. `HF_ENDPOINT` and `HF_TOKEN` configure the Hugging Face client shared by refresh and
 * downloads.
 */
export async function createComponents(settings: Settings): Promise<RunnerComponents> {
  const huggingface = new HuggingFaceClient();
  const catalog = new OnnxArtifactCatalog(settings.modelsDir, { remote: huggingface });
  await catalog.inventory();
  return {
    catalog,
    backends: [new OnnxBackendFactory(catalog)],
    provisioner: new OnnxModelProvisioner({
      modelsDir: settings.modelsDir,
      templatesDir: settings.templatesDir,
      huggingface,
    }),
  };
}
