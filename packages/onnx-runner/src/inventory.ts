import type { ArtifactCatalog } from "./catalog.js";
import type {
  ModelInventoryItem,
  ModelInventoryResponse,
  Precision,
  PrecisionProfile,
  ProfileGuidance,
} from "./contracts.js";
import { minimumMemoryBytes } from "./resources.js";
import type { Runtime } from "./runtime.js";

export const DEFAULT_MODEL = "decide-340m";

const MEMORY_BASIS =
  "Estimated model residency floor; post-load enforcement uses process RSS, not the sum " +
  "of process and device allocations.";
const BASELINE_PROFILE = "cpu/fp32 direct official PyTorch oracle";
const UNCHARACTERIZED = "No published numeric characterization for this model/profile.";
const PRECISION_PREFERENCE: readonly Precision[] = ["fp16", "bf16", "fp32"];

const compare = (left: string, right: string): number => (left < right ? -1 : left > right ? 1 : 0);

function supportedProfiles(runtime: Runtime): PrecisionProfile[] {
  const profiles = runtime
    .capabilities()
    .flatMap((capabilities) => [...capabilities.precision_profiles])
    .sort(
      (left, right) =>
        compare(left.device, right.device) || compare(left.precision, right.precision),
    );
  return profiles.length > 0 ? profiles : [{ device: "cpu", precision: "fp32" }];
}

function defaultProfile(profiles: readonly PrecisionProfile[], device: string): PrecisionProfile {
  for (const precision of PRECISION_PREFERENCE) {
    const match = profiles.find(
      (profile) => profile.device === device && profile.precision === precision,
    );
    if (match !== undefined) {
      return match;
    }
  }
  return profiles[0] as PrecisionProfile;
}

export async function buildModelsResponse(
  runtime: Runtime,
  catalog: ArtifactCatalog,
): Promise<ModelInventoryResponse> {
  const inventory = await catalog.inventory();
  const profiles = supportedProfiles(runtime);
  const preferred = defaultProfile(profiles, runtime.config.device);
  const loaded = runtime.loadedState();
  const models = inventory.models.map((model): ModelInventoryItem => {
    const isLoaded = loaded?.modelId === model.modelId;
    const guidance = (profile: PrecisionProfile): ProfileGuidance => {
      const characterization = catalog.characterize?.(model.modelId, profile);
      return {
        estimated_minimum_memory_bytes: minimumMemoryBytes(model.parameterCount, profile.precision),
        memory_basis: MEMORY_BASIS,
        baseline_profile: BASELINE_PROFILE,
        observed_max_abs_score_drift: characterization?.observedMaxAbsScoreDrift ?? null,
        drift_order_of_magnitude: characterization?.driftOrderOfMagnitude ?? null,
        characterization_scope: characterization?.scope ?? UNCHARACTERIZED,
      };
    };
    return {
      model_id: model.modelId,
      repository: model.repository,
      family: "GLiNER2.5-Decide",
      description: model.description,
      pinned_revision: model.pinnedRevision,
      discovered_revision: model.discoveredRevision,
      parameter_count: model.parameterCount,
      local_revisions: model.localRevisions,
      available_locally: model.localRevisions.includes(model.pinnedRevision),
      loaded: isLoaded,
      loaded_profile: isLoaded && loaded !== undefined ? loaded.profile : null,
      supported_profiles: profiles,
      default_profile: preferred,
      profile_guidance: Object.fromEntries(
        profiles.map((profile) => [`${profile.device}/${profile.precision}`, guidance(profile)]),
      ),
      download_destination: inventory.destination,
      download_required: !model.localRevisions.includes(model.pinnedRevision),
    };
  });
  return {
    default_model: DEFAULT_MODEL,
    device: runtime.config.device,
    idle_ttl_seconds: runtime.config.modelIdleTtlSeconds,
    memory_limit_bytes: runtime.memoryLimitBytes,
    discovered_models: inventory.discoveredModels,
    models,
  };
}
