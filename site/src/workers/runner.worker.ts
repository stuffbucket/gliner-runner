// The playground's in-tab runner: gliner-runner itself, in a dedicated worker (see @gliner-runner/onnx-runner-web).
import { serveRunner } from "@gliner-runner/onnx-runner-web/host";

serveRunner({ templatesBaseUrl: `${import.meta.env.BASE_URL.replace(/\/?$/, "/")}runner-templates/` });
