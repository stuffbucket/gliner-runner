export class GlinerRunnerError extends Error {
  constructor(message: string) {
    super(message);
    this.name = new.target.name;
  }
}

/** The explicitly requested backend is not registered in this process. */
export class BackendUnavailableError extends GlinerRunnerError {}

/** A backend does not implement the requested operation, precision, device or model. */
export class UnsupportedCapabilityError extends GlinerRunnerError {}

/** An inference schema failed backend-specific semantic validation. */
export class InvalidSchemaError extends GlinerRunnerError {}

/** The bounded scheduler queue has reached capacity. */
export class QueueFullError extends GlinerRunnerError {}

/** The runtime or scheduler is not accepting work. */
export class RunnerClosedError extends GlinerRunnerError {}

/** The caller abandoned the request before it produced a result. */
export class RequestCancelledError extends GlinerRunnerError {}

/** The request body exceeds the configured size limit. */
export class RequestTooLargeError extends GlinerRunnerError {
  readonly limitBytes: number;

  constructor(limitBytes: number) {
    super(`request body exceeds the limit of ${limitBytes} bytes`);
    this.limitBytes = limitBytes;
  }
}

/** A model artifact failed catalog validation. */
export class ModelIntegrityError extends GlinerRunnerError {}

/** A catalog model is not installed in its configured location. */
export class ModelDownloadRequiredError extends ModelIntegrityError {
  readonly model: string;
  readonly revision: string;
  readonly destination: string;

  constructor(model: string, revision: string, destination: string, reason?: string) {
    super(
      `model '${model}' at revision ${revision} is not installed; ` +
        (reason === undefined
          ? "install it with the ONNX model preparation tool"
          : `automatic installation failed: ${reason}`),
    );
    this.model = model;
    this.revision = revision;
    this.destination = destination;
  }
}

/** A download request was refused before a job was created (HTTP 422 `invalid_download_request`). */
export class InvalidDownloadRequestError extends GlinerRunnerError {}

/** No download job has the requested id (HTTP 404 `download_job_not_found`). */
export class DownloadJobNotFoundError extends GlinerRunnerError {}

/** A model cannot fit within the configured memory pressure limit. */
export class ModelMemoryLimitError extends GlinerRunnerError {
  readonly model: string;
  readonly profile: string;
  readonly limitBytes: number;
  readonly requiredBytes: number;
  readonly observedBytes: number | null;

  constructor(options: {
    model: string;
    profile: string;
    limitBytes: number;
    requiredBytes: number;
    observedBytes?: number;
  }) {
    const measurement =
      options.observedBytes === undefined
        ? `estimated minimum is ${options.requiredBytes} bytes`
        : `observed ${options.observedBytes} bytes after load`;
    super(
      `model '${options.model}' with profile ${options.profile} exceeds the configured ` +
        `memory limit of ${options.limitBytes} bytes: ${measurement}; no fallback was attempted`,
    );
    this.model = options.model;
    this.profile = options.profile;
    this.limitBytes = options.limitBytes;
    this.requiredBytes = options.requiredBytes;
    this.observedBytes = options.observedBytes ?? null;
  }
}

export interface ErrorDetail {
  code: string;
  message: string;
  retryable: boolean;
  [extra: string]: unknown;
}

export interface HttpError {
  status: number;
  detail: ErrorDetail | string | readonly unknown[];
  headers?: Record<string, string>;
}

export function envelope(
  status: number,
  code: string,
  message: string,
  retryable: boolean,
  extra: Record<string, unknown> = {},
  headers?: Record<string, string>,
): HttpError {
  return {
    status,
    detail: { code, message, retryable, ...extra },
    ...(headers === undefined ? {} : { headers }),
  };
}

/** Maps a domain error to its HTTP envelope; `undefined` means the error is unexpected. */
export function domainHttpError(error: unknown): HttpError | undefined {
  // Order matters: ModelDownloadRequiredError is a ModelIntegrityError.
  if (error instanceof QueueFullError) {
    return envelope(429, "queue_full", error.message, true, {}, { "retry-after": "1" });
  }
  if (error instanceof RequestTooLargeError) {
    // The unread remainder of the body is discarded, so the connection is closed after the reply.
    return envelope(413, "request_too_large", error.message, false, { limit_bytes: error.limitBytes }, {
      connection: "close",
    });
  }
  if (error instanceof BackendUnavailableError || error instanceof UnsupportedCapabilityError) {
    return envelope(409, "unsupported_configuration", error.message, false);
  }
  if (error instanceof ModelDownloadRequiredError) {
    return envelope(409, "model_download_required", error.message, false, {
      event: {
        type: "model_download_required",
        model: error.model,
        revision: error.revision,
        destination: error.destination,
        download_endpoint: `/v1/models/${error.model}/downloads`,
      },
    });
  }
  if (error instanceof InvalidDownloadRequestError) {
    return envelope(422, "invalid_download_request", error.message, false);
  }
  if (error instanceof DownloadJobNotFoundError) {
    return envelope(404, "download_job_not_found", `'${error.message}'`, false);
  }
  if (error instanceof ModelMemoryLimitError) {
    return envelope(409, "model_memory_limit", error.message, false, {
      model: error.model,
      profile: error.profile,
      limit_bytes: error.limitBytes,
      required_bytes: error.requiredBytes,
      observed_bytes: error.observedBytes,
      attempted_load_unloaded: error.observedBytes !== null,
    });
  }
  if (error instanceof InvalidSchemaError) {
    return envelope(422, "invalid_schema", error.message, false);
  }
  if (error instanceof ModelIntegrityError) {
    return envelope(422, "model_integrity", error.message, false);
  }
  if (error instanceof RunnerClosedError) {
    return envelope(503, "runtime_closed", error.message, true);
  }
  return undefined;
}

export function internalError(): HttpError {
  return envelope(500, "internal_error", "The server failed to process the request", false);
}

export function catalogRefreshFailed(error: unknown): HttpError {
  const message = error instanceof Error ? error.message : String(error);
  return envelope(502, "catalog_refresh_failed", message, true);
}
