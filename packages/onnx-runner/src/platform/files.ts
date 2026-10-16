import { platform } from "#platform";

import type { PlatformFile } from "./port.js";

/** Streams a handle through SHA-256 from offset 0 to its end in reads of `blockBytes`; `beforeBlock` may throw. */
export async function hashHandle(
  handle: PlatformFile,
  blockBytes: number,
  beforeBlock?: () => void,
): Promise<{ bytes: number; sha256: string }> {
  const hash = await platform.createSha256();
  const block = new Uint8Array(blockBytes);
  let bytes = 0;
  for (;;) {
    beforeBlock?.();
    const { bytesRead } = await handle.read(block, 0, block.length, bytes);
    if (bytesRead === 0) {
      break;
    }
    hash.update(block.subarray(0, bytesRead));
    bytes += bytesRead;
  }
  return { bytes, sha256: hash.digest() };
}

/** SHA-256 of a string's UTF-8 bytes. */
export async function sha256OfText(text: string): Promise<string> {
  const hash = await platform.createSha256();
  hash.update(text);
  return hash.digest();
}

/** The whole remaining content of a handle as UTF-8 text. */
export async function readHandleText(handle: PlatformFile): Promise<string> {
  const { size } = await handle.stat();
  const bytes = new Uint8Array(size);
  let done = 0;
  while (done < size) {
    const { bytesRead } = await handle.read(bytes, done, size - done, done);
    if (bytesRead === 0) {
      break;
    }
    done += bytesRead;
  }
  return new TextDecoder().decode(bytes.subarray(0, done));
}
