import { copyFile, unlink } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const packageDirectory = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repositoryRoot = resolve(packageDirectory, "..", "..");
const noticeFiles = ["LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"];
const action = process.argv[2];

if (action === "prepare") {
  await Promise.all(
    noticeFiles.map((name) =>
      copyFile(resolve(repositoryRoot, name), resolve(packageDirectory, name)),
    ),
  );
} else if (action === "clean") {
  await Promise.all(
    noticeFiles.map((name) => unlink(resolve(packageDirectory, name))),
  );
} else {
  throw new Error("expected 'prepare' or 'clean'");
}
