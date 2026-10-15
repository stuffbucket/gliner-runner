import assert from "node:assert/strict";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { runCorsCommand } from "../src/cors-cli.js";
import { DEFAULT_CORS_ORIGINS, userSettingsPath } from "../src/user-settings.js";

interface Result {
  code: number;
  stdout: string;
  stderr: string;
}

function setup(): { cors: (...argv: string[]) => Promise<Result>; file: string; saved: () => unknown } {
  const env = { XDG_CONFIG_HOME: mkdtempSync(join(tmpdir(), "gliner-config-")) };
  const file = join(env.XDG_CONFIG_HOME, "gliner-runner", "settings.json");
  const cors = async (...argv: string[]): Promise<Result> => {
    let stdout = "";
    let stderr = "";
    const code = await runCorsCommand(argv, {
      env,
      stdout: (text) => (stdout += text),
      stderr: (text) => (stderr += text),
    });
    return { code, stdout, stderr };
  };
  return { cors, file, saved: () => JSON.parse(readFileSync(file, "utf8")) as unknown };
}

const listed = async (cors: (...argv: string[]) => Promise<Result>): Promise<string[]> => {
  const result = await cors("list", "--json");
  assert.equal(result.code, 0, result.stderr);
  const parsed = JSON.parse(result.stdout) as { origins: { id: number; origin: string }[] };
  parsed.origins.forEach((entry, index) => assert.equal(entry.id, index + 1));
  return parsed.origins.map((entry) => entry.origin);
};

test("settings.json lives under XDG_CONFIG_HOME, ~/.config, or %APPDATA% on Windows", () => {
  assert.equal(userSettingsPath({ XDG_CONFIG_HOME: "/xdg" }, "linux"), join("/xdg", "gliner-runner", "settings.json"));
  assert.equal(userSettingsPath({ XDG_CONFIG_HOME: "/xdg" }, "darwin"), join("/xdg", "gliner-runner", "settings.json"));
  assert.match(userSettingsPath({}, "darwin"), /\.config[/\\]gliner-runner[/\\]settings\.json$/);
  assert.equal(userSettingsPath({ APPDATA: "C:\\AppData" }, "win32"), join("C:\\AppData", "gliner-runner", "settings.json"));
});

test("list shows the numbered defaults without writing a file", async () => {
  const { cors, file } = setup();
  const result = await cors("list");
  assert.equal(result.code, 0, result.stderr);
  assert.match(result.stdout, /^Defaults \(nothing saved yet; the first change creates .*settings\.json\)\n\n/);
  assert.match(result.stdout, /\[ 1\] http:\/\/localhost:3000\n/);
  assert.match(result.stdout, /\[ 5\] http:\/\/localhost:5173\n/);
  const json = JSON.parse((await cors("list", "--json")).stdout) as { path: string; saved: boolean };
  assert.equal(json.path, file);
  assert.equal(json.saved, false);
  assert.deepEqual(await listed(cors), DEFAULT_CORS_ORIGINS);
  assert.equal(existsSync(file), false);
});

test("add appends after the defaults, saves the file and skips duplicates", async () => {
  const { cors, saved } = setup();
  const added = await cors("add", "https://stuffbucket.github.io", "http://localhost:5173");
  assert.equal(added.code, 0, added.stderr);
  assert.match(added.stdout, /Already allowed: \[ 5\] http:\/\/localhost:5173\n/);
  assert.match(added.stdout, /Added \[ 7\] https:\/\/stuffbucket\.github\.io\n/);
  assert.match(added.stdout, /Restart 'gliner-runner serve' to apply/);
  assert.deepEqual(saved(), { cors_origins: [...DEFAULT_CORS_ORIGINS, "https://stuffbucket.github.io"] });
  assert.match((await cors("list")).stdout, /^Saved in .*settings\.json\n/);
  assert.equal((await cors("show", "7")).stdout, "https://stuffbucket.github.io\n");
});

test("set replaces by number, delete removes by number or origin, and numbers shift", async () => {
  const { cors } = setup();
  assert.equal((await cors("set", "1", "http://localhost:3001")).code, 0);
  assert.equal((await listed(cors))[0], "http://localhost:3001");

  const deleted = await cors("delete", "6", "2", "http://localhost:3001");
  assert.equal(deleted.code, 0, deleted.stderr);
  assert.match(deleted.stdout, /Deleted \[ 1\] http:\/\/localhost:3001\nDeleted \[ 2\] http:\/\/localhost:4173\nDeleted \[ 6\] http:\/\/localhost:8080\n/);
  assert.deepEqual(await listed(cors), ["http://localhost:4200", "http://localhost:4321", "http://localhost:5173"]);
  assert.equal((await cors("show", "1")).stdout, "http://localhost:4200\n");
});

test("an emptied list stays empty and reset restores the defaults", async () => {
  const { cors, saved } = setup();
  assert.equal((await cors("delete", "1", "2", "3", "4", "5", "6")).code, 0);
  assert.deepEqual(saved(), { cors_origins: [] });
  assert.match((await cors("list")).stdout, /No origins: web pages cannot call the runner/);
  assert.equal((await cors("reset")).code, 0);
  assert.deepEqual(saved(), { cors_origins: DEFAULT_CORS_ORIGINS });
});

test("writes keep other keys in settings.json", async () => {
  const { cors, file, saved } = setup();
  mkdirSync(join(file, ".."), { recursive: true });
  writeFileSync(file, JSON.stringify({ future_setting: { a: 1 }, cors_origins: ["http://a.test"] }));
  assert.equal((await cors("add", "http://b.test")).code, 0);
  assert.deepEqual(saved(), { future_setting: { a: 1 }, cors_origins: ["http://a.test", "http://b.test"] });
});

test("bad numbers, bad origins and wrong arity are usage errors that change nothing", async () => {
  const { cors, file } = setup();
  const cases: [string[], RegExp][] = [
    [["show", "0"], /no origin number 0: use 1 to 6/],
    [["show", "7"], /no origin number 7: use 1 to 6/],
    [["show", "x"], /no origin number x/],
    [["delete", "9"], /no origin number 9/],
    [["delete", "http://nope.test"], /http:\/\/nope\.test is not in the list/],
    [["add", "http://localhost:5173/"], /origin must list origins like/],
    [["add", "*"], /origin must list origins like/],
    [["set", "1", "http://localhost:5173"], /already allowed as \[ 5\]/],
    [["set", "1"], /expected: cors set <n> <origin>/],
    [["add"], /expected: cors add <origin>/],
    [["list", "extra"], /expected: cors list/],
    [["add", "--json", "http://a.test"], /Unknown option '--json'/],
    [["frobnicate"], /unknown cors command: frobnicate/],
  ];
  for (const [argv, message] of cases) {
    const result = await cors(...argv);
    assert.equal(result.code, 2, argv.join(" "));
    assert.match(result.stderr, message, argv.join(" "));
    assert.match(result.stderr, /Usage: gliner-runner cors/, argv.join(" "));
  }
  assert.equal(existsSync(file), false);
});

test("a malformed settings.json is reported with its path", async () => {
  const { cors, file } = setup();
  mkdirSync(join(file, ".."), { recursive: true });
  for (const [content, message] of [
    ["{not json", /settings\.json is not valid JSON/],
    ["[]", /settings\.json must hold a JSON object/],
    [JSON.stringify({ cors_origins: "http://a.test" }), /cors_origins must be a list of origin strings/],
  ] as const) {
    writeFileSync(file, content);
    const result = await cors("list");
    assert.equal(result.code, 1);
    assert.match(result.stderr, message);
  }
});

test("help prints usage; no command is a usage error", async () => {
  const { cors } = setup();
  const help = await cors("--help");
  assert.equal(help.code, 0);
  assert.match(help.stdout, /^Usage: gliner-runner cors <command>/);
  assert.equal((await cors()).code, 2);
});
