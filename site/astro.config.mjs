// @ts-check
import { fileURLToPath } from "node:url";

import { defineConfig } from "astro/config";
import react from "@astrojs/react";
import starlight from "@astrojs/starlight";

import { beTheModelPlugin } from "./scripts/be-the-model-data.mjs";
import tableFocus from "./scripts/table-focus.mjs";

export default defineConfig({
  site: "https://stuffbucket.github.io",
  base: "/gliner-runner/",
  vite: {
    // The playground's runner worker is an ES module (it imports onnxruntime-web, which ships its own
    // WebAssembly loader and must not be pre-bundled).
    worker: { format: "es" },
    optimizeDeps: { exclude: ["onnxruntime-web"] },
    // Be the model's data: the real file, or the made-up fixture until it exists (see the plugin).
    plugins: [beTheModelPlugin(fileURLToPath(new URL(".", import.meta.url)))],
  },
  integrations: [
    react(),
    // Wide tables scroll sideways; this lets a keyboard reach and scroll them.
    tableFocus(),
    starlight({
      title: "GLiNER Runner",
      description: "Sort text into labels you choose, on your own computer.",
      social: [
        { icon: "github", label: "GitHub", href: "https://github.com/stuffbucket/gliner-runner" },
      ],
      editLink: { baseUrl: "https://github.com/stuffbucket/gliner-runner/edit/main/site/" },
      // The "On this page" panel opens and closes from an icon button, and closed it gives the page its room
      // (see the component).
      components: { TwoColumnContent: "./src/components/starlight/TwoColumnContent.astro" },
      // Keeps tables from scrolling sideways where it can, and gives the header's controls one height and
      // one quiet style (see the files).
      customCss: ["./src/styles/tables.css", "./src/styles/header.css"],
      // In the order a new visitor needs them: see it work, understand it, install it, use it, choose a model,
      // then more examples and ideas. Reference material comes last.
      sidebar: [
        { label: "Try it in your browser", link: "/playground/" },
        { label: "How it works", link: "/how-it-works/" },
        { label: "Be the model, be the kid", link: "/be-the-model/" },
        { label: "Get started", link: "/start/" },
        {
          label: "Use it",
          items: [
            { label: "Command line and HTTP", link: "/command-line-and-http/" },
            { label: "In a React app", link: "/use-in-react/" },
            { label: "From a coding agent", link: "/use-from-a-coding-agent/" },
          ],
        },
        {
          label: "Choose a model",
          items: [{ autogenerate: { directory: "models" } }, { label: "Accuracy", link: "/accuracy/" }, { label: "Evaluation", link: "/evaluation/" }],
        },
        {
          label: "Examples",
          items: [
            { label: "Overview", slug: "examples" },
            {
              label: "Sort incoming messages",
              items: [
                "examples/support-tickets",
                "examples/text-messages",
                "examples/which-language",
                "examples/other-languages",
              ],
            },
            {
              label: "Find what matters",
              items: ["examples/email-lines", "examples/urgent-or-can-wait"],
            },
            {
              label: "Flag things to check",
              items: ["examples/personal-information", "examples/moderation"],
            },
            { label: "Understand feedback", items: ["examples/reviews"] },
            { label: "Clean up lists", items: ["examples/keep-or-discard"] },
          ],
        },
        { label: "More ideas", link: "/more-ideas/" },
        {
          label: "Reference",
          items: [
            { label: "Docs", items: [{ autogenerate: { directory: "docs" } }] },
            { label: "Releases", link: "/releases/" },
          ],
        },
      ],
    }),
  ],
});
