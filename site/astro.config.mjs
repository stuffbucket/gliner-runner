// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

export default defineConfig({
  site: "https://stuffbucket.github.io",
  base: "/gliner-runner/",
  integrations: [
    starlight({
      title: "GLiNER Runner",
      description: "Local, reproducible GLiNER inference: curated models, parity evidence, and operations.",
      social: [
        { icon: "github", label: "GitHub", href: "https://github.com/stuffbucket/gliner-runner" },
      ],
      editLink: { baseUrl: "https://github.com/stuffbucket/gliner-runner/edit/main/site/" },
      sidebar: [
        { label: "Home", link: "/" },
        { label: "Models", items: [{ autogenerate: { directory: "models" } }] },
        { label: "Evaluation", link: "/evaluation/" },
        { label: "Docs", items: [{ autogenerate: { directory: "docs" } }] },
        { label: "Releases", link: "/releases/" },
      ],
    }),
  ],
});
