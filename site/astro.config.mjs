// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

export default defineConfig({
  site: "https://stuffbucket.github.io",
  base: "/gliner-runner/",
  integrations: [
    starlight({
      title: "GLiNER Runner",
      description: "Sort text into labels you choose, on your own computer.",
      social: [
        { icon: "github", label: "GitHub", href: "https://github.com/stuffbucket/gliner-runner" },
      ],
      editLink: { baseUrl: "https://github.com/stuffbucket/gliner-runner/edit/main/site/" },
      sidebar: [
        { label: "Home", link: "/" },
        { label: "How it works", link: "/how-it-works/" },
        { label: "Examples", items: [{ autogenerate: { directory: "examples" } }] },
        { label: "Use it in a React app", link: "/use-in-react/" },
        { label: "More ideas", link: "/more-ideas/" },
        {
          label: "Reference",
          items: [
            { label: "Models", items: [{ autogenerate: { directory: "models" } }] },
            { label: "Evaluation", link: "/evaluation/" },
            { label: "Docs", items: [{ autogenerate: { directory: "docs" } }] },
            { label: "Releases", link: "/releases/" },
          ],
        },
      ],
    }),
  ],
});
