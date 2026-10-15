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
