Snippets for stuffbucket/homebrew-tap `README.md`. Do not copy this file itself.

## Quick Starts table: add this row inside `<tbody>`

```html
    <tr style="border-bottom: 1px solid #d0d7de;">
      <td style="padding: 12px 16px;"><a href="apps/gliner-runner.md">gliner-runner</a></td>
      <td style="padding: 12px 16px;">Local GLiNER2 inference runner (self-contained ONNX runtime, no Python)</td>
    </tr>
```

## Manual Installation: add this section after Bladerunner

````markdown
### Gliner Runner

```bash
  curl -fsSL https://raw.githubusercontent.com/stuffbucket/homebrew-tap/refs/heads/main/install/gliner-runner | bash && eval "$(brew shellenv)"
```

Release candidates: `brew install stuffbucket/tap/gliner-runner-rc` (conflicts with the stable formula).
````
