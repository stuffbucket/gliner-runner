/**
 * One known input per model and the logits onnxruntime-node (the parity-tested CPU path) produces for it. A GPU
 * session must reproduce them within the parity tolerance before it is used.
 *
 * decide-340m: "I was charged twice for my last order." with the labels billing, support, shipping.
 */
export interface Probe {
  inputIds: readonly number[];
  attentionMask: readonly number[];
  labelPositions: readonly number[];
  labelMask: readonly number[];
  logits: readonly number[];
}

export const PROBES: Readonly<Record<string, Probe>> = {
  "decide-340m": {
    inputIds: [
      287, 128003, 2509, 287, 128007, 9833, 128007, 523, 128007, 2741, 1263, 1263, 128002, 584, 284, 2824, 2736, 270,
      312, 437, 556, 323,
    ],
    attentionMask: Array.from({ length: 22 }, () => 1),
    labelPositions: [4, 6, 8],
    labelMask: [1, 1, 1],
    logits: [1.004750370979309, -1.903451681137085, -1.6455787420272827],
  },
};
